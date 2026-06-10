"""複数iPhone → Mac のライブ議事録サーバ.

各iPhoneがブラウザでURLを開き、名前を入れて「開始」 → マイク音声がWiFiでMacへ。
Macは各接続を1人=1チャンネルとして、数秒ごとに文字起こしし、
時系列の議事録をターミナルに表示＋ビューア(/view)に配信する。

前提: iOS Safariはマイク利用にHTTPSが必須。自己署名証明書で立てるので、
各iPhoneで初回に証明書の警告を「許可」してください。

準備(Mac):
  uv add aiohttp sounddevice            # sounddeviceは不要だが既存依存と整合
  # ASRは既存の mlx-whisper / whisper をそのまま使用

使い方:
  uv run python offshelf/meeting_server.py --engine mlx-whisper --lang ja
  # 表示される https://<MacのIP>:8443 を各iPhoneのSafariで開く
  # 司会のMacでは https://localhost:8443/view で議事録を見られる

注意: v1は漏れ込みゲートなし(各iPhone=各話者の独立処理)。
      iPhoneが十分離れていれば実用的。漏れが気になれば後でゲートを足せる。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import ssl
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SR = 16000
clients = {}     # ws -> state
viewers = set()


def make_asr(engine, lang, mlx_model):
    if engine == "mlx-whisper":
        import mlx_whisper

        def f(audio):
            r = mlx_whisper.transcribe(audio, path_or_hf_repo=mlx_model,
                                       language=lang, condition_on_previous_text=False)
            return [s["text"].strip() for s in r.get("segments", []) if s["text"].strip()]
        return f
    else:
        from spkattr.asr import Transcriber, transcribe_words
        tr = Transcriber("base", language=lang)

        def f(audio):
            words = transcribe_words(tr, audio, SR)
            out, cur = [], []
            for w, s, e in words:
                if cur and s - cur[-1][2] > 0.8:
                    out.append(" ".join(x[0] for x in cur)); cur = []
                cur.append((w, s, e))
            if cur:
                out.append(" ".join(x[0] for x in cur))
            return out
        return f


def resample_to_16k(x, sr):
    if sr == SR:
        return x
    n = int(len(x) * SR / sr)
    return np.interp(np.linspace(0, len(x), n, endpoint=False), np.arange(len(x)), x).astype(np.float32)


def ensure_cert(certdir):
    os.makedirs(certdir, exist_ok=True)
    cert = os.path.join(certdir, "cert.pem"); key = os.path.join(certdir, "key.pem")
    if not (os.path.exists(cert) and os.path.exists(key)):
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", key, "-out", cert, "-days", "365",
            "-subj", "/CN=meeting.local",
        ], check=True)
    return cert, key


def local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


async def ws_handler(request):
    from aiohttp import web, WSMsgType
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    name = request.query.get("name", "話者")
    st = {"name": name, "buf": np.zeros(0, np.float32), "total": 0,
          "emitted": set(), "sr": SR}
    clients[ws] = st
    print(f"# 接続: {name}", flush=True)
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                x = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0
                if st["sr"] != SR:
                    x = resample_to_16k(x, st["sr"])
                st["buf"] = np.concatenate([st["buf"], x])
                st["total"] += len(x)
            elif msg.type == WSMsgType.TEXT:
                import json
                try:
                    cfg = json.loads(msg.data)
                    if "sampleRate" in cfg:
                        st["sr"] = int(cfg["sampleRate"])
                    if "name" in cfg:
                        st["name"] = cfg["name"]
                except Exception:
                    pass
    finally:
        clients.pop(ws, None)
        print(f"# 切断: {name}", flush=True)
    return ws


async def view_ws(request):
    from aiohttp import web
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    viewers.add(ws)
    try:
        async for _ in ws:
            pass
    finally:
        viewers.discard(ws)
    return ws


async def broadcast(item):
    import json
    dead = []
    for v in list(viewers):
        try:
            await v.send_str(json.dumps(item))
        except Exception:
            dead.append(v)
    for v in dead:
        viewers.discard(v)


async def processor(app):
    """壁時計で chunk 秒ごとに全チャンネルをまとめて処理し、
    エネルギー優勢ゲート（本人の口に近いマイクが最大）で漏れを抑制する。
    バラバラのiPhoneでも、実時間で流れ込む各バッファの末尾を比較すればよい。
    """
    asr = app["asr"]; chunk = app["chunk"]; ctx = app["ctx"]; vad_on = app["vad"]
    margin = app["margin_db"]
    loop = asyncio.get_event_loop()
    win = int((chunk + ctx) * SR); newn = int(chunk * SR)
    while True:
        await asyncio.sleep(chunk)
        items = list(clients.items())
        if not items:
            continue
        infos = []
        for ws, st in items:
            buf = st["buf"]
            tail = buf[-win:].copy() if len(buf) else np.zeros(0, np.float32)
            newp = buf[-newn:] if len(buf) >= newn else buf
            rms = float(np.sqrt((newp ** 2).mean())) if len(newp) else 0.0
            infos.append([ws, st, tail, newp, rms])
        emax = max((i[4] for i in infos), default=0.0)
        for ws, st, tail, newp, rms in infos:
            if rms < 2e-3:
                continue
            # エネルギー優勢ゲート: 最大chより margin dB 以上小さい = 漏れ → 抑制
            if emax > 0 and 20 * np.log10(rms / emax + 1e-9) < -margin:
                continue
            if vad_on:
                from spkattr.vad import energy_vad
                if len(newp) == 0 or energy_vad(newp, SR).mean() <= 0.04:
                    continue
            texts = await loop.run_in_executor(None, asr, tail)
            tnow = st.get("total", len(st["buf"])) / SR
            for t in texts:
                t = t.strip()
                if not t or t in st["emitted"]:
                    continue
                st["emitted"].add(t)
                item = {"t": round(tnow, 1), "name": st["name"], "text": t}
                print(f"[{tnow:6.1f}s] {st['name']}: {t}", flush=True)
                await broadcast(item)
            # メモリ節約: 直近30秒だけ保持
            if len(st["buf"]) > 30 * SR:
                st["buf"] = st["buf"][-30 * SR:]


async def on_start(app):
    app["proc_task"] = asyncio.create_task(processor(app))


async def on_cleanup(app):
    app["proc_task"].cancel()


def build_app(args):
    from aiohttp import web
    app = web.Application()
    app["asr"] = make_asr(args.engine, args.lang, args.mlx_model)
    app["chunk"] = args.chunk_s
    app["ctx"] = args.context_s
    app["vad"] = args.vad
    app["margin_db"] = args.margin_db
    app.router.add_get("/", lambda r: web.Response(text=CLIENT_HTML, content_type="text/html"))
    app.router.add_get("/view", lambda r: web.Response(text=VIEW_HTML, content_type="text/html"))
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/viewws", view_ws)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_cleanup)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["mlx-whisper", "whisper"], default="mlx-whisper")
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--mlx-model", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--chunk-s", type=float, default=4.0)
    ap.add_argument("--context-s", type=float, default=1.5)
    ap.add_argument("--no-vad", dest="vad", action="store_false")
    ap.add_argument("--margin-db", type=float, default=6.0,
                    help="エネルギー優勢ゲート: 最大chよりこのdB以上小さい声は漏れとして抑制")
    ap.add_argument("--port", type=int, default=8443)
    args = ap.parse_args()

    from aiohttp import web
    print("# モデル読み込み中…", flush=True)
    app = build_app(args)
    cert, key = ensure_cert(os.path.join(os.path.dirname(__file__), "_cert"))
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(cert, key)
    ip = local_ip()
    print(f"\n# 各iPhoneのSafariで開く →  https://{ip}:{args.port}")
    print(f"# 議事録ビューア(Mac) →        https://localhost:{args.port}/view")
    print("# ※初回は証明書の警告が出ます。『詳細→このサイトを閲覧』で許可してください。\n", flush=True)
    web.run_app(app, host="0.0.0.0", port=args.port, ssl_context=ssl_ctx, print=None)


CLIENT_HTML = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>会議マイク</title><style>
 body{font-family:-apple-system,sans-serif;max-width:480px;margin:40px auto;padding:0 20px;text-align:center}
 input{font-size:18px;padding:10px;width:80%;margin:10px 0}
 button{font-size:20px;padding:14px 28px;border-radius:10px;border:none;background:#0a84ff;color:#fff}
 #st{margin-top:20px;font-size:16px;color:#555}</style></head><body>
<h2>会議マイク</h2>
<p>名前を入れて「開始」を押してください</p>
<input id="name" placeholder="あなたの名前" />
<br><button id="go">開始</button>
<div id="st">未接続</div>
<script>
let ws, ctx, proc, stream;
document.getElementById('go').onclick = async () => {
  const name = document.getElementById('name').value || '話者';
  const stEl = document.getElementById('st');
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true}});
  } catch(e){ stEl.textContent='マイク許可が必要です: '+e; return; }
  ctx = new (window.AudioContext||window.webkitAudioContext)();
  ws = new WebSocket(`wss://${location.host}/ws?name=`+encodeURIComponent(name));
  ws.binaryType='arraybuffer';
  ws.onopen = () => { ws.send(JSON.stringify({sampleRate: ctx.sampleRate, name})); stEl.textContent='接続中（話してください）'; };
  ws.onclose = () => stEl.textContent='切断されました';
  const src = ctx.createMediaStreamSource(stream);
  proc = ctx.createScriptProcessor(4096,1,1);
  src.connect(proc); proc.connect(ctx.destination);
  proc.onaudioprocess = e => {
    if(!ws||ws.readyState!==1) return;
    const f = e.inputBuffer.getChannelData(0);
    const i16 = new Int16Array(f.length);
    for(let i=0;i<f.length;i++){ let s=Math.max(-1,Math.min(1,f[i])); i16[i]= s<0? s*32768 : s*32767; }
    ws.send(i16.buffer);
  };
};
</script></body></html>"""


VIEW_HTML = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>議事録</title><style>
 body{font-family:-apple-system,"Hiragino Sans",sans-serif;max-width:820px;margin:24px auto;padding:0 16px;line-height:1.6}
 .ev{display:flex;gap:10px;padding:6px 0;border-bottom:1px solid #eef0f3}
 .ts{color:#aab;font-size:12px;width:54px;flex:none;text-align:right;padding-top:2px}
 .sp{font-weight:600;width:90px;flex:none} .tx{flex:1}</style></head><body>
<h2>ライブ議事録</h2><div id="log"></div>
<script>
const colors=['#0a84ff','#ff9500','#34c759','#af52de','#ff2d55','#5ac8fa'];
const names={}; let n=0;
const ws=new WebSocket(`wss://${location.host}/viewws`);
ws.onmessage=e=>{const d=JSON.parse(e.data);
 if(!(d.name in names)) names[d.name]=colors[(n++)%colors.length];
 const div=document.createElement('div'); div.className='ev';
 div.innerHTML=`<span class="ts">${d.t}s</span><span class="sp" style="color:${names[d.name]}">${d.name}</span><span class="tx">${d.text}</span>`;
 document.getElementById('log').appendChild(div); window.scrollTo(0,document.body.scrollHeight);};
</script></body></html>"""


if __name__ == "__main__":
    main()
