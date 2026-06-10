"""1マイク + Soniox リアルタイム議事録ツール（日本語・話者分離内蔵）.

1本のマイクで「誰が・何を」をライブ取得する本線ツール。
Sonioxのストリーミング(WebSocket)に音声を流し、speaker付きトークンが返るので、
多マイク・ゲート・同期なしで who-said-what が出る。

機能:
  - 話者ごとに色分けしたライブ表示（確定前のテキストは薄く表示）
  - Markdown議事録 + HTML を transcripts/ に自動保存（発話確定ごと＝クラッシュ安全）
  - HTMLはブラウザで開くとライブ中2秒ごと自動更新（open transcripts/～.html）
  - 実行中に「1=松井」と打つと話者1を実名にマッピング（過去の発言も遡って置換）

準備(Mac):
  uv add websockets sounddevice
  export SONIOX_API_KEY=...   # https://console.soniox.com で取得

使い方:
  uv run python offshelf/live_soniox.py            # 実マイクでライブ
  uv run python offshelf/live_soniox.py --wav offshelf/ami_raw/mic0.wav  # ファイル擬似ライブ
  実行中: 「1=松井」Enter で話者名設定 / Ctrl+C で終了（保存先を表示）
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import queue
import re
import sys
import threading
import time

import numpy as np

SR = 16000
WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"

RESET = "\x1b[0m"
DIM = "\x1b[2m"
CLEAR_LINE = "\r\x1b[K"
PALETTE = ["\x1b[36m", "\x1b[33m", "\x1b[35m", "\x1b[32m", "\x1b[34m", "\x1b[91m"]
HTML_PALETTE = ["#0e7490", "#a16207", "#7e22ce", "#15803d", "#1d4ed8", "#dc2626"]

HTML_TMPL = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
{refresh}<title>議事録 {title}</title>
<style>
body {{ font-family: -apple-system, "Hiragino Sans", sans-serif; max-width: 760px;
       margin: 2rem auto; padding: 0 1rem; background: #fafafa; color: #1f2937; }}
h1 {{ font-size: 1.2rem; }} .meta {{ color: #6b7280; font-size: .85rem; }}
.u {{ margin: .5rem 0; padding: .55rem .8rem; background: #fff; border-radius: 10px;
     border: 1px solid #e5e7eb; }}
.u .who {{ font-weight: 700; margin-right: .5em; }}
.u .ts {{ color: #9ca3af; font-size: .8rem; margin-right: .6em; font-variant-numeric: tabular-nums; }}
.live {{ color: #16a34a; font-size: .8rem; }}
</style></head><body>
<h1>議事録 {title}</h1>
<p class="meta">{status} / 話者: {speakers}</p>
{body}
<script>window.scrollTo(0, document.body.scrollHeight);</script>
</body></html>
"""


def fmt_ts(ms: int | None) -> str:
    if ms is None:
        return "--:--"
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--model", default="stt-rt-v4")
    ap.add_argument("--wav", default=None, help="指定で実マイクの代わりにファイル擬似ライブ")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="保存先mdファイル（省略時 transcripts/日時.md）")
    ap.add_argument("--no-open", action="store_true", help="ブラウザを自動で開かない")
    args = ap.parse_args()

    api_key = os.environ.get("SONIOX_API_KEY")
    if not api_key:
        raise SystemExit("環境変数 SONIOX_API_KEY を設定してください（https://console.soniox.com）")

    try:
        from websockets.sync.client import connect
    except ImportError:
        raise SystemExit("uv add websockets を実行してください")

    config = {
        "api_key": api_key,
        "model": args.model,
        "language_hints": [args.lang],
        "enable_speaker_diarization": True,
        "enable_endpoint_detection": True,
        "audio_format": "pcm_s16le",
        "sample_rate": SR,
        "num_channels": 1,
    }

    started = datetime.datetime.now()
    if args.out:
        out_path = args.out
    else:
        os.makedirs("transcripts", exist_ok=True)
        out_path = os.path.join("transcripts", started.strftime("%Y-%m-%d_%H%M") + ".md")
    html_path = os.path.splitext(out_path)[0] + ".html"

    # --- 状態 ---
    names: dict[str, str] = {}          # speaker id -> 実名
    colors: dict[str, str] = {}         # speaker id -> ANSI色（出現順に割当）
    records: list[dict] = []            # 確定発話 {"ms", "speaker", "text"}
    state_lock = threading.Lock()

    def disp_name(sp) -> str:
        sp = str(sp)
        return names.get(sp, f"話者{sp}")

    def color_of(sp) -> str:
        sp = str(sp)
        if sp not in colors:
            colors[sp] = PALETTE[len(colors) % len(PALETTE)]
        return colors[sp]

    def write_md():
        with state_lock:
            lines = [
                f"# 議事録 {started.strftime('%Y-%m-%d %H:%M')}",
                "",
                f"話者: " + (", ".join(f"{disp_name(s)}" for s in sorted(colors)) or "（未検出）"),
                "",
            ]
            for r in records:
                lines.append(f"- **[{fmt_ts(r['ms'])}] {disp_name(r['speaker'])}**: {r['text']}")
            tmp = out_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(tmp, out_path)

    def write_html(live: bool = True):
        import html as _html
        with state_lock:
            parts = []
            for r in records:
                sp = str(r["speaker"])
                idx = list(colors).index(sp) if sp in colors else 0
                c = HTML_PALETTE[idx % len(HTML_PALETTE)]
                parts.append(
                    f'<div class="u"><span class="ts">{fmt_ts(r["ms"])}</span>'
                    f'<span class="who" style="color:{c}">{_html.escape(disp_name(sp))}</span>'
                    f'{_html.escape(r["text"])}</div>'
                )
            doc = HTML_TMPL.format(
                refresh='<meta http-equiv="refresh" content="2">' if live else "",
                title=started.strftime("%Y-%m-%d %H:%M"),
                status='<span class="live">● ライブ（2秒ごと自動更新）</span>' if live else "終了",
                speakers=_html.escape(", ".join(disp_name(s) for s in sorted(colors)) or "（未検出）"),
                body="\n".join(parts) or '<p class="meta">（まだ発話なし）</p>',
            )
            tmp = html_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(doc)
            os.replace(tmp, html_path)

    def save(live: bool = True):
        write_md()
        write_html(live)

    def print_line(text: str):
        sys.stdout.write(CLEAR_LINE + text + "\n")
        sys.stdout.flush()

    def show_partial(sp, text: str):
        if not text.strip():
            sys.stdout.write(CLEAR_LINE)
        else:
            cols = os.get_terminal_size().columns if sys.stdout.isatty() else 120
            line = f"{disp_name(sp)}: {text.strip()}"
            sys.stdout.write(CLEAR_LINE + DIM + line[-(cols - 2):] + RESET)
        sys.stdout.flush()

    # --- 音声入力（マイク or ファイル）→ audio_q ---
    audio_q: "queue.Queue[bytes | None]" = queue.Queue()
    stop = threading.Event()

    def from_mic():
        import sounddevice as sd

        def cb(indata, frames, t, status):
            pcm = (np.clip(indata[:, 0], -1, 1) * 32767).astype("<i2").tobytes()
            audio_q.put(pcm)
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            device=args.device, callback=cb, blocksize=int(SR * 0.1)):
            while not stop.is_set():
                time.sleep(0.1)
        audio_q.put(None)

    def from_wav():
        import librosa
        y, _ = librosa.load(args.wav, sr=SR)
        step = int(SR * 0.12)
        for i in range(0, len(y), step):
            if stop.is_set():
                break
            pcm = (np.clip(y[i:i + step], -1, 1) * 32767).astype("<i2").tobytes()
            audio_q.put(pcm)
            time.sleep(0.12)
        audio_q.put(None)

    # --- 実行中コマンド: 「1=松井」で話者名マッピング ---
    def stdin_commands():
        while not stop.is_set():
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            m = re.match(r"^\s*(\S+?)\s*=\s*(.+?)\s*$", line)
            if m:
                with state_lock:
                    names[m.group(1)] = m.group(2)
                save()
                print_line(f"# 話者{m.group(1)} → {m.group(2)}（過去の発言も置換済み）")
            elif line.strip():
                print_line("# コマンド: 「話者番号=名前」（例: 1=松井）/ Ctrl+Cで終了")

    print("# Sonioxに接続中…", flush=True)
    with connect(WS_URL) as ws:
        ws.send(json.dumps(config))
        threading.Thread(target=from_wav if args.wav else from_mic, daemon=True).start()
        threading.Thread(target=stdin_commands, daemon=True).start()

        def sender():
            while True:
                pcm = audio_q.get()
                if pcm is None:
                    ws.send("")   # 終端
                    break
                ws.send(pcm)
        threading.Thread(target=sender, daemon=True).start()

        save()
        print("# 開始。話してください（「1=松井」で話者名設定 / Ctrl+Cで終了）", flush=True)
        print(f"# 保存先: {out_path}", flush=True)
        print(f"# ブラウザ表示: open {html_path}（ライブ中は2秒ごと自動更新）\n", flush=True)
        if not args.no_open:
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(html_path))

        cur_speaker = None
        cur_text = ""
        cur_ms: int | None = None

        def flush():
            nonlocal cur_text, cur_ms
            if cur_text.strip():
                with state_lock:
                    records.append({"ms": cur_ms, "speaker": str(cur_speaker), "text": cur_text.strip()})
                c = color_of(cur_speaker)
                print_line(f"{c}[{fmt_ts(cur_ms)}] {disp_name(cur_speaker)}{RESET}: {cur_text.strip()}")
                save()
            cur_text = ""
            cur_ms = None

        try:
            while True:
                res = json.loads(ws.recv())
                if res.get("error_code") is not None:
                    print_line(f"# エラー: {res['error_code']} - {res.get('error_message')}")
                    break
                partial = ""
                partial_sp = cur_speaker
                for token in res.get("tokens", []):
                    text = token.get("text") or ""
                    if text == "<end>":
                        flush()
                        continue
                    if not text:
                        continue
                    if token.get("is_final"):
                        sp = token.get("speaker")
                        if sp != cur_speaker:
                            flush()
                            cur_speaker = sp
                        if cur_ms is None:
                            cur_ms = token.get("start_ms")
                        cur_text += text
                    else:
                        partial += text
                        partial_sp = token.get("speaker") or partial_sp
                show_partial(partial_sp if partial else cur_speaker, cur_text + partial)
                if res.get("finished"):
                    flush()
                    print_line("# 終了")
                    break
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            flush()
            save(live=False)
            print_line(f"# 議事録を保存しました: {out_path} / {html_path}")


if __name__ == "__main__":
    main()
