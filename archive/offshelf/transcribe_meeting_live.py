"""擬似ライブの通し: ストリーミング・ゲート → parakeet stream（逐次の議事録）.

音声を hop 単位で流し込み、
  1) StreamingGate で「このhopは各chで本人か漏れか」を逐次判定（先読み少しだけ）
  2) 漏れと判定したhopは無音化して、各chのparakeet stream に送る
  3) 各chの確定テキストが伸びていく → 時系列に統合して表示
本物のライブに近い挙動・遅延を体感するためのもの。Mac(Apple Silicon, parakeet)で実行。

使い方:
  uv run python offshelf/transcribe_meeting_live.py --dir offshelf/ami_raw
  uv run python offshelf/transcribe_meeting_live.py --dir offshelf/ami_raw --realtime  # 実時間で流す
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from contextlib import ExitStack

import numpy as np
import soundfile as sf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from spkattr.streaming import StreamingGate  # type: ignore

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--names", default=None)
    ap.add_argument("--hop-s", type=float, default=0.25)
    ap.add_argument("--lookahead-s", type=float, default=0.25)
    ap.add_argument("--realtime", action="store_true", help="実時間で流して体感する")
    ap.add_argument("--no-gate", dest="gate", action="store_false")
    ap.add_argument("--out", default="meeting_live.html")
    args = ap.parse_args()

    import librosa
    files = sorted(glob.glob(os.path.join(args.dir, "*.wav")))
    names = args.names.split(",") if args.names else [os.path.basename(f).split(".")[0] for f in files]
    clips = [librosa.load(f, sr=SR)[0] for f in files]
    n = min(len(c) for c in clips)
    channels = np.stack([c[:n] for c in clips]).astype(np.float32)
    n_ch = len(names)

    print("# parakeetモデルを読み込み中…（初回はDLで数十秒かかることがあります）", flush=True)
    from parakeet_mlx import from_pretrained
    import mlx.core as mx
    model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    print(f"# 開始: {n/SR:.0f}秒 / {n_ch}ch / {'実時間' if args.realtime else '最速'}再生 / hop {args.hop_s*1000:.0f}ms\n", flush=True)

    gate = StreamingGate(n_ch, SR, hop_s=args.hop_s, lookahead_s=args.lookahead_s)
    hop = int(args.hop_s * SR)
    raw_hops = []        # グローバルhop番号 -> (n_ch, hop)
    worst_ms = 0.0
    last_len = [0] * n_ch   # 各chの既表示テキスト長

    with ExitStack() as stack:
        streams = [stack.enter_context(model.transcribe_stream(context_size=(256, 256)))
                   for _ in range(n_ch)]

        def feed(idx, own):
            rh = raw_hops[idx]
            for ch in range(n_ch):
                a = rh[ch] if (own[ch] or not args.gate) else np.zeros(hop, dtype=np.float32)
                streams[ch].add_audio(mx.array(a))

        def show(t_now):
            # 各chで新しく確定した分だけを逐次表示
            for ch, nm in enumerate(names):
                try:
                    txt = streams[ch].result.text.strip()
                except Exception:
                    continue
                if len(txt) > last_len[ch]:
                    new = txt[last_len[ch]:].strip()
                    last_len[ch] = len(txt)
                    if new:
                        print(f"[{t_now:6.1f}s] {nm:8s}: {new}", flush=True)

        n_hops = channels.shape[1] // hop
        for i in range(n_hops):
            t0 = time.time()
            chunk = channels[:, i * hop:(i + 1) * hop]
            raw_hops.append(chunk)
            for d in gate.push(chunk):
                idx = int(round(d["start"] / args.hop_s))
                if idx < len(raw_hops):
                    feed(idx, d["own"])
            worst_ms = max(worst_ms, (time.time() - t0) * 1000)
            if i % 4 == 0:                      # ~1秒ごとに逐次表示
                show((i + 1) * args.hop_s)
            if args.realtime:
                time.sleep(max(0, args.hop_s - (time.time() - t0)))
        show(n_hops * args.hop_s)

        # 各chの結果を集める（最終）
        events = []
        for ch, nm in enumerate(names):
            for s in streams[ch].result.sentences:
                if s.text.strip():
                    events.append((s.start, nm, s.text.strip()))
    events.sort(key=lambda x: x[0])

    print(f"\n=== ライブ議事録（{'ゲートあり' if args.gate else 'ゲートなし'} / parakeet stream） ===\n")
    for s, nm, text in events:
        print(f"[{s:6.1f}s] {nm:8s}: {text}")
    print(f"\n# 1hopの最悪処理時間: {worst_ms:.0f}ms（hop {args.hop_s*1000:.0f}ms ＋ 先読み {args.lookahead_s*1000:.0f}ms が遅延の主因）")

    _html(events, names, args.gate, args.out)
    print(f"# wrote {os.path.abspath(args.out)}")


COLORS = ["#0a84ff", "#ff9500", "#34c759", "#af52de", "#ff2d55", "#5ac8fa"]


def _html(events, names, gate, out):
    rows = ""
    for s, nm, text in events:
        col = COLORS[names.index(nm) % len(COLORS)]
        rows += (f'<div class="ev"><span class="ts">{s:.1f}s</span>'
                 f'<span class="sp" style="color:{col}">{nm}</span><span class="tx">{text}</span></div>')
    open(out, "w").write(f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>ライブ議事録</title><style>
 body{{font-family:-apple-system,"Hiragino Sans",sans-serif;max-width:820px;margin:24px auto;padding:0 16px;line-height:1.6}}
 .ev{{display:flex;gap:10px;padding:5px 0;border-bottom:1px solid #f0f2f5}}
 .ts{{color:#aab;font-size:12px;width:54px;flex:none;text-align:right;font-variant-numeric:tabular-nums;padding-top:2px}}
 .sp{{font-weight:500;width:80px;flex:none}} .tx{{flex:1}}</style></head><body>
<h1 style="font-size:18px">ライブ議事録 <span style="font-size:12px;color:#889">{'ゲートあり' if gate else 'ゲートなし'} / parakeet stream</span></h1>
{rows}</body></html>""")


if __name__ == "__main__":
    main()
