"""準リアルタイム議事録: 数秒チャンクで処理（バッチ並みのきれいさ＋数秒遅れ）.

超低遅延(0.25秒)はparakeetの文脈を切って品質が落ちるため、
「数秒ぶんの連続音声をまとめてASRし、新しい区間だけ吐く」方式にする。
  - ゲートで漏れ込みを落とす
  - chunk_s 秒ごとに、[t-context, t+chunk_s] の連続音声をASR
  - そのチャンクで新しく出た文だけを時系列に表示
遅延 ≒ chunk_s 秒（数秒）。会議用途ならこれで実用的。

使い方(Mac):
  uv run python offshelf/transcribe_meeting_nearlive.py --dir offshelf/ami_raw --engine parakeet
  uv run python offshelf/transcribe_meeting_nearlive.py --dir offshelf/ami_raw --engine parakeet --realtime
"""
from __future__ import annotations

import argparse
import glob
import os
import tempfile
import time

import numpy as np
import soundfile as sf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from spkattr.pipeline import run            # type: ignore
from spkattr.asr import merge_own_spans     # type: ignore

SR = 16000


def gate_channels(channels, names):
    n_ch, n = channels.shape
    segs = run(channels, SR, names, enroll=None)
    spans = merge_own_spans(segs, n_ch, names, max_gap_s=2.5)
    ramp = int(0.03 * SR); k = np.ones(ramp) / ramp
    out = np.zeros_like(channels)
    for ch in range(n_ch):
        m = np.zeros(n)
        for s, e in spans.get(ch, []):
            a = int(max(0, (s - 0.4) * SR)); b = int(min(n, (e + 0.4) * SR)); m[a:b] = 1.0
        m = np.clip(np.convolve(m, k, mode="same"), 0, 1)
        out[ch] = channels[ch] * m
    return out


def asr_sentences(audio, engine, model_obj):
    if engine == "parakeet":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            sf.write(tf.name, audio, SR); path = tf.name
        res = model_obj.transcribe(path); os.unlink(path)
        return [(s.start, s.end, s.text.strip()) for s in res.sentences if s.text.strip()]
    else:
        from spkattr.asr import transcribe_words
        words = transcribe_words(model_obj, audio, SR)
        sents = []; cur = []
        for w, s, e in words:
            if cur and s - cur[-1][2] > 0.8:
                sents.append((cur[0][1], cur[-1][2], " ".join(x[0] for x in cur))); cur = []
            cur.append((w, s, e))
        if cur:
            sents.append((cur[0][1], cur[-1][2], " ".join(x[0] for x in cur)))
        return sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--names", default=None)
    ap.add_argument("--engine", choices=["parakeet", "whisper"], default="parakeet")
    ap.add_argument("--chunk-s", type=float, default=5.0, help="数秒ごとに処理（遅延の主因）")
    ap.add_argument("--context-s", type=float, default=1.5, help="境界で語を切らないための前文脈")
    ap.add_argument("--no-gate", dest="gate", action="store_false")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--out", default="meeting_nearlive.html")
    args = ap.parse_args()

    import librosa
    files = sorted(glob.glob(os.path.join(args.dir, "*.wav")))
    names = args.names.split(",") if args.names else [os.path.basename(f).split(".")[0] for f in files]
    clips = [librosa.load(f, sr=SR)[0] for f in files]
    n = min(len(c) for c in clips)
    channels = np.stack([c[:n] for c in clips]).astype(np.float32)

    print("# 準備中（モデル読み込み）…", flush=True)
    if args.engine == "parakeet":
        from parakeet_mlx import from_pretrained
        model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    else:
        from spkattr.asr import Transcriber
        model = Transcriber("base")

    if args.gate:
        channels = gate_channels(channels, names)
    total_s = n / SR
    print(f"# 開始: {total_s:.0f}秒 / chunk {args.chunk_s:.0f}秒ごと / {'ゲートあり' if args.gate else 'なし'} / {args.engine}\n", flush=True)

    emitted = set()
    events = []
    t = 0.0
    while t < total_s:
        a = max(0.0, t - args.context_s); b = min(total_s, t + args.chunk_s)
        for ch, nm in enumerate(names):
            seg = channels[ch][int(a * SR):int(b * SR)]
            if np.abs(seg).max() < 1e-4:
                continue
            for s, e, text in asr_sentences(seg, args.engine, model):
                abs_s = a + s
                if abs_s < t - 0.05:       # 文脈領域の再掲はスキップ
                    continue
                key = (round(abs_s, 1), nm, text)
                if key in emitted:
                    continue
                emitted.add(key)
                events.append((abs_s, nm, text))
                print(f"[{abs_s:6.1f}s] {nm:8s}: {text}", flush=True)
        t += args.chunk_s
        if args.realtime:
            time.sleep(args.chunk_s)

    events.sort(key=lambda x: x[0])
    _html(events, names, args.gate, args.engine, args.out)
    print(f"\n# 遅延の目安: 約 {args.chunk_s:.0f} 秒（chunk長）。 wrote {os.path.abspath(args.out)}", flush=True)


COLORS = ["#0a84ff", "#ff9500", "#34c759", "#af52de", "#ff2d55", "#5ac8fa"]


def _html(events, names, gate, engine, out):
    rows = ""
    for s, nm, text in events:
        col = COLORS[names.index(nm) % len(COLORS)]
        rows += (f'<div class="ev"><span class="ts">{s:.1f}s</span>'
                 f'<span class="sp" style="color:{col}">{nm}</span><span class="tx">{text}</span></div>')
    open(out, "w").write(f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>準リアルタイム議事録</title><style>
 body{{font-family:-apple-system,"Hiragino Sans",sans-serif;max-width:820px;margin:24px auto;padding:0 16px;line-height:1.6}}
 .ev{{display:flex;gap:10px;padding:5px 0;border-bottom:1px solid #f0f2f5}}
 .ts{{color:#aab;font-size:12px;width:54px;flex:none;text-align:right;font-variant-numeric:tabular-nums;padding-top:2px}}
 .sp{{font-weight:500;width:80px;flex:none}} .tx{{flex:1}}</style></head><body>
<h1 style="font-size:18px">準リアルタイム議事録 <span style="font-size:12px;color:#889">{'ゲートあり' if gate else 'ゲートなし'} / {engine} / 数秒遅れ</span></h1>
{rows}</body></html>""")


if __name__ == "__main__":
    main()
