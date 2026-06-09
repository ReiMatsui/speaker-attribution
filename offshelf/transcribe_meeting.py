"""会議を流し込むと『誰が・いつ・何を』の一本の書き起こしを出す実用ツール.

入力 : 1人1マイクのwav群（各wav=各話者）
処理 : 本研究のゲートで漏れ込みを落とす → 各chをASR → 時刻で1本に統合
出力 : 時系列の話者ラベル付きトランスクリプト（テキスト＋HTML）

ASRエンジン:
  --engine parakeet  … Mac(Apple Silicon)で最高精度・推奨
  --engine whisper   … どこでも動く（検証用）

使い方(Mac):
  uv run python offshelf/transcribe_meeting.py --dir offshelf/ami_raw --engine parakeet
  uv run python offshelf/transcribe_meeting.py --dir offshelf/ami_raw --engine parakeet --no-gate  # 比較
"""
from __future__ import annotations

import argparse
import glob
import os
import tempfile

import numpy as np
import soundfile as sf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from spkattr.pipeline import run            # type: ignore
from spkattr.asr import merge_own_spans     # type: ignore

SR = 16000


def gate_channels(channels, names):
    """漏れ込みを落とした連続音声（本人時間以外を無音化）を返す."""
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
    """音声 → [(start, end, text)] の文単位."""
    if engine == "parakeet":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            sf.write(tf.name, audio, SR)
            path = tf.name
        res = model_obj.transcribe(path)
        os.unlink(path)
        return [(s.start, s.end, s.text.strip()) for s in res.sentences if s.text.strip()]
    else:  # whisper
        from spkattr.asr import transcribe_words
        words = transcribe_words(model_obj, audio, SR)
        # 単語を ~0.8秒ギャップで文に区切る
        sents = []; cur = [];
        for w, s, e in words:
            if cur and s - cur[-1][2] > 0.8:
                sents.append((cur[0][1], cur[-1][2], " ".join(x[0] for x in cur))); cur = []
            cur.append((w, s, e))
        if cur:
            sents.append((cur[0][1], cur[-1][2], " ".join(x[0] for x in cur)))
        return sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="各wav=各話者 のフォルダ")
    ap.add_argument("--names", default=None, help="話者名をカンマ区切りで（省略時はファイル名）")
    ap.add_argument("--engine", choices=["parakeet", "whisper"], default="parakeet")
    ap.add_argument("--no-gate", dest="gate", action="store_false")
    ap.add_argument("--out", default="meeting_transcript.html")
    args = ap.parse_args()

    import librosa
    files = sorted(glob.glob(os.path.join(args.dir, "*.wav")))
    names = args.names.split(",") if args.names else [os.path.basename(f).split(".")[0] for f in files]
    clips = [librosa.load(f, sr=SR)[0] for f in files]
    n = min(len(c) for c in clips)
    channels = np.stack([c[:n] for c in clips])

    if args.gate:
        channels = gate_channels(channels, names)

    if args.engine == "parakeet":
        from parakeet_mlx import from_pretrained
        model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    else:
        from spkattr.asr import Transcriber
        model = Transcriber("base")

    events = []
    for ch, nm in enumerate(names):
        for s, e, text in asr_sentences(channels[ch], args.engine, model):
            events.append((s, nm, text))
    events.sort(key=lambda x: x[0])

    print(f"\n=== 会議の書き起こし（{'ゲートあり' if args.gate else 'ゲートなし'} / {args.engine}） ===\n")
    for s, nm, text in events:
        print(f"[{s:6.1f}s] {nm:6s}: {text}")

    write_html(events, names, args.gate, args.engine, args.out)
    print(f"\nwrote {os.path.abspath(args.out)}")


COLORS = ["#0a84ff", "#ff9500", "#34c759", "#af52de", "#ff2d55", "#5ac8fa"]


def write_html(events, names, gate, engine, out):
    rows = ""
    for s, nm, text in events:
        col = COLORS[names.index(nm) % len(COLORS)]
        rows += (f'<div class="ev"><span class="ts">{s:.1f}s</span>'
                 f'<span class="sp" style="color:{col}">{nm}</span>'
                 f'<span class="tx">{text}</span></div>')
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>会議の書き起こし</title><style>
 body{{font-family:-apple-system,"Hiragino Sans",sans-serif;max-width:820px;margin:24px auto;padding:0 16px;color:#222;line-height:1.6}}
 h1{{font-size:18px}} .ev{{display:flex;gap:10px;padding:5px 0;border-bottom:1px solid #f0f2f5}}
 .ts{{color:#aab;font-size:12px;width:54px;flex:none;text-align:right;font-variant-numeric:tabular-nums;padding-top:2px}}
 .sp{{font-weight:500;width:70px;flex:none}} .tx{{flex:1}}
</style></head><body>
<h1>会議の書き起こし（誰が・いつ・何を）　<span style="font-size:12px;color:#889">{'ゲートあり' if gate else 'ゲートなし'} / {engine}</span></h1>
{rows}
</body></html>"""
    with open(out, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
