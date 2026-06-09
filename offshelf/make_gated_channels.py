"""本研究のゲートを通した「漏れ込み除去済みチャンネル」を書き出す（効果実証用）.

流れ: samples → 漏れ込み合成 → 本研究のゲート(VAD→相関→声紋)で本人時間以外を無音化
      → チャンネルごとにwav保存。これを parakeet に当てて「またきれいに本人だけ出るか」を見る。

使い方:
  uv run python offshelf/make_gated_channels.py --leak-db -6
  uv run python offshelf/run_parakeet_mlx.py --dir offshelf/gated_chans
  # 比較:
  #   offshelf/leaked_chans  … 漏れ込みあり・ゲートなし（崩壊する）
  #   offshelf/gated_chans   … 漏れ込みあり・ゲートあり（本研究。きれいに戻るか）
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import soundfile as sf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from spkattr.synth import SynthConfig, synthesize  # type: ignore
from spkattr.pipeline import run  # type: ignore
from spkattr.embedding import Enrollment  # type: ignore
from spkattr.asr import merge_own_spans  # type: ignore

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leak-db", type=float, default=-6.0)
    ap.add_argument("--overlap-s", type=float, default=2.0)
    ap.add_argument("--pad-s", type=float, default=0.4)
    ap.add_argument("--merge-gap-s", type=float, default=2.5)
    ap.add_argument("--use-embedding", action="store_true", help="声紋照合も使う")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "gated_chans"))
    args = ap.parse_args()

    import librosa
    sdir = os.path.join(os.path.dirname(__file__), "..", "samples")
    files = sorted(glob.glob(os.path.join(sdir, "*.wav")))
    names = [os.path.basename(f).split(".")[0] for f in files]
    clips = [librosa.load(f, sr=SR)[0] for f in files]

    overlap = int(SR * args.overlap_s)
    starts, cur = [], 0
    for c in clips:
        starts.append(cur); cur += max(1, len(c) - overlap)
    total = max(s + len(c) for s, c in zip(starts, clips))
    sources = [np.pad(c, (s, total - s - len(c))) for s, c in zip(starts, clips)]

    channels, _ = synthesize(sources, SynthConfig(sr=SR, leak_db=args.leak_db))

    enroll = None
    if args.use_embedding:
        enroll = Enrollment.create("cpu")
        for nm, src in zip(names, sources):
            enroll.register(nm, src)

    segs = run(channels, SR, names, enroll=enroll)

    # 本人時間以外を無音化した連続音声を作る（pipeline.transcribe と同じマスク）
    n_ch, n = channels.shape
    spans = merge_own_spans(segs, n_ch, names, max_gap_s=args.merge_gap_s)
    ramp = max(1, int(0.03 * SR)); kernel = np.ones(ramp) / ramp

    os.makedirs(args.out, exist_ok=True)
    for ch in range(n_ch):
        mask = np.zeros(n)
        for s, e in spans.get(ch, []):
            a = int(max(0, (s - args.pad_s) * SR)); b = int(min(n, (e + args.pad_s) * SR))
            mask[a:b] = 1.0
        mask = np.clip(np.convolve(mask, kernel, mode="same"), 0, 1)
        p = os.path.join(args.out, f"{names[ch]}.wav")
        sf.write(p, (channels[ch] * mask).astype(np.float32), SR)
        print("wrote", p)
    print(f"\n# leak={args.leak_db}dB をゲート除去した各チャンネルを {args.out} に保存")
    print("# 次: uv run python offshelf/run_parakeet_mlx.py --dir offshelf/gated_chans")


if __name__ == "__main__":
    main()
