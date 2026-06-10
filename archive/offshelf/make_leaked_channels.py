"""漏れ込み入りの「各チャンネル」wavを書き出す（既製品が漏れ込みに耐えるか試すため）.

samples/ の各話者を、会話状に並べて漏れ込み(クロストーク)を合成し、
チャンネルごとに別wavとして offshelf/leaked_chans/ に保存する。
これを parakeet-mlx 等に1チャンネルずつ当てれば、
「既製品は漏れ込みで崩れるか？＝本研究のゲートが要るか」を体感できる。

使い方:
  uv run python offshelf/make_leaked_channels.py --leak-db -6
  uv run python offshelf/run_parakeet_mlx.py --dir offshelf/leaked_chans
  # 比較: クリーンなら --dir samples（漏れ込みなし）
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

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leak-db", type=float, default=-6.0, help="漏れ込み量(大きいほど難しい。例 -6)")
    ap.add_argument("--overlap-s", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "leaked_chans"))
    args = ap.parse_args()

    import librosa
    sdir = os.path.join(os.path.dirname(__file__), "..", "samples")
    files = sorted(glob.glob(os.path.join(sdir, "*.wav")))
    if not files:
        raise SystemExit("samples/ に wav がありません")
    names = [os.path.basename(f).split(".")[0] for f in files]
    clips = [librosa.load(f, sr=SR)[0] for f in files]

    # 会話状に配置
    overlap = int(SR * args.overlap_s)
    starts, cur = [], 0
    for c in clips:
        starts.append(cur); cur += max(1, len(c) - overlap)
    total = max(s + len(c) for s, c in zip(starts, clips))
    sources = [np.pad(c, (s, total - s - len(c))) for s, c in zip(starts, clips)]

    channels, _ = synthesize(sources, SynthConfig(sr=SR, leak_db=args.leak_db))

    os.makedirs(args.out, exist_ok=True)
    for nm, ch in zip(names, channels):
        p = os.path.join(args.out, f"{nm}.wav")
        sf.write(p, ch.astype(np.float32), SR)
        print("wrote", p)
    print(f"\n# leak={args.leak_db}dB の各チャンネルを {args.out} に書き出し")
    print("# 次: uv run python offshelf/run_parakeet_mlx.py --dir offshelf/leaked_chans")


if __name__ == "__main__":
    main()
