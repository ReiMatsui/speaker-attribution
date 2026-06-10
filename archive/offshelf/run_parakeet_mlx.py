"""既製品（M4ローカルの本命）: parakeet-mlx を「1人1マイク」に当てる.

NVIDIA Parakeet（精度トップクラス）をApple Silicon向けMLXで実行。
各wav=各話者マイクとして別々に文字起こし（=multichannel方式で話者分離は自明）。
高精度・高速・ローカル完結・APIキー不要。MLXのためMac(Apple Silicon)専用。

準備（Mac, ffmpeg必須）:
  brew install ffmpeg
  uv add parakeet-mlx -U

使い方:
  uv run python offshelf/run_parakeet_mlx.py --dir samples
  # 自前のwav群（各wav=各話者）なら --dir <フォルダ>
"""
from __future__ import annotations

import argparse
import glob
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="samples", help="各wav=各話者マイク のフォルダ")
    ap.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v3")
    args = ap.parse_args()

    try:
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import load_audio
    except ImportError:
        raise SystemExit("parakeet-mlx 未インストール。Mac で `uv add parakeet-mlx -U`（ffmpeg必須）")

    model = from_pretrained(args.model)
    sr = model.preprocessor_config.sample_rate

    files = sorted(glob.glob(os.path.join(args.dir, "*.wav")))
    if not files:
        raise SystemExit(f"{args.dir} に wav がありません")

    total_audio = 0.0
    t0 = time.time()
    print(f"# parakeet-mlx / {args.model} / {len(files)}ch\n")
    for f in files:
        name = os.path.basename(f).split(".")[0]
        total_audio += len(load_audio(f, sr)) / sr
        result = model.transcribe(f)
        print(f"[{name}] {result.text.strip()}\n")
    dt = time.time() - t0
    print(f"# 処理 {dt:.1f}s / 音声 {total_audio:.1f}s / RTF={dt/max(total_audio,1e-9):.2f}")


if __name__ == "__main__":
    main()
