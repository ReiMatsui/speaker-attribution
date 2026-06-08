"""公開音声サンプル(別話者4名)を samples/ にダウンロードして16kHzモノラルに整える.

録音不要で実音声テストを試すための補助スクリプト。
ffmpeg が必要(なければ各wavをそのまま保存し、librosa側で読み込む)。
"""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request

SAMPLES = {
    "spkA": "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav",
    "spkB": "https://models.silero.ai/vad_models/en.wav",
    "spkC": "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav",
    "spkD": "https://dldata-public.s3.us-east-2.amazonaws.com/1919-142785-0028.wav",
}


def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def main() -> int:
    here = os.path.dirname(__file__)
    out = os.path.join(here, "..", "samples")
    os.makedirs(out, exist_ok=True)
    ff = have_ffmpeg()
    for name, url in SAMPLES.items():
        raw = os.path.join(out, f"_{name}_raw.wav")
        dst = os.path.join(out, f"{name}.wav")
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, raw)
        if ff:
            # 16kHz mono / 先頭8秒 / ラウドネス正規化
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw, "-ac", "1", "-ar", "16000",
                 "-t", "8", "-af", "loudnorm=I=-20", dst],
                capture_output=True,
            )
            os.remove(raw)
        else:
            os.replace(raw, dst)
        print(f"  -> {dst}")
    print("\n完了。次を実行:")
    print("  uv run python scripts/demo_synth.py --use-embedding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
