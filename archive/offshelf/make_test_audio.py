"""既製品を試すための入力音声を samples/ から作る.

- mix_singlemic.wav : 全員を1本に混ぜた「1本マイク」想定（Moonshine等に渡す）
- multichannel.wav  : 各話者を別チャンネルに入れた多ch wav（Deepgram multichannel に渡す）
両方とも、会話のように少しずらして配置する。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import soundfile as sf

SR = 16000


def main():
    here = os.path.dirname(__file__)
    sdir = os.path.join(here, "..", "samples")
    files = sorted(glob.glob(os.path.join(sdir, "*.wav")))
    if not files:
        raise SystemExit("samples/ に wav がありません（scripts/fetch_samples.py で取得）")

    import librosa
    clips = [librosa.load(f, sr=SR)[0] for f in files]

    overlap = int(SR * 2.0)
    starts, cur = [], 0
    for c in clips:
        starts.append(cur); cur += max(1, len(c) - overlap)
    total = max(s + len(c) for s, c in zip(starts, clips))

    chans = np.zeros((len(clips), total), dtype=np.float32)
    for i, (s, c) in enumerate(zip(starts, clips)):
        chans[i, s:s + len(c)] = c

    mix = chans.mean(axis=0)
    out_mix = os.path.join(here, "mix_singlemic.wav")
    out_mc = os.path.join(here, "multichannel.wav")
    sf.write(out_mix, mix, SR)
    sf.write(out_mc, chans.T, SR)  # (samples, channels)
    print("wrote", out_mix)
    print("wrote", out_mc, f"({len(clips)}ch)")


if __name__ == "__main__":
    main()
