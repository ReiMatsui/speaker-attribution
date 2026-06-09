"""AMI実会議の各チャンネルに本研究のゲートを当てて、生／ゲート後のwavを書き出す.

Mac側で parakeet を「生(漏れ込みあり) vs ゲート後」で当てれば、
最強ASR×実会議×本研究ゲート の絶対精度を体感できる（タスク10）。

手順:
  uv run python offshelf/make_ami_gated.py            # EN2002a 先頭60秒を取得→ゲート
  uv run python offshelf/run_parakeet_mlx.py --dir offshelf/ami_raw     # 生(崩れるはず)
  uv run python offshelf/run_parakeet_mlx.py --dir offshelf/ami_gated   # ゲート後(本人だけ)
"""
from __future__ import annotations

import argparse
import os
import urllib.request

import numpy as np
import soundfile as sf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from spkattr.pipeline import run  # type: ignore
from spkattr.asr import merge_own_spans  # type: ignore

SR = 16000
MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting", default="EN2002a")
    ap.add_argument("--n-mics", type=int, default=4)
    ap.add_argument("--window-s", type=float, default=60.0)
    args = ap.parse_args()

    here = os.path.dirname(__file__)
    raw_dir = os.path.join(here, "ami_raw"); os.makedirs(raw_dir, exist_ok=True)
    gated_dir = os.path.join(here, "ami_gated"); os.makedirs(gated_dir, exist_ok=True)

    # 先頭 window_s 秒ぶんだけ範囲ダウンロード（WAVは16kHz/mono/16bit）
    nbytes = 44 + int(args.window_s * SR) * 2 + 1_000_000
    chans = []
    for i in range(args.n_mics):
        dst = os.path.join(raw_dir, f"mic{i}.wav")
        if not os.path.exists(dst):
            url = f"{MIRROR}/{args.meeting}/audio/{args.meeting}.Headset-{i}.wav"
            print(f"downloading mic{i} ...")
            req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes}"})
            data = urllib.request.urlopen(req, timeout=60).read()
            with open(dst, "wb") as f:
                f.write(data)
        y, sr = sf.read(dst)
        chans.append(y.astype(np.float32))
    n = min(min(len(c) for c in chans), int(args.window_s * SR))
    channels = np.stack([c[:n] for c in chans])
    names = [f"mic{i}" for i in range(args.n_mics)]

    # 生チャンネルを保存(parakeetで比較用)
    for i in range(args.n_mics):
        sf.write(os.path.join(raw_dir, f"mic{i}.wav"), channels[i], SR)

    # ゲート（時間差ゲート, 声紋なし）→ 本人時間以外を無音化
    segs = run(channels, SR, names, enroll=None)
    spans = merge_own_spans(segs, args.n_mics, names, max_gap_s=2.5)
    ramp = int(0.03 * SR); k = np.ones(ramp) / ramp
    for i in range(args.n_mics):
        m = np.zeros(n)
        for s, e in spans.get(i, []):
            a = int(max(0, (s - 0.4) * SR)); b = int(min(n, (e + 0.4) * SR)); m[a:b] = 1.0
        m = np.clip(np.convolve(m, k, mode="same"), 0, 1)
        sf.write(os.path.join(gated_dir, f"mic{i}.wav"), (channels[i] * m).astype(np.float32), SR)

    print(f"\n# {args.meeting} 先頭{n/SR:.0f}秒: 生={raw_dir} / ゲート後={gated_dir}")
    print("# 次: uv run python offshelf/run_parakeet_mlx.py --dir offshelf/ami_raw")
    print("#     uv run python offshelf/run_parakeet_mlx.py --dir offshelf/ami_gated")


if __name__ == "__main__":
    main()
