"""合成データでパイプライン全体を動かすデモ.

実音声がなくてもサイン波/ノイズで動作確認できるよう、
--real を付けなければ簡易な擬似音源で回す。
--real <wav1> <wav2> ... を渡すと実音声で合成・評価する。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from spkattr.synth import SynthConfig, synthesize
from spkattr.pipeline import run
from spkattr.evaluate import evaluate, labels_to_windows
from spkattr.embedding import Enrollment


def make_fake_sources(sr=16000, dur=6.0, n_spk=3, seed=0):
    """擬似話者: 各話者は別の基本周波数を持つ有声音を、別々のタイミングで発話."""
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    t = np.arange(n) / sr
    f0s = [110, 175, 240, 300, 90, 200][:n_spk]
    sources = []
    for i, f0 in enumerate(f0s):
        sig = np.zeros(n)
        # 話者ごとに2〜3個の発話区間
        for _ in range(3):
            st = rng.uniform(0, dur - 1.2)
            ln = rng.uniform(0.6, 1.1)
            a, b = int(st * sr), int((st + ln) * sr)
            seg = sum(np.sin(2 * np.pi * f0 * k * t[a:b]) / k for k in (1, 2, 3))
            sig[a:b] += seg * np.hanning(b - a)
        sources.append(sig / (np.abs(sig).max() + 1e-9) * 0.5)
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="+", help="実音声wavのパス(2つ以上)")
    ap.add_argument("--samples-dir", help="このフォルダ内の *.wav を全話者として使う")
    ap.add_argument("--leak-db", type=float, default=-12.0)
    ap.add_argument("--use-embedding", action="store_true",
                    help="声紋照合も使う(実音声推奨)")
    args = ap.parse_args()
    sr = 16000

    # --real 未指定でも、samples/ があれば自動で使う
    if not args.real:
        import glob
        import os
        sdir = args.samples_dir or os.path.join(os.path.dirname(__file__), "..", "samples")
        found = sorted(glob.glob(os.path.join(sdir, "*.wav")))
        if found:
            args.real = found
            print(f"# samples/ を自動使用: {[os.path.basename(p) for p in found]}")

    if args.real:
        import librosa
        sources, names = [], []
        for p in args.real:
            y, _ = librosa.load(p, sr=sr)
            sources.append(y)
            names.append(p.split("/")[-1].split(".")[0])
        # 長さを揃える
        m = max(len(s) for s in sources)
        sources = [np.pad(s, (0, m - len(s))) for s in sources]
    else:
        sources = make_fake_sources(sr=sr, n_spk=3)
        names = [f"spk{i}" for i in range(len(sources))]

    cfg = SynthConfig(sr=sr, leak_db=args.leak_db)
    channels, labels = synthesize(sources, cfg)
    print(f"# {len(sources)}話者 / leak={args.leak_db}dB / {channels.shape[1]/sr:.1f}s")

    enroll = None
    if args.use_embedding:
        enroll = Enrollment.create("cpu")
        for nm, src in zip(names, sources):
            enroll.register(nm, src)  # クリーン音源で登録
        print("# enrollment 登録済み:", names)

    segs = run(channels, sr, names, enroll=enroll)
    truth_w = labels_to_windows(labels, sr, 500.0, 250.0)
    m = evaluate(segs, truth_w, names, sr)
    print(f"precision           : {m['precision']:.3f}")
    print(f"recall              : {m['recall']:.3f}")
    print(f"f1                  : {m['f1']:.3f}")
    print(f"crosstalk_rejection : {m['crosstalk_rejection']:.3f}")
    print(f"tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")


if __name__ == "__main__":
    sys.exit(main())
