"""話者ラベル付き文字起こしのデモ.

samples/ の音声を会話状に合成し、漏れ込みを除いて
「[話者] テキスト」形式のトランスクリプトを出力する。
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

from spkattr.synth import SynthConfig, synthesize
from spkattr.pipeline import run
from spkattr.embedding import Enrollment
from spkattr.asr import Transcriber, transcribe

SR = 16000


def arrange(clips, overlap_s):
    overlap = int(SR * overlap_s)
    starts, cur = [], 0
    for c in clips:
        starts.append(cur); cur += max(1, len(c) - overlap)
    total = max(s + len(c) for s, c in zip(starts, clips))
    return [np.pad(c, (s, total - s - len(c))) for s, c in zip(starts, clips)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default=os.path.join(os.path.dirname(__file__), "..", "samples"))
    ap.add_argument("--leak-db", type=float, default=-12.0)
    ap.add_argument("--overlap-s", type=float, default=2.0)
    ap.add_argument("--model", default="base", help="whisper モデル(tiny/base/small)")
    ap.add_argument("--gate", action="store_true",
                    help="漏れ込みゲートを使う(既定: 使う)。--no-gate で全チャンネル素通し比較")
    ap.add_argument("--no-gate", dest="gate", action="store_false")
    ap.set_defaults(gate=True)
    args = ap.parse_args()

    import librosa
    files = sorted(glob.glob(os.path.join(args.samples_dir, "*.wav")))
    clips, names = [], []
    for p in files:
        y, _ = librosa.load(p, sr=SR)
        clips.append(y); names.append(os.path.basename(p).split(".")[0])
    sources = arrange(clips, args.overlap_s)

    channels, _ = synthesize(sources, SynthConfig(sr=SR, leak_db=args.leak_db))
    enroll = Enrollment.create("cpu")
    for nm, src in zip(names, sources):
        enroll.register(nm, src)

    segments = run(channels, SR, names, enroll=enroll)
    if not args.gate:
        # ゲートなし比較: 全 active 窓を本人扱いにする
        for s in segments:
            s.is_own_voice = s.is_speech

    print(f"# モデル={args.model} ゲート={'あり' if args.gate else 'なし'} leak={args.leak_db}dB")
    t0 = time.time()
    tr = Transcriber(args.model)
    utts = transcribe(channels, SR, names, segments, tr)
    dt = time.time() - t0
    audio_s = channels.shape[1] / SR
    print(f"# 処理 {dt:.1f}s / 音声 {audio_s:.1f}s / RTF={dt/audio_s:.2f}\n")
    for u in utts:
        print(f"[{u.start:5.1f}-{u.end:4.1f}s] {u.speaker:6s}: {u.text}")


if __name__ == "__main__":
    main()
