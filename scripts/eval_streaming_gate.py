"""ストリーミング・ゲートの検証: バッチ版と一致するか＆遅延・計算速度.

合成データで、
  - バッチ版 pipeline.run の本人/漏れ判定（全体を見て決める）
  - StreamingGate の逐次判定（先読みだけで決める）
をhop単位で突き合わせ、一致率を出す。あわせて1hopの計算時間とRTFを測る。
リアルタイムで使えるか（計算が実時間に間に合うか、遅延がどれだけか）を確認する。
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

from spkattr.synth import SynthConfig, synthesize
from spkattr.pipeline import run
from spkattr.streaming import StreamingGate

SR = 16000


def batch_own_grid(channels, names, hop_s):
    """バッチ版判定を hop グリッドの own[bool] (n_hop, n_ch) にする."""
    segs = run(channels, SR, names, enroll=None)
    n = channels.shape[1]
    n_hop = int(n / (hop_s * SR))
    grid = np.zeros((n_hop, len(names)), dtype=bool)
    # segments は窓(500ms)。各hop中心がownなsegに入るか
    for seg in segs:
        if not seg.is_own_voice:
            continue
        h0 = int(seg.start / hop_s); h1 = int(seg.end / hop_s)
        grid[h0:h1, seg.channel] = True
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="samples")
    ap.add_argument("--leak-db", type=float, default=-6.0)
    ap.add_argument("--overlap-s", type=float, default=2.0)
    ap.add_argument("--hop-s", type=float, default=0.25)
    ap.add_argument("--lookahead-s", type=float, default=0.25)
    args = ap.parse_args()

    import librosa
    files = sorted(glob.glob(os.path.join(args.dir, "*.wav")))
    names = [os.path.basename(f).split(".")[0] for f in files]
    clips = [librosa.load(f, sr=SR)[0] for f in files]
    ov = int(SR * args.overlap_s); st = []; c = 0
    for cl in clips:
        st.append(c); c += max(1, len(cl) - ov)
    tot = max(s + len(cl) for s, cl in zip(st, clips))
    sources = [np.pad(cl, (s, tot - s - len(cl))) for s, cl in zip(st, clips)]
    channels, _ = synthesize(sources, SynthConfig(sr=SR, leak_db=args.leak_db))

    # バッチ判定
    bgrid = batch_own_grid(channels, names, args.hop_s)

    # ストリーミング判定（hopずつ流し込む）
    g = StreamingGate(len(names), SR, hop_s=args.hop_s, lookahead_s=args.lookahead_s)
    hop = int(args.hop_s * SR)
    sgrid = []
    t0 = time.time(); pushes = 0
    for i in range(0, channels.shape[1], hop):
        out = g.push(channels[:, i:i + hop])
        pushes += 1
        for o in out:
            sgrid.append(o["own"])
    dt = time.time() - t0
    sgrid = np.array(sgrid)

    m = min(len(bgrid), len(sgrid))
    bgrid, sgrid = bgrid[:m], sgrid[:m]
    agree = (bgrid == sgrid).mean()
    audio_s = channels.shape[1] / SR
    print(f"# hop={args.hop_s}s lookahead={args.lookahead_s}s 音声 {audio_s:.0f}s")
    print(f"バッチ vs ストリーミング 判定一致率: {agree*100:.1f}%")
    print(f"本人発話hopの再現率: {(sgrid & bgrid).sum()/max(1,bgrid.sum())*100:.1f}%")
    print(f"本質遅延(先読み): {g.latency_s*1000:.0f}ms")
    print(f"処理 {dt:.2f}s / 音声 {audio_s:.0f}s / RTF={dt/audio_s:.3f}  (<<1なら実時間に余裕)")
    print(f"1チャンク平均計算時間: {dt/max(1,pushes)*1000:.1f}ms (hop {args.hop_s*1000:.0f}ms より小さければ間に合う)")


if __name__ == "__main__":
    main()
