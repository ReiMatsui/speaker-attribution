"""「1人1台のiPhone（非同期）」前提のゲートを評価する.

同期マイク前提のチャンネル間時間差(GCC-PHAT)は使わず、
各チャンネルの各窓で「声紋がどの登録話者に最も近いか」を見て、
最も近いのが本人ならその窓を本人発話とみなす（=各マイクで優勢な声が本人か）。
非同期のバラバラ端末でも使える手がかり（声紋＋本人マイクでの優勢）だけで漏れを落とす。

比較条件（正解=LibriSpeech公式トランスクリプト, Whisper base）:
  clean      : 漏れ込みなし下限
  no-gate    : 漏れ込みあり・ゲートなし
  timing-gate: 同期マイク前提（チャンネル間時間差）= 既存 run()
  phone-gate : 非同期iPhone前提（声紋優勢）= 本実験
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from spkattr.synth import SynthConfig, synthesize
from spkattr.pipeline import run, Segment
from spkattr.embedding import Enrollment, cosine
from spkattr.asr import Transcriber, transcribe
from spkattr import vad as vadmod

import sys
sys.path.insert(0, os.path.dirname(__file__))
from eval_separation import load_speakers, arrange, overall_wer  # type: ignore

SR = 16000
WIN_MS, HOP_MS = 1000.0, 250.0


def phone_gate_segments(channels, names, enroll, active_ratio=0.05):
    """声紋優勢ゲート: 各窓で本人声紋が最も近ければ本人発話."""
    n_ch, n = channels.shape
    win = int(SR * WIN_MS / 1000)
    hop = int(SR * HOP_MS / 1000)
    # チャンネル全体VAD（ノイズ床を全体推定）
    vad_frames = [vadmod.energy_vad(channels[ch], SR, hop_ms=10.0) for ch in range(n_ch)]
    segs = []
    for s0 in range(0, max(1, n - win + 1), hop):
        s1 = min(s0 + win, n)
        f0 = int((s0 / SR) * 100); f1 = int((s1 / SR) * 100)
        for ch in range(n_ch):
            seg = vad_frames[ch][f0:f1]
            if len(seg) == 0 or seg.mean() <= active_ratio:
                continue
            emb = enroll.embed(channels[ch, s0:s1])
            scores = {nm: cosine(emb, enroll.profiles[nm]) for nm in names}
            best = max(scores, key=scores.get)
            own = (best == names[ch])
            segs.append(Segment(start=s0 / SR, end=s1 / SR, channel=ch,
                                owner_name=names[ch], is_speech=True,
                                is_own_voice=own, verify_score=scores[names[ch]]))
    return segs


def wer_of(channels, names, segs, tr, refs):
    utts = transcribe(channels, SR, names, segs, tr)
    txt = {nm: "" for nm in names}
    for u in utts:
        txt[u.speaker] = (txt[u.speaker] + " " + u.text).strip()
    return overall_wer(refs, txt, names), txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/tmp/dl_done.txt")
    ap.add_argument("--n-speakers", type=int, default=4)
    ap.add_argument("--overlap-s", type=float, default=2.0)
    ap.add_argument("--leak-db", type=float, default=-6.0)
    ap.add_argument("--model", default="base")
    args = ap.parse_args()

    pq_path = args.parquet
    if pq_path.endswith(".txt") and os.path.exists(pq_path):
        pq_path = open(pq_path).read().strip()
    spk = load_speakers(pq_path, args.n_speakers)
    names = [f"id{s}" for s in spk]
    clips = [spk[s][0] for s in spk]
    refs = {f"id{s}": spk[s][1] for s in spk}
    sources = arrange(clips, args.overlap_s)
    channels, _ = synthesize(sources, SynthConfig(sr=SR, leak_db=args.leak_db))

    enroll = Enrollment.create("cpu")
    for nm, src in zip(names, sources):
        enroll.register(nm, src)
    tr = Transcriber(args.model)

    # clean下限
    clean = {nm: tr.transcribe_span(src, SR) for nm, src in zip(names, sources)}
    print(f"clean        WER={overall_wer(refs, clean, names)*100:.0f}%")

    # no-gate
    segs_ng = run(channels, SR, names, enroll=None)
    for s in segs_ng:
        s.is_own_voice = s.is_speech
    w, _ = wer_of(channels, names, segs_ng, tr, refs)
    print(f"no-gate      WER={w*100:.0f}%")

    # timing-gate（同期マイク）
    segs_t = run(channels, SR, names, enroll=None)
    w, _ = wer_of(channels, names, segs_t, tr, refs)
    print(f"timing-gate  WER={w*100:.0f}%  (同期マイク前提)")

    # phone-gate（非同期iPhone・声紋優勢）
    segs_p = phone_gate_segments(channels, names, enroll)
    w, txt = wer_of(channels, names, segs_p, tr, refs)
    print(f"phone-gate   WER={w*100:.0f}%  (非同期iPhone・声紋優勢)")
    print("\n# phone-gate 各話者:")
    for nm in names:
        print(f"  [{nm}] {txt.get(nm,'')}")


if __name__ == "__main__":
    main()
