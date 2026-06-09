"""AMI実会議（生のヘッドセット録音＋人手トランスクリプト）での検証.

合成ではなく、実際に録音された1人1ヘッドセットマイクの会議（本物の漏れ込み）で、
本研究のゲートが効くかを人手の正解テキスト基準で測る。

データ:
  音声 = AMIミラーの連続ヘッドセット録音（/tmp/ami_h0..3.wav, 先頭~103s）
  正解 = HF edinburghcstr/ami(ihm) の人手テキスト＋時刻＋話者＋マイク

条件（正解=人手トランスクリプト, Whisper base）:
  no-gate    : 漏れ込み除去なし（各chをそのままASR）
  timing-gate: チャンネル間時間差ゲート（同期マイク前提）
  phone-gate : 声紋優勢ゲート（非同期iPhone前提）
"""
from __future__ import annotations

import argparse
import difflib
import io
import os
import re

import numpy as np
import soundfile as sf

from spkattr.pipeline import run, Segment
from spkattr.embedding import Enrollment, cosine
from spkattr.asr import Transcriber, transcribe
from spkattr import vad as vadmod

SR = 16000
MIC_ORDER = ["H00", "H01", "H02", "H03"]


def norm(t):
    return re.sub(r"[^\w\s]", "", t.lower()).split()


def wer(ref, hyp):
    rw, hw = norm(ref), norm(hyp)
    sm = difflib.SequenceMatcher(a=rw, b=hw, autojunk=False)
    e = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace": e += max(i2 - i1, j2 - j1)
        elif tag == "delete": e += i2 - i1
        elif tag == "insert": e += j2 - j1
    return e, len(rw)


def overall(refs, hyp, names):
    e = w = 0
    for nm in names:
        ee, ww = wer(refs[nm], hyp.get(nm, ""))
        e += ee; w += ww
    return e / max(1, w)


def load_meeting(parquet, meeting, window_s):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(parquet)
    refs = {m: [] for m in MIC_ORDER}          # mic -> [(begin, text)]
    enroll_audio = {}                           # mic -> np.array（登録用）
    spk_of = {}
    for b in f.iter_batches(batch_size=500):
        d = b.to_pydict()
        for i in range(len(d["meeting_id"])):
            if d["meeting_id"][i] != meeting:
                continue
            mic = d["microphone_id"][i]
            if mic not in refs:
                continue
            spk_of[mic] = d["speaker_id"][i]
            if d["begin_time"][i] < window_s:
                refs[mic].append((d["begin_time"][i], d["text"][i]))
            # 登録用に、各micの2秒以上の発話を1つ確保
            if mic not in enroll_audio and (d["end_time"][i] - d["begin_time"][i]) > 2.0:
                y, sr = sf.read(io.BytesIO(d["audio"][i]["bytes"]))
                if sr == SR:
                    enroll_audio[mic] = y.astype(np.float32)
    ref_text = {}
    for m in MIC_ORDER:
        ref_text[m] = " ".join(t for _, t in sorted(refs[m]))
    return ref_text, enroll_audio, spk_of


def phone_gate(channels, names, enroll, active_ratio=0.05, win_ms=1000.0, hop_ms=250.0):
    n_ch, n = channels.shape
    win = int(SR * win_ms / 1000); hop = int(SR * hop_ms / 1000)
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
            best = max(names, key=lambda nm: cosine(emb, enroll.profiles[nm]))
            segs.append(Segment(s0 / SR, s1 / SR, ch, names[ch], True, best == names[ch],
                                cosine(emb, enroll.profiles[names[ch]])))
    return segs


def asr_text(channels, names, segs, tr):
    utts = transcribe(channels, SR, names, segs, tr)
    txt = {nm: "" for nm in names}
    for u in utts:
        txt[u.speaker] = (txt[u.speaker] + " " + u.text).strip()
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/tmp/ami_done.txt")
    ap.add_argument("--meeting", default="EN2002a")
    ap.add_argument("--wav-prefix", default="/tmp/ami_h")
    ap.add_argument("--window-s", type=float, default=100.0)
    ap.add_argument("--model", default="base")
    args = ap.parse_args()

    pq_path = args.parquet
    if pq_path.endswith(".txt") and os.path.exists(pq_path):
        pq_path = open(pq_path).read().strip()

    # 連続チャンネル（実録音）
    chans = []
    for i in range(4):
        y, sr = sf.read(f"{args.wav_prefix}{i}.wav")
        chans.append(y.astype(np.float32))
    n = min(len(c) for c in chans)
    n = min(n, int(args.window_s * SR))
    channels = np.stack([c[:n] for c in chans])

    ref_text, enroll_audio, spk_of = load_meeting(pq_path, args.meeting, args.window_s)
    names = [spk_of[m] for m in MIC_ORDER]
    refs = {spk_of[m]: ref_text[m] for m in MIC_ORDER}

    enroll = Enrollment.create("cpu")
    for m in MIC_ORDER:
        enroll.register(spk_of[m], enroll_audio[m])
    tr = Transcriber(args.model)

    print(f"# AMI {args.meeting} 実録音 {n/SR:.0f}s / 話者 {names}\n")
    print("# 参照(人手)語数:", {spk_of[m]: len(norm(ref_text[m])) for m in MIC_ORDER})

    # no-gate
    segs = run(channels, SR, names, enroll=None)
    for s in segs:
        s.is_own_voice = s.is_speech
    print(f"no-gate      WER={overall(refs, asr_text(channels, names, segs, tr), names)*100:.0f}%")

    # timing-gate
    segs = run(channels, SR, names, enroll=None)
    print(f"timing-gate  WER={overall(refs, asr_text(channels, names, segs, tr), names)*100:.0f}%")

    # phone-gate
    segs = phone_gate(channels, names, enroll)
    txt = asr_text(channels, names, segs, tr)
    print(f"phone-gate   WER={overall(refs, txt, names)*100:.0f}%")
    print("\n# phone-gate 各話者(先頭120字):")
    for nm in names:
        print(f"  [{nm}] {txt.get(nm,'')[:120]}")
    print("\n# 参照(人手, 先頭120字):")
    for m in MIC_ORDER:
        print(f"  [{spk_of[m]}] {ref_text[m][:120]}")


if __name__ == "__main__":
    main()
