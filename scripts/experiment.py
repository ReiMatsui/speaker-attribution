"""漏れ込み除去アイデアのablation実験.

LibriSpeech公式トランスクリプトを正解に、複数手法のWERを比較する。
各手法は「ASRに渡す前にチャンネルをどう処理するか」が違う。

手法:
  m0 baseline      : 生の漏れ込み入りチャンネル(現状)
  m1 +tfmask       : 時間周波数の支配チャンネルマスクでクリーン化
  m2 +lsq          : 最小二乗FIRで漏れ込みキャンセル
  m3 +tfmask+robust: tfmask + ASR頑健化設定
  ref clean        : 漏れ込みなし(下限)

結果は experiment.html と標準出力に。
"""
from __future__ import annotations

import argparse
import difflib
import io
import os
import re
import time

import numpy as np
import soundfile as sf

from spkattr.synth import SynthConfig, synthesize
from spkattr.pipeline import run
from spkattr.embedding import Enrollment
from spkattr.asr import Transcriber, transcribe
from spkattr import cancel

SR = 16000


def load_speakers(parquet_path, n, min_s=4.0, max_s=12.0):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    chosen = {}
    for batch in pf.iter_batches(batch_size=200):
        d = batch.to_pydict()
        for au, txt, spk in zip(d["audio"], d["text"], d["speaker_id"]):
            if spk in chosen:
                continue
            y, sr = sf.read(io.BytesIO(au["bytes"]))
            if sr == SR and min_s <= len(y) / SR <= max_s:
                chosen[spk] = (y.astype(np.float32), txt)
                if len(chosen) >= n:
                    return chosen
    return chosen


def arrange(clips, overlap_s):
    overlap = int(SR * overlap_s)
    starts, cur = [], 0
    for c in clips:
        starts.append(cur); cur += max(1, len(c) - overlap)
    total = max(s + len(c) for s, c in zip(starts, clips))
    return [np.pad(c, (s, total - s - len(c))) for s, c in zip(starts, clips)]


def words(t):
    return re.sub(r"[^\w\s]", "", t.lower()).split()


def overall_wer(refs, sys_text, names):
    e = w = 0
    for nm in names:
        rw, hw = words(refs[nm]), words(sys_text.get(nm, ""))
        sm = difflib.SequenceMatcher(a=rw, b=hw, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "replace": e += max(i2 - i1, j2 - j1)
            elif tag == "delete": e += i2 - i1
            elif tag == "insert": e += j2 - j1
        w += len(rw)
    return e / max(1, w)


def transcribe_channels(proc_channels, names, segments, tr, robust):
    """proc_channels(処理後音声)をゲート区間で書き起こし、話者ごとに連結."""
    # transcribe() は asr.transcribe を流用するため、robust設定を一時適用
    utts = transcribe(proc_channels, SR, names, segments, tr)
    txt = {nm: "" for nm in names}
    for u in utts:
        txt[u.speaker] = (txt[u.speaker] + " " + u.text).strip()
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/tmp/dl_done.txt")
    ap.add_argument("--n-speakers", type=int, default=4)
    ap.add_argument("--leak-db", type=float, default=-12.0)
    ap.add_argument("--overlap-s", type=float, default=2.0)
    ap.add_argument("--model", default="base")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "experiment.html"))
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

    segments = run(channels, SR, names, enroll=enroll)

    results = []

    def record(tag, proc, robust, note):
        # robust設定を asr に伝えるため monkeypatch 的に既定を切替
        import spkattr.asr as A
        orig = A.Transcriber.transcribe_span
        A.Transcriber.transcribe_span = lambda self, a, s, robust=robust: orig(self, a, s, robust)
        t0 = time.time()
        txt = transcribe_channels(proc, names, segments, tr, robust)
        dt = time.time() - t0
        A.Transcriber.transcribe_span = orig
        w = overall_wer(refs, txt, names)
        results.append((tag, w, dt, note))
        print(f"{tag:18s} WER={w*100:5.1f}%  ({dt:.1f}s)  {note}")

    # 下限(クリーン)
    clean_txt = {nm: tr.transcribe_span(src, SR) for nm, src in zip(names, sources)}
    w_clean = overall_wer(refs, clean_txt, names)
    results.append(("ref clean", w_clean, 0.0, "漏れ込みなし=下限"))
    print(f"{'ref clean':18s} WER={w_clean*100:5.1f}%  (下限)")

    record("m0 baseline", channels, False, "生の漏れ込み入り(現状)")
    record("m1 +tfmask", cancel.tf_channel_mask(channels, SR), False, "T-F支配チャンネルマスク")
    record("m2 +lsq", cancel.lsq_cancel(channels), False, "最小二乗FIRキャンセル")
    record("m3 tfmask+robust", cancel.tf_channel_mask(channels, SR), True, "tfmask + ASR頑健化")

    base = next(w for t, w, _, _ in results if t == "m0 baseline")
    rows = "".join(
        f'<tr><td>{t}</td><td class="num">{w*100:.1f}%</td>'
        f'<td class="num">{(w-base)*100:+.1f}</td><td class="note">{note}</td></tr>'
        for t, w, _, note in results)
    html = TEMPLATE.format(model=args.model, leak=args.leak_db, n=len(names),
                           clean=f"{w_clean*100:.1f}", rows=rows)
    with open(args.out, "w") as f:
        f.write(html)
    print("wrote", os.path.abspath(args.out))


TEMPLATE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>漏れ込み除去 ablation</title><style>
 body{{font-family:-apple-system,"Hiragino Sans",sans-serif;max-width:820px;margin:24px auto;padding:0 16px;color:#222;line-height:1.7}}
 h1{{font-size:20px}} .desc{{background:#f5f7fa;border-radius:8px;padding:12px 16px;font-size:13px}}
 table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}}
 td,th{{border-bottom:1px solid #e8ebee;padding:8px 10px;text-align:left}}
 .num{{font-variant-numeric:tabular-nums;text-align:right;width:80px}}
</style></head><body>
<h1>漏れ込み除去アイデアの比較（WER, 正解=公式トランスクリプト）</h1>
<div class="desc">別話者 {n} 名、漏れ込み {leak}dB、Whisper {model}。
各手法は「ASRに渡す前に音をどう処理するか」が違う。3列目はm0(現状)からの増減pt(マイナスが改善)。
下限(漏れ込みなしクリーン)は <b>{clean}%</b>。</div>
<table><tr><th>手法</th><th>WER</th><th>vs m0</th><th>内容</th></tr>{rows}</table>
</body></html>"""


if __name__ == "__main__":
    main()
