"""分離評価: 誤りの内訳を切り分ける ablation.

同じLibriSpeech話者で3条件のWER(正解=公式トランスクリプト)を比較する:
  A. クリーン        : 漏れ込みなしの本人音声をASR → ASR自体の誤り下限
  B. 漏れ込み・ゲートなし: 各マイク素通しでASR     → 漏れ込みの害がそのまま出る
  C. 漏れ込み・ゲートあり: 本研究(相関+声紋で除去) → 関門で回復した結果

切り分け:
  (A)           = Whisperの地力(これ以上良くはならない)
  (B - A)       = 漏れ込みが持ち込む害
  (B - C)       = 本研究のゲートが回復させた分
  (C - A)       = ゲート後に残る本研究の残差(本当の伸びしろ)
"""
from __future__ import annotations

import argparse
import difflib
import io
import os
import re

import numpy as np
import soundfile as sf

from spkattr.synth import SynthConfig, synthesize
from spkattr.pipeline import run
from spkattr.embedding import Enrollment
from spkattr.asr import Transcriber, transcribe

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


def wer(ref, hyp):
    rw, hw = words(ref), words(hyp)
    sm = difflib.SequenceMatcher(a=rw, b=hw, autojunk=False)
    err = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            err += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            err += i2 - i1
        elif tag == "insert":
            err += j2 - j1
    return err, len(rw)


def overall_wer(refs, sys_text, names):
    e = w = 0
    for nm in names:
        ee, ww = wer(refs[nm], sys_text.get(nm, ""))
        e += ee; w += ww
    return e / max(1, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/tmp/dl_done.txt")
    ap.add_argument("--n-speakers", type=int, default=4)
    ap.add_argument("--leak-db", type=float, default=-12.0)
    ap.add_argument("--overlap-s", type=float, default=2.0)
    ap.add_argument("--model", default="base")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "separation.html"))
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

    # A. クリーン
    A = {nm: tr.transcribe_span(src, SR) for nm, src in zip(names, sources)}
    wer_A = overall_wer(refs, A, names)

    # 共通: パイプライン区間
    segments = run(channels, SR, names, enroll=enroll)

    # C. ゲートあり(本研究)
    utts_C = transcribe(channels, SR, names, segments, tr)
    C = {nm: "" for nm in names}
    for u in utts_C:
        C[u.speaker] = (C[u.speaker] + " " + u.text).strip()
    wer_C = overall_wer(refs, C, names)

    # B. ゲートなし(全 active 窓を本人扱い)
    for s in segments:
        s.is_own_voice = s.is_speech
    utts_B = transcribe(channels, SR, names, segments, tr)
    B = {nm: "" for nm in names}
    for u in utts_B:
        B[u.speaker] = (B[u.speaker] + " " + u.text).strip()
    wer_B = overall_wer(refs, B, names)

    rows = [
        ("A. クリーン（漏れ込みなし）", wer_A, "Whisperの地力。これ以上は良くならない下限。"),
        ("B. 漏れ込み・ゲートなし", wer_B, "各マイク素通し。漏れ込みの害がそのまま出る。"),
        ("C. 漏れ込み・ゲートあり（本研究）", wer_C, "相関＋声紋で漏れ込み除去。"),
    ]
    html = TEMPLATE.format(
        model=args.model, leak=args.leak_db, n=len(names),
        rows="".join(
            f'<tr><td>{n}</td><td class="num">{w*100:.0f}%</td><td class="note">{d}</td></tr>'
            for n, w, d in rows),
        d_leak=f"{(wer_B - wer_A)*100:+.0f}",
        d_gate=f"{(wer_B - wer_C)*100:+.0f}",
        d_resid=f"{(wer_C - wer_A)*100:+.0f}",
    )
    with open(args.out, "w") as f:
        f.write(html)
    print(f"A(clean)={wer_A*100:.0f}%  B(no-gate)={wer_B*100:.0f}%  C(gated)={wer_C*100:.0f}%")
    print("wrote", os.path.abspath(args.out))


TEMPLATE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>分離評価</title><style>
 body{{font-family:-apple-system,"Hiragino Sans",sans-serif;max-width:820px;margin:24px auto;padding:0 16px;color:#222;line-height:1.7}}
 h1{{font-size:20px}} .desc{{background:#f5f7fa;border-radius:8px;padding:12px 16px;font-size:13px}}
 table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}}
 td,th{{border-bottom:1px solid #e8ebee;padding:8px 10px;text-align:left}}
 .num{{font-variant-numeric:tabular-nums;font-weight:700;font-size:16px;width:70px}}
 .note{{color:#667;font-size:12px}}
 .break{{background:#fff;border:1px solid #e2e6ea;border-radius:10px;padding:12px 16px;margin:12px 0}}
 .break b{{font-size:18px}} .break div{{margin:6px 0}}
 .good{{color:#0a84ff}} .bad{{color:#ff3b30}}
</style></head><body>
<h1>分離評価（誤りの内訳）</h1>
<div class="desc">
正解＝LibriSpeech公式トランスクリプト。別話者 {n} 名、漏れ込み {leak}dB、Whisper {model}。
同じ音声で3条件のWERを比較し、誤りがどこ由来かを切り分ける。
</div>
<table>
 <tr><th>条件</th><th>WER</th><th></th></tr>
 {rows}
</table>
<div class="break">
 <div>漏れ込みが持ち込む害　<b>(B − A) = {d_leak}pt</b>　← 何もしないとこれだけ悪化</div>
 <div class="good">本研究のゲートが回復させた分　<b>(B − C) = {d_gate}pt</b>　← 関門の効果</div>
 <div class="bad">ゲート後に残る残差　<b>(C − A) = {d_resid}pt</b>　← まだ伸ばせる本当の余地</div>
</div>
<p style="font-size:12px;color:#778">
 (C − A) が大きいほど、漏れ込み除去の改善余地が残っている。ここを縮めるのが次の研究目標。
</p>
</body></html>"""


if __name__ == "__main__":
    main()
