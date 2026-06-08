"""LibriSpeech(公式参照テキストあり)で話者帰属+文字起こしを評価する.

正解 = LibriSpeech の公式トランスクリプト(人手の参照)。Whisper出力ではない。
手順:
  1. dev-clean の parquet から、別話者 N 名の発話を1本ずつ取り出す(音声+正解テキスト)。
  2. 会話状に並べ、漏れ込みを合成。
  3. パイプライン(VAD→相関→声紋→Whisper)で話者付き書き起こしを生成。
  4. 各話者の system テキストを公式参照と比較し WER を出す(jiwer があれば使用)。
出力: accuracy_librispeech.html
"""
from __future__ import annotations

import argparse
import io
import os
import re
import difflib

import numpy as np
import soundfile as sf

from spkattr.synth import SynthConfig, synthesize
from spkattr.pipeline import run
from spkattr.embedding import Enrollment
from spkattr.asr import Transcriber, transcribe

SR = 16000


def load_speakers(parquet_path, n_speakers, min_s=4.0, max_s=12.0):
    """別話者 n_speakers 名から、長さが手頃な発話を1つずつ取り出す."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    chosen = {}  # speaker_id -> (audio float32, text)
    for batch in pf.iter_batches(batch_size=200):
        d = batch.to_pydict()
        for au, txt, spk in zip(d["audio"], d["text"], d["speaker_id"]):
            if spk in chosen:
                continue
            y, sr = sf.read(io.BytesIO(au["bytes"]))
            if sr != SR:
                continue
            dur = len(y) / SR
            if min_s <= dur <= max_s:
                chosen[spk] = (y.astype(np.float32), txt)
                if len(chosen) >= n_speakers:
                    return chosen
    return chosen


def arrange(clips, overlap_s):
    overlap = int(SR * overlap_s)
    starts, cur = [], 0
    for c in clips:
        starts.append(cur); cur += max(1, len(c) - overlap)
    total = max(s + len(c) for s, c in zip(starts, clips))
    return [np.pad(c, (s, total - s - len(c))) for s, c in zip(starts, clips)]


def norm(t):
    return re.sub(r"[^\w\s]", "", t.lower()).split()


def wer_and_diff(ref, hyp):
    rw, hw = norm(ref), norm(hyp)
    sm = difflib.SequenceMatcher(a=rw, b=hw, autojunk=False)
    rhtml, hhtml = [], []
    err = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            rhtml += rw[i1:i2]; hhtml += hw[j1:j2]
        elif tag == "replace":
            err += max(i2 - i1, j2 - j1)
            rhtml += [f'<span class="miss">{w}</span>' for w in rw[i1:i2]]
            hhtml += [f'<span class="extra">{w}</span>' for w in hw[j1:j2]]
        elif tag == "delete":
            err += i2 - i1
            rhtml += [f'<span class="miss">{w}</span>' for w in rw[i1:i2]]
        elif tag == "insert":
            err += j2 - j1
            hhtml += [f'<span class="extra">{w}</span>' for w in hw[j1:j2]]
    return err / max(1, len(rw)), " ".join(rhtml), " ".join(hhtml), len(rw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/tmp/dl_done.txt",
                    help="parquetパス、または完了パスを書いたテキスト")
    ap.add_argument("--n-speakers", type=int, default=4)
    ap.add_argument("--leak-db", type=float, default=-12.0)
    ap.add_argument("--overlap-s", type=float, default=2.0)
    ap.add_argument("--model", default="base")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "accuracy_librispeech.html"))
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
    utts = transcribe(channels, SR, names, segments, tr)
    sys_text = {nm: "" for nm in names}
    for u in utts:
        sys_text[u.speaker] = (sys_text[u.speaker] + " " + u.text).strip()

    rows, tot_err, tot_w = [], 0.0, 0
    for nm in names:
        w, rh, hh, nw = wer_and_diff(refs[nm], sys_text[nm])
        tot_err += w * nw; tot_w += nw
        rows.append(f"""<div class="spk"><div class="spkhead"><b>{nm}</b>
          <span class="wer">WER {w*100:.0f}%</span></div>
          <div class="line"><span class="lbl">正解(公式)</span><div class="txt">{rh}</div></div>
          <div class="line"><span class="lbl">システム</span><div class="txt">{hh or '<i>(取りこぼし)</i>'}</div></div></div>""")
    overall = tot_err / max(1, tot_w)

    html = TEMPLATE.format(model=args.model, leak=args.leak_db,
                           n=len(names), overall=f"{overall*100:.0f}", rows="".join(rows))
    with open(args.out, "w") as f:
        f.write(html)
    print("wrote", os.path.abspath(args.out), f"overall WER={overall*100:.0f}%")


TEMPLATE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>LibriSpeech 精度レポート</title><style>
 body{{font-family:-apple-system,"Hiragino Sans",sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#222;line-height:1.7}}
 h1{{font-size:20px}} .desc{{background:#f5f7fa;border-radius:8px;padding:12px 16px;font-size:13px}}
 .miss{{color:#ff3b30;text-decoration:line-through}} .extra{{background:#ffe08a;border-radius:3px;padding:0 2px}}
 .overall{{font-size:15px;margin:14px 0}} .spk{{border:1px solid #e2e6ea;border-radius:10px;padding:10px 14px;margin:10px 0}}
 .spkhead{{display:flex;justify-content:space-between}} .wer{{color:#667;font-size:13px}}
 .line{{display:flex;gap:8px;margin:3px 0}} .lbl{{width:84px;flex:none;font-size:12px;color:#778;text-align:right}} .txt{{flex:1}}
</style></head><body>
<h1>LibriSpeech 精度レポート（正解＝公式トランスクリプト）</h1>
<div class="desc">
正解は <b>LibriSpeech の人手による公式参照テキスト</b>（Whisper出力ではない）。別話者 {n} 名の発話を
会話状に並べ、漏れ込み {leak}dB を合成し、パイプライン(VAD→相関→声紋→Whisper {model})で書き起こした結果。<br>
<span class="miss">赤取り消し線</span>=落とした語、<span class="extra">黄</span>=余分/誤りの語。
</div>
<div class="overall">全体WER: <b>{overall}%</b>（0%が完璧）</div>
{rows}
</body></html>"""


if __name__ == "__main__":
    main()
