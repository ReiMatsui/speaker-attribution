"""1本マイク vs 1人1マイク を、同じ会話・同じ正解で比較する.

問い: 1本のマイクで「誰が何を言ったか」はどこまでいけるか。

- 1本マイク : 全員の声を1本に混ぜる → Whisperで全体を文字起こし(単語タイムスタンプ)
              → 各単語の音から声紋を取り、登録話者の中で最も近い人に振り分ける。
              (物理の手がかりが無いので、声紋だけが頼り = 同時発話に弱い)
- 1人1マイク : 既存パイプライン(VAD→相関→連続マスク→Whisper)。

指標: 話者帰属つきWER。各話者の参照テキストと、その話者に割り当てられた書き起こしを比較。
      他人の発言を取り違えると、正しい話者では欠落・誤った話者では挿入として二重に効く。
正解: LibriSpeech公式トランスクリプト。
"""
from __future__ import annotations

import argparse
import difflib
import os

import numpy as np

from spkattr.synth import SynthConfig, synthesize
from spkattr.pipeline import run
from spkattr.embedding import Enrollment, cosine
from spkattr.asr import Transcriber, transcribe, transcribe_words

import sys
sys.path.insert(0, os.path.dirname(__file__))
from eval_separation import load_speakers, arrange, overall_wer  # type: ignore

SR = 16000


def assign_single_mic(mix, names, enroll, tr):
    """1本マイク: 全体ASR→各単語を最も近い登録話者へ振り分け."""
    words = transcribe_words(tr, mix, SR)
    hyp = {nm: [] for nm in names}
    embs = enroll.profiles
    for word, s, e in words:
        a = int(max(0, (s - 0.3) * SR))
        b = int(min(len(mix), (e + 0.3) * SR))
        if b - a < int(0.15 * SR):
            who = names[0]
        else:
            emb = enroll.embed(mix[a:b])
            who = max(names, key=lambda nm: cosine(emb, embs[nm]))
        hyp[who].append(word)
    return {nm: " ".join(ws) for nm, ws in hyp.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/tmp/dl_done.txt")
    ap.add_argument("--n-speakers", type=int, default=4)
    ap.add_argument("--overlap-s", type=float, default=2.0)
    ap.add_argument("--leak-db", type=float, default=-12.0,
                    help="1人1マイク側の漏れ込み量")
    ap.add_argument("--model", default="base")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "single_vs_multi.html"))
    args = ap.parse_args()

    pq_path = args.parquet
    if pq_path.endswith(".txt") and os.path.exists(pq_path):
        pq_path = open(pq_path).read().strip()
    spk = load_speakers(pq_path, args.n_speakers)
    names = [f"id{s}" for s in spk]
    clips = [spk[s][0] for s in spk]
    refs = {f"id{s}": spk[s][1] for s in spk}
    sources = arrange(clips, args.overlap_s)

    tr = Transcriber(args.model)
    enroll = Enrollment.create("cpu")
    for nm, src in zip(names, sources):
        enroll.register(nm, src)

    # 下限: 各話者クリーンを単独ASR
    A = {nm: tr.transcribe_span(src, SR) for nm, src in zip(names, sources)}
    wer_clean = overall_wer(refs, A, names)

    # 1本マイク
    mix = np.mean(np.stack(sources), axis=0)
    single = assign_single_mic(mix, names, enroll, tr)
    wer_single = overall_wer(refs, single, names)

    # 1人1マイク
    channels, _ = synthesize(sources, SynthConfig(sr=SR, leak_db=args.leak_db))
    segs = run(channels, SR, names, enroll=enroll)
    um = transcribe(channels, SR, names, segs, tr)
    multi = {nm: "" for nm in names}
    for u in um:
        multi[u.speaker] = (multi[u.speaker] + " " + u.text).strip()
    wer_multi = overall_wer(refs, multi, names)

    print(f"clean下限     WER={wer_clean*100:.0f}%")
    print(f"1本マイク      WER={wer_single*100:.0f}%")
    print(f"1人1マイク     WER={wer_multi*100:.0f}%")

    def per_spk_rows(hyp):
        out = ""
        for nm in names:
            rw = refs[nm].lower().split(); hw = hyp.get(nm, "").lower().split()
            sm = difflib.SequenceMatcher(a=rw, b=hw, autojunk=False)
            e = sum((max(i2-i1, j2-j1) if t == 'replace' else (i2-i1 if t == 'delete' else (j2-j1 if t == 'insert' else 0)))
                    for t, i1, i2, j1, j2 in sm.get_opcodes())
            out += f"<tr><td>{nm}</td><td class='n'>{e/max(1,len(rw))*100:.0f}%</td><td class='txt'>{hyp.get(nm,'') or '<i>―</i>'}</td></tr>"
        return out

    html = TEMPLATE.format(
        model=args.model, leak=args.leak_db, n=len(names),
        wclean=f"{wer_clean*100:.0f}", wsingle=f"{wer_single*100:.0f}", wmulti=f"{wer_multi*100:.0f}",
        single_rows=per_spk_rows(single), multi_rows=per_spk_rows(multi))
    with open(args.out, "w") as f:
        f.write(html)
    print("wrote", os.path.abspath(args.out))


TEMPLATE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>1本 vs 複数マイク</title>
<style>body{{font-family:-apple-system,"Hiragino Sans",sans-serif;max-width:860px;margin:24px auto;padding:0 16px;line-height:1.7;color:#222}}
h1{{font-size:19px}} h2{{font-size:15px;margin-top:22px}} table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}}
td,th{{border-bottom:1px solid #e8ebee;padding:7px 10px;text-align:left;vertical-align:top}} .n{{text-align:right;width:60px;font-variant-numeric:tabular-nums}}
.cards{{display:flex;gap:12px;margin:12px 0}} .card{{flex:1;background:#f5f7fa;border-radius:8px;padding:12px 14px}}
.card b{{font-size:24px;display:block}} .card span{{font-size:12px;color:#667}} .desc{{background:#f5f7fa;border-radius:8px;padding:12px 16px;font-size:13px}}</style></head><body>
<h1>1本マイク vs 1人1マイク（誰が何を言ったか, 正解=公式トランスクリプト）</h1>
<div class="desc">別話者 {n} 名・ターンテイキング会話・Whisper {model}。1本マイクは全員の声を1本に混ぜ、声紋だけで各単語を担当者に振り分ける。1人1マイクは漏れ込み {leak}dB。</div>
<div class="cards">
 <div class="card"><b>{wclean}%</b><span>クリーン下限</span></div>
 <div class="card"><b>{wsingle}%</b><span>1本マイク</span></div>
 <div class="card"><b>{wmulti}%</b><span>1人1マイク（本研究）</span></div>
</div>
<h2>1本マイクの内訳</h2><table><tr><th>話者</th><th class="n">WER</th><th>書き起こし</th></tr>{single_rows}</table>
<h2>1人1マイクの内訳</h2><table><tr><th>話者</th><th class="n">WER</th><th>書き起こし</th></tr>{multi_rows}</table>
</body></html>"""


if __name__ == "__main__":
    main()
