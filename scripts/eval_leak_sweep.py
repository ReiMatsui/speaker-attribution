"""漏れ込み量(leak-db)を振って、ゲートの効果を特性化するスイープ.

各 leak について A(クリーン)/B(ゲートなし)/C(ゲートあり=本研究) のWERを測り、
「どの漏れ込み量でゲートが効くか」を1枚の表にまとめる(leak_sweep.html)。
"""
from __future__ import annotations

import argparse
import io
import os

import numpy as np
import soundfile as sf

from spkattr.synth import SynthConfig, synthesize
from spkattr.pipeline import run
from spkattr.embedding import Enrollment
from spkattr.asr import Transcriber, transcribe
from scripts.eval_separation import load_speakers, arrange, overall_wer  # type: ignore

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/tmp/dl_done.txt")
    ap.add_argument("--n-speakers", type=int, default=4)
    ap.add_argument("--overlap-s", type=float, default=2.0)
    ap.add_argument("--model", default="base")
    ap.add_argument("--leaks", default="-12,-6,0")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "leak_sweep.html"))
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
    A = {nm: tr.transcribe_span(src, SR) for nm, src in zip(names, sources)}
    wer_A = overall_wer(refs, A, names)

    rows = []
    for leak in [float(x) for x in args.leaks.split(",")]:
        channels, _ = synthesize(sources, SynthConfig(sr=SR, leak_db=leak))
        enroll = Enrollment.create("cpu")
        for nm, src in zip(names, sources):
            enroll.register(nm, src)
        segs = run(channels, SR, names, enroll=enroll)

        # C: ゲートあり
        uc = transcribe(channels, SR, names, segs, tr)
        C = {nm: "" for nm in names}
        for u in uc:
            C[u.speaker] = (C[u.speaker] + " " + u.text).strip()
        wer_C = overall_wer(refs, C, names)
        # B: ゲートなし
        for s in segs:
            s.is_own_voice = s.is_speech
        ub = transcribe(channels, SR, names, segs, tr)
        B = {nm: "" for nm in names}
        for u in ub:
            B[u.speaker] = (B[u.speaker] + " " + u.text).strip()
        wer_B = overall_wer(refs, B, names)
        print(f"leak {leak:+.0f}dB : clean={wer_A*100:.0f}% no-gate={wer_B*100:.0f}% gated={wer_C*100:.0f}%")
        rows.append((leak, wer_A, wer_B, wer_C))

    with open(args.out, "w") as f:
        f.write(render(rows, args.model))
    print("wrote", os.path.abspath(args.out))


def render(rows, model):
    trs = ""
    for leak, a, b, c in rows:
        trs += (f'<tr><td>{leak:+.0f} dB</td><td class="n">{a*100:.0f}%</td>'
                f'<td class="n bad">{b*100:.0f}%</td><td class="n good">{c*100:.0f}%</td></tr>')
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>漏れ込みスイープ</title>
<style>body{{font-family:-apple-system,"Hiragino Sans",sans-serif;max-width:760px;margin:24px auto;padding:0 16px;line-height:1.7}}
table{{border-collapse:collapse;width:100%;margin:14px 0}} td,th{{border-bottom:1px solid #e8ebee;padding:8px 10px;text-align:left}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} .good{{color:#0a84ff;font-weight:700}} .bad{{color:#ff3b30}}
.desc{{background:#f5f7fa;border-radius:8px;padding:12px 16px;font-size:13px}}</style></head><body>
<h1>漏れ込み量とゲートの効果（WER, 正解=公式トランスクリプト, Whisper {model}）</h1>
<div class="desc">漏れ込みが大きいほど、ゲートなし(漏れ込み除去なし)は他人の声まで書き起こして崩壊する。
本研究のゲートはそれを抑える。漏れ込みが小さい領域ではASR単体でほぼ解け、ゲートの必要性は低い。</div>
<table><tr><th>漏れ込み</th><th>クリーン下限</th><th>ゲートなし</th><th>ゲートあり(本研究)</th></tr>{trs}</table>
<p style="font-size:12px;color:#778">0dB(等音量)はゲートありでも崩れる最難領域＝次の研究フロンティア(target-speaker抽出等)。</p>
</body></html>"""


if __name__ == "__main__":
    main()
