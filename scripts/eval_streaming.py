"""ストリーミング(リアルタイム)化の核心的トレードオフを実測する.

現手法の強み = 本人の声を「長い連続音声」として一括ASRすること。
リアルタイム化 = まだ全部聞かないうちに短いチャンクで処理する必要がある。
→ チャンクを短くするほど低遅延だが、断片化でWERが悪化するはず。

この実験は、チャンク長 W（≒ そのぶんの待ち時間=レイテンシ）を振り、
各 W での話者帰属つきWERを測って、トレードオフ曲線を出す。
W=full は一括(オフライン)＝現手法の上限。

加えて各チャンクの計算時間からRTFを測り、計算が律速でない（待ちは音声長由来）ことを確認する。
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from spkattr.synth import SynthConfig, synthesize
from spkattr.pipeline import run
from spkattr.embedding import Enrollment
from spkattr.asr import Transcriber, merge_own_spans

import sys
sys.path.insert(0, os.path.dirname(__file__))
from eval_separation import load_speakers, arrange, overall_wer  # type: ignore

SR = 16000


def build_masked(channels, segs, names, pad_s=0.4, merge_gap_s=2.5, ramp_s=0.03):
    """各チャンネルの『本人時間以外を無音化した連続音声』を返す."""
    n_ch, n = channels.shape
    spans = merge_own_spans(segs, n_ch, names, max_gap_s=merge_gap_s)
    ramp = max(1, int(ramp_s * SR))
    kernel = np.ones(ramp) / ramp
    masked = np.zeros_like(channels)
    for ch in range(n_ch):
        m = np.zeros(n)
        for s, e in spans.get(ch, []):
            a = int(max(0, (s - pad_s) * SR)); b = int(min(n, (e + pad_s) * SR))
            m[a:b] = 1.0
        m = np.clip(np.convolve(m, kernel, mode="same"), 0, 1)
        masked[ch] = channels[ch] * m
    return masked


def transcribe_chunked(masked, names, tr, chunk_s):
    """masked音声を chunk_s 秒ごとに区切ってASRし、話者ごとに連結.

    chunk_s=None なら一括(オフライン上限)。
    返り値: (話者->テキスト, 計算時間秒)
    """
    n_ch, n = masked.shape
    txt = {nm: "" for nm in names}
    t0 = time.time()
    for ch in range(n_ch):
        if chunk_s is None:
            pieces = [masked[ch]]
        else:
            step = int(chunk_s * SR)
            pieces = [masked[ch, i:i + step] for i in range(0, n, step)]
        parts = []
        for p in pieces:
            if np.abs(p).max() < 1e-4:
                continue
            t = tr.transcribe_span(p, SR)
            if t:
                parts.append(t)
        txt[names[ch]] = " ".join(parts).strip()
    return txt, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/tmp/dl_done.txt")
    ap.add_argument("--n-speakers", type=int, default=4)
    ap.add_argument("--overlap-s", type=float, default=2.0)
    ap.add_argument("--leak-db", type=float, default=-12.0)
    ap.add_argument("--model", default="base")
    ap.add_argument("--chunks", default="1,2,4,8")  # 秒。fullは別途
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "streaming.html"))
    args = ap.parse_args()

    pq_path = args.parquet
    if pq_path.endswith(".txt") and os.path.exists(pq_path):
        pq_path = open(pq_path).read().strip()
    spk = load_speakers(pq_path, args.n_speakers)
    names = [f"id{s}" for s in spk]
    clips = [spk[s][0] for s in spk]
    refs = {f"id{s}": spk[s][1] for s in spk}
    sources = arrange(clips, args.overlap_s)
    audio_s = max(len(s) for s in sources) / SR

    tr = Transcriber(args.model)
    enroll = Enrollment.create("cpu")
    for nm, src in zip(names, sources):
        enroll.register(nm, src)
    channels, _ = synthesize(sources, SynthConfig(sr=SR, leak_db=args.leak_db))
    segs = run(channels, SR, names, enroll=enroll)
    masked = build_masked(channels, segs, names)

    rows = []
    # full（オフライン上限）
    txt, dt = transcribe_chunked(masked, names, tr, None)
    w = overall_wer(refs, txt, names)
    rows.append(("一括(オフライン)", None, w, dt))
    print(f"full      WER={w*100:.0f}%  rtf={dt/audio_s:.2f}")
    for cs in [float(x) for x in args.chunks.split(",")]:
        txt, dt = transcribe_chunked(masked, names, tr, cs)
        w = overall_wer(refs, txt, names)
        rows.append((f"{cs:g}秒チャンク", cs, w, dt))
        print(f"chunk {cs:g}s WER={w*100:.0f}%  rtf={dt/audio_s:.2f}")

    with open(args.out, "w") as f:
        f.write(render(rows, audio_s, args.model))
    print("wrote", os.path.abspath(args.out))


def render(rows, audio_s, model):
    trs = ""
    for label, cs, w, dt in rows:
        lat = "—（全体待ち）" if cs is None else f"約 {cs:g} 秒"
        trs += (f"<tr><td>{label}</td><td class='n'>{w*100:.0f}%</td>"
                f"<td>{lat}</td><td class='n'>{dt/audio_s:.2f}</td></tr>")
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>ストリーミング検証</title>
<style>body{{font-family:-apple-system,"Hiragino Sans",sans-serif;max-width:780px;margin:24px auto;padding:0 16px;line-height:1.7;color:#222}}
h1{{font-size:19px}} table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}} td,th{{border-bottom:1px solid #e8ebee;padding:8px 11px;text-align:left}}
.n{{text-align:right;font-variant-numeric:tabular-nums;width:90px}} .desc{{background:#f5f7fa;border-radius:8px;padding:12px 16px;font-size:13px}}</style></head><body>
<h1>ストリーミング化のトレードオフ（話者帰属つきWER）</h1>
<div class="desc">同じ会話・Whisper {model}・正解=公式トランスクリプト。チャンク長を短くするほど低遅延だが、
本人の声を細切れにして文脈が失われWERが悪化する。RTF&lt;1なら計算は実時間に間に合う（待ちは音声長由来であり計算律速ではない）。</div>
<table><tr><th>処理方式</th><th class="n">WER</th><th>1語あたりの待ち（レイテンシの主因）</th><th class="n">計算RTF</th></tr>{trs}</table>
<p style="font-size:12px;color:#778">レイテンシの主因は「チャンクが埋まるまでの待ち時間」。短くするとWERが上がる＝現手法の弱点。
対策候補: オーバーラップ付きチャンク／直前文脈の保持／ストリーミング対応ASR。</p>
</body></html>"""


if __name__ == "__main__":
    main()
