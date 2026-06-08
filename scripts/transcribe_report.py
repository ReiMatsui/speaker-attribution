"""文字起こし結果をHTMLで出力(ゲートあり/なし比較)."""
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
COLORS = ["#34c759", "#ff9500", "#af52de", "#5ac8fa"]


def arrange(clips, overlap_s):
    overlap = int(SR * overlap_s)
    starts, cur = [], 0
    for c in clips:
        starts.append(cur); cur += max(1, len(c) - overlap)
    total = max(s + len(c) for s, c in zip(starts, clips))
    return [np.pad(c, (s, total - s - len(c))) for s, c in zip(starts, clips)]


def run_one(channels, names, enroll, tr, gate):
    segments = run(channels, SR, names, enroll=enroll)
    if not gate:
        for s in segments:
            s.is_own_voice = s.is_speech
    t0 = time.time()
    utts = transcribe(channels, SR, names, segments, tr)
    return utts, time.time() - t0


def rows_html(utts, names):
    out = []
    for u in utts:
        col = COLORS[names.index(u.speaker) % len(COLORS)]
        out.append(
            f'<tr><td class="t">{u.start:.1f}–{u.end:.1f}s</td>'
            f'<td><b style="color:{col}">{u.speaker}</b></td>'
            f'<td>{u.text}</td></tr>')
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default=os.path.join(os.path.dirname(__file__), "..", "samples"))
    ap.add_argument("--leak-db", type=float, default=-12.0)
    ap.add_argument("--overlap-s", type=float, default=2.0)
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "transcript.html"))
    args = ap.parse_args()

    import librosa
    files = sorted(glob.glob(os.path.join(args.samples_dir, "*.wav")))
    clips, names = [], []
    for p in files:
        y, _ = librosa.load(p, sr=SR)
        clips.append(y); names.append(os.path.basename(p).split(".")[0])
    sources = arrange(clips, args.overlap_s)
    channels, _ = synthesize(sources, SynthConfig(sr=SR, leak_db=args.leak_db))
    audio_s = channels.shape[1] / SR

    enroll = Enrollment.create("cpu")
    for nm, src in zip(names, sources):
        enroll.register(nm, src)
    tr = Transcriber(args.model)

    gate_u, gate_t = run_one(channels, names, enroll, tr, True)
    nog_u, nog_t = run_one(channels, names, enroll, tr, False)

    html = TEMPLATE.format(
        model=args.model, leak=args.leak_db, audio=f"{audio_s:.1f}",
        grtf=f"{gate_t/audio_s:.2f}", nrtf=f"{nog_t/audio_s:.2f}",
        gate_rows=rows_html(gate_u, names), nogate_rows=rows_html(nog_u, names),
        n_gate=len(gate_u), n_nogate=len(nog_u),
    )
    with open(args.out, "w") as f:
        f.write(html)
    print("wrote", os.path.abspath(args.out))


TEMPLATE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>話者付き文字起こし 結果</title><style>
 body{{font-family:-apple-system,"Hiragino Sans",sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#222;line-height:1.6}}
 h1{{font-size:20px}} h2{{font-size:15px;margin:22px 0 6px}}
 .desc{{background:#f5f7fa;border-radius:8px;padding:12px 16px;font-size:13px}}
 .cols{{display:flex;gap:18px;flex-wrap:wrap}} .col{{flex:1;min-width:330px}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 td{{border-bottom:1px solid #eef0f3;padding:5px 8px;vertical-align:top}}
 td.t{{color:#889;white-space:nowrap;font-variant-numeric:tabular-nums;width:78px}}
 .tag{{display:inline-block;border-radius:6px;padding:1px 8px;font-size:12px;color:#fff}}
 .good{{background:#34c759}} .bad{{background:#ff3b30}}
</style></head><body>
<h1>話者付き文字起こし 結果</h1>
<div class="desc">
モデル=Whisper {model} / 漏れ込み {leak}dB / 音声 {audio}秒。
左=<b>漏れ込みゲートあり</b>(本研究の出力)、右=<b>ゲートなし</b>(各マイク素通し)。
ゲートなしだと、隣の人の声を自分の発言として二重に拾う(誤帰属)。<br>
リアルタイム係数 RTF: ゲートあり <b>{grtf}</b> / なし {nrtf}(1未満＝実時間より速い)。
</div>
<div class="cols">
 <div class="col"><h2><span class="tag good">ゲートあり</span> {n_gate}発話</h2>
   <table>{gate_rows}</table></div>
 <div class="col"><h2><span class="tag bad">ゲートなし</span> {n_nogate}発話</h2>
   <table>{nogate_rows}</table></div>
</div>
</body></html>"""


if __name__ == "__main__":
    main()
