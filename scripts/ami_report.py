"""AMI実会議の結果を「聴いて・読んで・正解と比べる」HTMLにする.

各話者ごとに:
  - 生録音(このマイク)の音声プレイヤー … 本物の漏れ込みが聴ける
  - ゲート後の音声プレイヤー … 漏れだけの時間が無音化されたもの
  - 正解(人手) / ゲートなし書き起こし / ゲートあり書き起こし の3行
余計な指標は乗せず、本質（ゲートなしは全員混ざる→ゲートで本人だけ）が分かる形。
"""
from __future__ import annotations

import argparse
import base64
import copy
import io
import os
import re

import numpy as np
import soundfile as sf

from spkattr.pipeline import run
from spkattr.asr import Transcriber, transcribe, merge_own_spans

SR = 16000
MIC = ["H00", "H01", "H02", "H03"]


def b64(x):
    x = x / (np.abs(x).max() + 1e-9) * 0.9
    buf = io.BytesIO(); sf.write(buf, x.astype(np.float32), SR, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def norm(t):
    return re.sub(r"[^\w\s]", "", t.lower()).split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--wav-prefix", default="ami_h")
    ap.add_argument("--meeting", default="EN2002a")
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--model", default="base")
    ap.add_argument("--out", default="ami_report.html")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    f = pq.ParquetFile(args.parquet)
    refs = {m: [] for m in MIC}; spk = {}; seen = False
    for b in f.iter_batches(batch_size=2000,
                            columns=["meeting_id", "microphone_id", "speaker_id", "begin_time", "text"]):
        d = b.to_pydict(); hit = False
        for i in range(len(d["meeting_id"])):
            if d["meeting_id"][i] != args.meeting:
                continue
            hit = True; seen = True; m = d["microphone_id"][i]
            if m in refs:
                spk[m] = d["speaker_id"][i]
                if d["begin_time"][i] < args.window_s:
                    refs[m].append((d["begin_time"][i], d["text"][i]))
        if seen and not hit:
            break
    names = [spk[m] for m in MIC]
    ref_text = {spk[m]: " ".join(t for _, t in sorted(refs[m])) for m in MIC}

    chans = [sf.read(f"{args.wav_prefix}{i}.wav")[0].astype(np.float32) for i in range(4)]
    n = min(min(len(c) for c in chans), int(args.window_s * SR))
    channels = np.stack([c[:n] for c in chans])

    tr = Transcriber(args.model)

    def to_txt(segs):
        u = transcribe(channels, SR, names, segs, tr)
        d = {nm: "" for nm in names}
        for x in u:
            d[x.speaker] = (d[x.speaker] + " " + x.text).strip()
        return d

    segs = run(channels, SR, names, enroll=None)
    ng = copy.deepcopy(segs)
    for x in ng:
        x.is_own_voice = x.is_speech
    no_gate = to_txt(ng)
    gated = to_txt(segs)

    # ゲート後の音声（本人時間以外を無音化）
    spans = merge_own_spans(segs, 4, names, max_gap_s=2.5)
    ramp = int(0.03 * SR); k = np.ones(ramp) / ramp
    gated_audio = {}
    for ch in range(4):
        m = np.zeros(n)
        for s, e in spans.get(ch, []):
            a = int(max(0, (s - 0.4) * SR)); bb = int(min(n, (e + 0.4) * SR)); m[a:bb] = 1.0
        m = np.clip(np.convolve(m, k, mode="same"), 0, 1)
        gated_audio[names[ch]] = channels[ch] * m

    rows = ""
    for ch, nm in enumerate(names):
        rows += f"""
        <div class="spk">
          <div class="hd"><b>{nm}</b>
            <span class="pl"><span class="lab">生録音(このマイク)</span>
              <audio controls preload="none" src="data:audio/wav;base64,{b64(channels[ch])}"></audio>
              <span class="lab">ゲート後</span>
              <audio controls preload="none" src="data:audio/wav;base64,{b64(gated_audio[nm])}"></audio></span></div>
          <div class="ln"><span class="t t1">正解(人手)</span><div class="x">{ref_text[nm] or '―'}</div></div>
          <div class="ln"><span class="t t2">ゲートなし</span><div class="x bad">{no_gate.get(nm,'') or '―'}</div></div>
          <div class="ln"><span class="t t3">ゲートあり</span><div class="x good">{gated.get(nm,'') or '―'}</div></div>
        </div>"""

    html = TEMPLATE.format(meeting=args.meeting, secs=int(n / SR), model=args.model, rows=rows)
    with open(args.out, "w") as fo:
        fo.write(html)
    print("wrote", os.path.abspath(args.out))


TEMPLATE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>AMI 実会議 結果</title><style>
 body{{font-family:-apple-system,"Hiragino Sans",sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#222;line-height:1.7}}
 h1{{font-size:19px}} .desc{{background:#f5f7fa;border-radius:8px;padding:12px 16px;font-size:13px}}
 audio{{height:30px;vertical-align:middle}} .pl .lab{{font-size:11px;color:#667;margin:0 4px 0 12px}}
 .spk{{border:1px solid #e2e6ea;border-radius:10px;padding:10px 14px;margin:12px 0}}
 .hd{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px}}
 .ln{{display:flex;gap:8px;margin:4px 0}} .t{{width:74px;flex:none;font-size:12px;text-align:right;padding-top:1px}}
 .t1{{color:#555}} .t2{{color:#c0392b}} .t3{{color:#0a6}} .x{{flex:1}}
 .bad{{color:#c0392b}} .good{{color:#0a6}}
</style></head><body>
<h1>AMI 実会議の結果（{meeting}, 先頭{secs}秒, Whisper {model}）</h1>
<div class="desc">
本物の1人1ヘッドセットマイク会議の録音（実際の漏れ込みあり）。<b>正解は人手の書き起こし</b>。<br>
各マイクの「生録音」を聴くと、隣の人の声が漏れて入っているのが分かる。
<b style="color:#c0392b">ゲートなし</b>はその漏れまで全部書き起こして<b>全員の言葉が混ざる</b>。
<b style="color:#0a6">ゲートあり</b>（本研究）は漏れだけの時間を無音化し、<b>ほぼ本人だけ</b>に戻す。
「ゲート後」の音声を聴くと、他人がしゃべる時間が消えているのが分かる。
</div>
{rows}
</body></html>"""


if __name__ == "__main__":
    main()
