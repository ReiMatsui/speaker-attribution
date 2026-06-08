"""正解 vs システム出力 を話者ごとに並べて差分表示するHTMLを生成.

正解   = 各話者のクリーン音声(漏れ込みなし)をASRした「本当に言った内容」。
システム = 漏れ込み入りマイクから、ゲートを通して得た書き起こし。
語ごとに difflib で対応を取り、欠落=赤取り消し線 / 余分=黄ハイライト。
話者ごとに簡易WER(単語誤り率)も出す。
"""
from __future__ import annotations

import argparse
import difflib
import glob
import os
import re

import numpy as np

from spkattr.synth import SynthConfig, synthesize
from spkattr.pipeline import run
from spkattr.embedding import Enrollment
from spkattr.asr import Transcriber, transcribe

SR = 16000


def arrange(clips, overlap_s):
    overlap = int(SR * overlap_s)
    starts, cur = [], 0
    for c in clips:
        starts.append(cur); cur += max(1, len(c) - overlap)
    total = max(s + len(c) for s, c in zip(starts, clips))
    return [np.pad(c, (s, total - s - len(c))) for s, c in zip(starts, clips)]


def norm_words(text):
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def diff_html_and_wer(truth, system):
    """正解語列と出力語列を比較。HTML2本(正解側/出力側)とWERを返す."""
    tw, sw = norm_words(truth), norm_words(system)
    sm = difflib.SequenceMatcher(a=tw, b=sw, autojunk=False)
    truth_html, sys_html = [], []
    sub = dele = ins = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            truth_html += tw[i1:i2]
            sys_html += sw[j1:j2]
        elif tag == "replace":
            sub += max(i2 - i1, j2 - j1)
            truth_html += [f'<span class="miss">{w}</span>' for w in tw[i1:i2]]
            sys_html += [f'<span class="extra">{w}</span>' for w in sw[j1:j2]]
        elif tag == "delete":
            dele += (i2 - i1)
            truth_html += [f'<span class="miss">{w}</span>' for w in tw[i1:i2]]
        elif tag == "insert":
            ins += (j2 - j1)
            sys_html += [f'<span class="extra">{w}</span>' for w in sw[j1:j2]]
    n = max(1, len(tw))
    wer = (sub + dele + ins) / n
    return " ".join(truth_html), " ".join(sys_html), wer, len(tw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default=os.path.join(os.path.dirname(__file__), "..", "samples"))
    ap.add_argument("--leak-db", type=float, default=-12.0)
    ap.add_argument("--overlap-s", type=float, default=2.0)
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "accuracy.html"))
    args = ap.parse_args()

    import librosa
    files = sorted(glob.glob(os.path.join(args.samples_dir, "*.wav")))
    clips, names = [], []
    for p in files:
        y, _ = librosa.load(p, sr=SR)
        clips.append(y); names.append(os.path.basename(p).split(".")[0])
    sources = arrange(clips, args.overlap_s)
    channels, _ = synthesize(sources, SynthConfig(sr=SR, leak_db=args.leak_db))

    enroll = Enrollment.create("cpu")
    for nm, src in zip(names, sources):
        enroll.register(nm, src)
    tr = Transcriber(args.model)

    # 正解: クリーン音源を1チャンネルずつASR
    truth = {}
    for nm, src in zip(names, sources):
        truth[nm] = tr.transcribe_span(src, SR)

    # システム: 漏れ込み入りマイク + ゲート
    segments = run(channels, SR, names, enroll=enroll)
    utts = transcribe(channels, SR, names, segments, tr)
    sys_text = {nm: "" for nm in names}
    for u in utts:
        sys_text[u.speaker] = (sys_text[u.speaker] + " " + u.text).strip()

    rows, tot_err, tot_words = [], 0.0, 0
    for nm in names:
        th, sh, wer, nw = diff_html_and_wer(truth[nm], sys_text[nm])
        tot_err += wer * nw; tot_words += nw
        rows.append(f"""
        <div class="spk">
          <div class="spkhead"><b>{nm}</b><span class="wer">WER {wer*100:.0f}%</span></div>
          <div class="line"><span class="lbl">正解</span><div class="txt">{th or '<i>―</i>'}</div></div>
          <div class="line"><span class="lbl">システム</span><div class="txt">{sh or '<i>(出力なし＝取りこぼし)</i>'}</div></div>
        </div>""")
    overall = tot_err / max(1, tot_words)

    html = TEMPLATE.format(
        model=args.model, leak=args.leak_db, overall=f"{overall*100:.0f}",
        rows="".join(rows))
    with open(args.out, "w") as f:
        f.write(html)
    print("wrote", os.path.abspath(args.out), f"overall WER={overall*100:.0f}%")


TEMPLATE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>精度レポート 正解 vs システム</title><style>
 body{{font-family:-apple-system,"Hiragino Sans",sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#222;line-height:1.7}}
 h1{{font-size:20px}}
 .desc{{background:#f5f7fa;border-radius:8px;padding:12px 16px;font-size:13px}}
 .miss{{color:#ff3b30;text-decoration:line-through}}
 .extra{{background:#ffe08a;border-radius:3px;padding:0 2px}}
 .overall{{font-size:15px;margin:14px 0}}
 .spk{{border:1px solid #e2e6ea;border-radius:10px;padding:10px 14px;margin:10px 0}}
 .spkhead{{display:flex;justify-content:space-between}} .wer{{color:#667;font-size:13px}}
 .line{{display:flex;gap:8px;margin:3px 0}}
 .lbl{{width:64px;flex:none;font-size:12px;color:#778;text-align:right}}
 .txt{{flex:1}}
 .key span{{font-size:12px;margin-right:14px}}
</style></head><body>
<h1>精度レポート（正解 vs システム）</h1>
<div class="desc">
<b>正解</b>＝各話者のクリーン音声(漏れ込みなし)をそのまま文字起こしした「本当に言った内容」。<br>
<b>システム</b>＝漏れ込み入りのマイクから、関門(ゲート)を通して得た書き起こし。<br>
<span class="key"><span class="miss">赤取り消し線</span>＝正解にあるのにシステムが落とした語(取りこぼし)。
<span class="extra">黄ハイライト</span>＝システムが余分に出した/誤った語。</span><br>
モデル=Whisper {model} / 漏れ込み {leak}dB。
</div>
<div class="overall">全体の単語誤り率(WER): <b>{overall}%</b>　<span style="font-size:12px;color:#778">（0%が完璧。低いほど良い）</span></div>
{rows}
</body></html>"""


if __name__ == "__main__":
    main()
