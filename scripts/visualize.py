"""実験の中身を「聴いて・見て」分かるHTMLレポートを生成する.

出力 report.html には各マイクごとに:
  - そのマイクの音声プレイヤー(漏れ込みが実際に聴ける)
  - 「正解(誰がいつ話したか)」と「システム判定(本人と認識した区間)」の時間軸バー
  - 正解=緑 / 正解一致=青 / 誤検出=赤 / 取りこぼし=緑の枠だけ
が並ぶ。ブラウザでダブルクリックして開くだけ。
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import os

import numpy as np
import soundfile as sf

from spkattr.synth import SynthConfig, synthesize
from spkattr.pipeline import run
from spkattr.evaluate import labels_to_windows

SR = 16000
WIN_MS, HOP_MS = 500.0, 250.0


def wav_b64(x: np.ndarray, sr: int = SR) -> str:
    x = x / (np.abs(x).max() + 1e-9) * 0.9
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def segments_to_pred(segments, n_win, n_ch):
    hop = int(SR * HOP_MS / 1000)
    pred = np.zeros((n_win, n_ch), dtype=bool)
    for seg in segments:
        w = int(round(seg.start * SR / hop))
        if 0 <= w < n_win and seg.is_own_voice:
            pred[w, seg.channel] = True
    return pred


def bar_html(truth_row, pred_row, total_s):
    """1チャンネル分の時間軸バーをHTMLで作る."""
    n = len(truth_row)
    cells = []
    for w in range(n):
        left = 100 * w / n
        width = 100 / n
        t, p = truth_row[w], pred_row[w]
        if p and t:
            cls = "ok"      # 正解一致(青)
        elif p and not t:
            cls = "fp"      # 誤検出(赤)
        elif t and not p:
            cls = "miss"    # 取りこぼし(緑枠)
        else:
            cls = "none"
        cells.append(f'<span class="cell {cls}" style="left:{left}%;width:{width}%"></span>')
    return "".join(cells)


COLORS = ["#34c759", "#ff9500", "#af52de", "#5ac8fa", "#ff2d55", "#ffcc00"]


def gridlines(total_s):
    """1秒ごとの薄い縦線を返す(全トラック共通)."""
    out = []
    for s in range(1, int(np.ceil(total_s))):
        out.append(f'<span class="grid" style="left:{100*s/total_s}%"></span>')
    return "".join(out)


def ruler(total_s):
    """秒の目盛り(0,1,2,...)を返す."""
    ticks = []
    for s in range(0, int(np.ceil(total_s)) + 1):
        ticks.append(f'<span class="tick" style="left:{100*s/total_s}%">{s}</span>')
    return "".join(ticks)


def build(sources, names, leak_db, use_embedding, overlap_s):
    cfg = SynthConfig(sr=SR, leak_db=leak_db)
    channels, labels = synthesize(sources, cfg)
    n_ch, n = channels.shape

    enroll = None
    if use_embedding:
        from spkattr.embedding import Enrollment
        enroll = Enrollment.create("cpu")
        for nm, src in zip(names, sources):
            enroll.register(nm, src)

    segments = run(channels, SR, names, enroll=enroll, win_ms=WIN_MS, hop_ms=HOP_MS)
    truth_w = labels_to_windows(labels, SR, WIN_MS, HOP_MS)
    n_win = len(truth_w)
    pred_w = segments_to_pred(segments, n_win, n_ch)
    truth_w = truth_w[:n_win]

    tp = int((pred_w & truth_w).sum()); fp = int((pred_w & ~truth_w).sum())
    fn = int((~pred_w & truth_w).sum()); tn = int((~pred_w & ~truth_w).sum())
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    ctr = tn / (tn + fp + 1e-9)

    total_s = n / SR
    grid = gridlines(total_s)

    # 会話全体図: 誰がいつ話すか(正解)を1枚に重ねて表示
    over_rows = []
    for ch in range(n_ch):
        col = COLORS[ch % len(COLORS)]
        bar = truth_bar(truth_w[:, ch], color=col)
        over_rows.append(
            f'<div class="lane"><span class="lbl" style="color:{col}">{names[ch]}</span>'
            f'<div class="track">{grid}{bar}</div></div>')
    overview = "".join(over_rows)

    # マイクごとの詳細
    rows = []
    for ch in range(n_ch):
        audio = wav_b64(channels[ch])
        clean = wav_b64(sources[ch])
        col = COLORS[ch % len(COLORS)]
        rows.append(f"""
        <div class="ch">
          <div class="chhead">
            <b style="color:{col}">マイク {ch+1}（持ち主: {names[ch]}）</b>
            <span class="players">
              <label>このマイク(漏れ込みあり)</label>
              <audio controls preload="none" src="data:audio/wav;base64,{audio}"></audio>
              <label>本人クリーン(参考)</label>
              <audio controls preload="none" src="data:audio/wav;base64,{clean}"></audio>
            </span>
          </div>
          <div class="lane"><span class="lbl">正解</span>
            <div class="track">{grid}{truth_bar(truth_w[:, ch])}</div></div>
          <div class="lane"><span class="lbl">判定</span>
            <div class="track">{grid}{bar_html(truth_w[:, ch], pred_w[:, ch], total_s)}</div></div>
        </div>""")

    html = TEMPLATE.format(
        n_spk=n_ch, leak_db=leak_db, total_s=f"{total_s:.1f}",
        overlap_s=f"{overlap_s:.0f}", win_ms=int(WIN_MS), hop_ms=int(HOP_MS),
        emb="あり(声紋照合)" if use_embedding else "なし(相関のみ)",
        prec=f"{prec:.2f}", rec=f"{rec:.2f}", ctr=f"{ctr:.2f}",
        tp=tp, fp=fp, fn=fn, tn=tn, rows="".join(rows),
        ruler=ruler(total_s), overview=overview,
    )
    return html


def arrange_conversation(clips, overlap_s=2.0):
    """各話者のクリップを、一定の重なりを持たせて順番に配置する.

    話者 i の開始 = i * (clip_len - overlap)。
    返り値: 全話者が同じ長さ(タイムライン全長)に揃った音源リスト。
    隣り合う話者だけが overlap_s 秒だけ重なる = 現実の会話に近い。
    """
    overlap = int(SR * overlap_s)
    starts, cur = [], 0
    for c in clips:
        starts.append(cur)
        cur += max(1, len(c) - overlap)
    total = max(s + len(c) for s, c in zip(starts, clips))
    out = []
    for s, c in zip(starts, clips):
        buf = np.zeros(total, dtype=np.float32)
        buf[s:s + len(c)] = c
        out.append(buf)
    return out


def truth_bar(truth_col, color=None):
    n = len(truth_col)
    cells = []
    style_color = f";background:{color}" if color else ""
    for w in range(n):
        if truth_col[w]:
            left = 100 * w / n
            cells.append(f'<span class="cell truth" style="left:{left}%;width:{100/n}%{style_color}"></span>')
    return "".join(cells)


TEMPLATE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>話者帰属 実験レポート</title>
<style>
 body{{font-family:-apple-system,"Hiragino Sans",sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#222;line-height:1.6}}
 h1{{font-size:20px}} h2{{font-size:15px;margin:26px 0 6px}}
 .desc{{background:#f5f7fa;border-radius:8px;padding:14px 18px;font-size:13.5px}}
 .desc code{{background:#e9edf2;padding:1px 5px;border-radius:4px;font-size:12px}}
 .metrics{{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}}
 .metric{{background:#fff;border:1px solid #e2e6ea;border-radius:8px;padding:10px 16px;min-width:130px}}
 .metric b{{display:block;font-size:22px}} .metric span{{font-size:12px;color:#667}}
 .ch{{border:1px solid #e2e6ea;border-radius:10px;padding:12px 14px;margin:12px 0}}
 .chhead{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}}
 .players label{{font-size:11px;color:#667;margin:0 4px 0 10px}}
 audio{{height:30px;vertical-align:middle}}
 .lane{{display:flex;align-items:center;margin:5px 0}}
 .lbl{{width:46px;font-size:12px;color:#556;text-align:right;padding-right:8px;flex:none}}
 .track{{position:relative;flex:1;height:20px;background:#f0f2f5;border-radius:4px;overflow:hidden}}
 .cell{{position:absolute;top:0;height:100%}}
 .grid{{position:absolute;top:0;height:100%;width:1px;background:#d3d8de}}
 .truth{{background:#34c759}}
 .ok{{background:#0a84ff}} .fp{{background:#ff3b30}}
 .miss{{background:transparent;border:2px solid #ff9f0a;box-sizing:border-box}}
 .rulerwrap{{display:flex;align-items:flex-end;margin:2px 0 2px}}
 .ruler{{position:relative;flex:1;height:16px;font-size:10px;color:#889}}
 .tick{{position:absolute;transform:translateX(-50%);bottom:0}}
 .tick:after{{content:"";display:block;width:1px;height:5px;background:#bcc2c9;margin:1px auto 0}}
 .legend span{{display:inline-block;margin-right:14px;font-size:12px}}
 .sw{{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:middle;margin-right:4px}}
 .axisnote{{font-size:11px;color:#889;text-align:right;margin:2px 0 0}}
</style></head><body>
<h1>話者帰属 実験レポート</h1>
<div class="desc">
<b>① どんなデータか（合成の作り方）</b><br>
{n_spk}人がそれぞれ専用マイクを持つ会話を、公開音声から合成しています。手順:<br>
1. 各話者の音声クリップを、会話のように時間をずらして1本のタイムラインに並べる
（隣どうしが <code>{overlap_s}秒</code> だけ重なる）。<br>
2. マイク k の信号 ＝ <b>本人のクリーン音声</b> ＋ 他の全員の声を <code>{leak_db}dB</code>
減衰・わずかに遅延させた<b>漏れ込み</b>の合計。<br>
→ 下の「このマイク(漏れ込みあり)」で実際に聴けます。<br><br>
<b>② 何を正解とするか</b><br>
各話者の<b>クリーン音声</b>そのものから「声が出ている区間」をエネルギーで判定し、
それを正解(緑)としています。誰がいつ話すかは合成時に分かっているので確実です
（※区間の細かい境界はエネルギー閾値依存で、ここは今後実データ/Sileroで固める予定）。<br><br>
<b>③ システムは何をするか</b><br>
VAD→チャンネル間相関→声紋照合({emb})で、各マイクについて「今鳴っているのは本当に持ち主か」を
<code>{win_ms}ms</code>窓・<code>{hop_ms}ms</code>間隔で判定します。<br>
<b>正解バー</b>＝実際に話した区間。<b>判定バー</b>＝システムが本人と認識した区間。
</div>

<div class="metrics">
 <div class="metric"><b>{prec}</b><span>誤帰属のなさ (precision)</span></div>
 <div class="metric"><b>{rec}</b><span>本人発話の拾い (recall)</span></div>
 <div class="metric"><b>{ctr}</b><span>漏れ込み棄却率</span></div>
 <div class="metric"><b>{total_s}s</b><span>tp={tp} fp={fp} fn={fn} tn={tn}</span></div>
</div>
<div class="legend">
 <span><i class="sw" style="background:#34c759"></i>正解(実際に発話)</span>
 <span><i class="sw" style="background:#0a84ff"></i>正しく本人と判定</span>
 <span><i class="sw" style="background:#ff3b30"></i>誤検出(漏れ込みを本人扱い)</span>
 <span><i class="sw" style="border:2px solid #ff9f0a"></i>取りこぼし</span>
</div>

<h2>会話全体図（誰がいつ話すか＝正解の台本）</h2>
<div class="rulerwrap"><span class="lbl"></span><div class="ruler">{ruler}</div></div>
{overview}
<p class="axisnote">横軸＝時間（秒）。縦の薄い線が1秒ごと。</p>

<h2>マイクごとの詳細（正解 vs システム判定）</h2>
<div class="rulerwrap"><span class="lbl"></span><div class="ruler">{ruler}</div></div>
{rows}
<p style="font-size:12px;color:#778;margin-top:20px">
 横軸は全パネル共通で時間(秒)。<code>--leak-db</code> を小さく(例 -6)すると漏れ込みが増えて難しく、
 <code>--overlap-s</code> を大きくすると重なりが増えます。recall の落ち方が、重複下のロバスト化(次の課題)の伸びしろです。
</p>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default=os.path.join(os.path.dirname(__file__), "..", "samples"))
    ap.add_argument("--leak-db", type=float, default=-12.0)
    ap.add_argument("--overlap-s", type=float, default=2.0,
                    help="隣り合う話者が重なる秒数(大きいほど難しい)")
    ap.add_argument("--no-embedding", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "report.html"))
    args = ap.parse_args()

    import librosa
    files = sorted(glob.glob(os.path.join(args.samples_dir, "*.wav")))
    assert files, f"no wav in {args.samples_dir}"
    clips, names = [], []
    for p in files:
        y, _ = librosa.load(p, sr=SR)
        clips.append(y); names.append(os.path.basename(p).split(".")[0])

    # 会話のように発話時刻をずらして配置(一部だけ重なる).
    # 各話者を共通のタイムライン上の別オフセットに置く。
    sources = arrange_conversation(clips, overlap_s=args.overlap_s)

    html = build(sources, names, args.leak_db, not args.no_embedding, args.overlap_s)
    with open(args.out, "w") as f:
        f.write(html)
    print("wrote", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
