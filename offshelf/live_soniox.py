"""1マイク + Soniox リアルタイム議事録ツール（日本語・話者分離内蔵）.

1本のマイクで「誰が・何を」をライブ取得する本線ツール。
Sonioxのストリーミング(WebSocket)に音声を流し、speaker付きトークンが返るので、
多マイク・ゲート・同期なしで who-said-what が出る。

機能:
  - 話者ごとに色分けしたライブ表示（確定前のテキストは薄く表示）
  - Markdown議事録 + HTML を transcripts/ に自動保存（発話確定ごと＝クラッシュ安全）
  - HTMLはブラウザ自動オープン、ライブ中2秒ごと自動更新（--no-openで無効）
  - 声紋プロファイル方式の話者特定（登録不要で自動補正）。判定は2経路のみ:
      ① 即時判定: 声紋が強一致(0.75)した発話はその場で人物確定（入れ替わりも補正）
      ② それ以外は3発話バッファ: 一貫した3発話を束ねて「既存人物に合流 or 新規人物N」
      不変条件: 一度確定した人物キーは書き換えない（遡及置換は 話者N→人物N の昇格のみ）。
      「1=松井」で実名化、実名のみ voices.json に永続化 → 次回から自動で実名表示。
      検証: AMI実音声シミュレーションで素ラベル66%→簡素化版81%（回帰時の害1.9%）。
  - 「fix 3=1」で誤ラベルの表示を統合（過去の発言も修正）

準備(Mac):
  uv add websockets sounddevice
  export SONIOX_API_KEY=...   # https://console.soniox.com で取得

使い方:
  uv run python offshelf/live_soniox.py            # 実マイクでライブ
  uv run python offshelf/live_soniox.py --wav offshelf/ami_raw/mic0.wav  # ファイル擬似ライブ
  実行中: 「1=松井」Enter で話者登録 / Ctrl+C で終了（保存先を表示）
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import queue
import re
import sys
import threading
import time

import numpy as np

SR = 16000
WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"

RESET = "\x1b[0m"
DIM = "\x1b[2m"
CLEAR_LINE = "\r\x1b[K"
PALETTE = ["\x1b[36m", "\x1b[33m", "\x1b[35m", "\x1b[32m", "\x1b[34m", "\x1b[91m"]
HTML_PALETTE = ["#0e7490", "#a16207", "#7e22ce", "#15803d", "#1d4ed8", "#dc2626"]

HTML_TMPL = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
{refresh}<title>議事録 {title}</title>
<style>
body {{ font-family: -apple-system, "Hiragino Sans", sans-serif; max-width: 760px;
       margin: 2rem auto; padding: 0 1rem; background: #fafafa; color: #1f2937; }}
h1 {{ font-size: 1.2rem; }} .meta {{ color: #6b7280; font-size: .85rem; }}
.u {{ margin: .5rem 0; padding: .55rem .8rem; background: #fff; border-radius: 10px;
     border: 1px solid #e5e7eb; }}
.u .who {{ font-weight: 700; margin-right: .5em; }}
.u .ts {{ color: #9ca3af; font-size: .8rem; margin-right: .6em; font-variant-numeric: tabular-nums; }}
.live {{ color: #16a34a; font-size: .8rem; }}
</style></head><body>
<h1>議事録 {title}</h1>
<p class="meta">{status} / 話者: {speakers}</p>
{body}
<script>window.scrollTo(0, document.body.scrollHeight);</script>
</body></html>
"""


def fmt_ts(ms: int | None) -> str:
    if ms is None:
        return "--:--"
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


class VoiceProfiles:
    """登録済み声紋プロファイルとの照合で話者名を決める（台帳固定・誤り非伝播）.

    設計方針: 「台帳をその場で学習する」オンラインクラスタリングは、台帳自体が
    間違えるため補正ルール（自動統合・脱出・汚染防止…）が増殖した。本方式は
    台帳=プロファイルを固定し、各発話を独立に判定する。

    - 発話(min_sec以上) → 声紋 → 登録者の中で一番似ている人（floor未満なら未知）
    - 短い相槌 → そのSonioxラベルの直近の判定に追従
    - 未登録の声 → "#<Sonioxラベル>" キー（話者1/2…と表示）
    - enroll() でラベルの直近声紋をプロファイル登録・voices.jsonに永続化
      → 次回の会議からは発話だけで自動的に実名表示
    """

    ANON = re.compile(r"^人物\d+$")

    def __init__(self, path: str = "voices.json", thresh: float = 0.75, min_sec: float = 1.0,
                 margin: float = 0.05, auto: bool = True, consist: float = 0.62,
                 dedupe: float = 0.70):
        from resemblyzer import VoiceEncoder, preprocess_wav  # 初回ロード数秒
        self._pre = preprocess_wav
        self._enc = VoiceEncoder("cpu", verbose=False)
        self.path = path
        self.thresh = thresh   # 即時判定: 単発声紋がこれ以上強く一致した時だけその場で人物確定
        self.margin = margin   # 即時判定の追加条件: 2位との差（似た声の誤マッチ防止）
        self.auto = auto       # 未知の声の自動登録（匿名「人物N」プロファイル）
        self.consist = consist  # バッファ確定条件: 3発話の全ペア類似がこれ以上
        self.dedupe = dedupe    # 3発話プロファイルが既存人物に合流するしきい値
        self.min_sec = min_sec
        self.profiles: dict[str, np.ndarray] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.profiles = {k: np.asarray(v, dtype=np.float64)
                                 for k, v in json.load(f).items()}
        self.sp_map: dict[str, str] = {}                    # Sonioxラベル -> 表示キー
        self.label_embs: dict[str, list[np.ndarray]] = {}   # ラベル -> 直近声紋（手動登録用）
        self.buf: dict[str, list[np.ndarray]] = {}          # ラベル -> 即時判定できなかった声紋
        self.n_anon = 0
        self.embed_ms: list[float] = []                     # レイテンシ統計
        self.counts: dict[str, int] = {}                    # 判定種別の集計
        self.last: dict | None = None                       # 直近の判定内容（可視化用）

    def _note(self, kind: str, **info) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.last = {"kind": kind, **info}

    def _embed(self, wav: np.ndarray) -> np.ndarray | None:
        t0 = time.perf_counter()
        try:
            w = self._pre(wav, source_sr=SR)
            if w.size < SR * 0.6:
                return None
            emb = self._enc.embed_utterance(w)
        except Exception:
            return None
        self.embed_ms.append((time.perf_counter() - t0) * 1000)
        return emb / np.linalg.norm(emb)

    def classify(self, wav: np.ndarray, sp, overlapped: bool = False) -> str:
        """overlapped: この発話が他の発話と時間的に重なっていたか.

        重なった発話は声が混ざっていて声紋がデタラメになるため、声での判定を
        スキップして直前の対応を維持する（誤マッチで対応表を汚さない）。
        """
        """判定は2経路だけ:
        ① 即時判定 — 単発声紋が強く一致(thresh以上＋2位とmargin差)した時だけ、その場で人物確定。
        ② それ以外（弱い一致・未知・僅差）は区別せず3発話バッファへ — 一貫した3発話を束ねた
           プロファイルで「既存人物に合流(dedupe以上) or 新規人物N」を確定。
        不変条件: 過去に確定した人物キーは書き換えない。遡及置換は「#ラベル → 人物」の昇格のみ。
        """
        sp = str(sp)
        prev = self.sp_map.get(sp)
        kind, info = "相槌追従", {}
        if overlapped and wav.size >= SR * self.min_sec:
            kind = "重なりスキップ"
        elif wav.size >= SR * self.min_sec:
            emb = self._embed(wav)
            if emb is None:
                kind = "声紋計算不可"
            else:
                self.label_embs.setdefault(sp, []).append(emb)
                del self.label_embs[sp][:-10]    # 手動登録用に直近10発話だけ保持
                if self.profiles:
                    ranked = sorted(((float(np.dot(p, emb)), n)
                                     for n, p in self.profiles.items()), reverse=True)
                    sim, cand = ranked[0]
                    second = ranked[1][0] if len(ranked) > 1 else -1.0
                    info = {"sim": sim, "name": cand, "prev": prev}
                    if sim >= self.thresh and sim - second >= self.margin:
                        self.sp_map[sp] = cand
                        self.buf.pop(sp, None)
                        self._note("補正" if (prev is not None and not prev.startswith("#")
                                              and prev != cand) else "声紋一致", label=sp, **info)
                        return cand
                kind = "蓄積中"
                if self.auto:
                    b = self.buf.setdefault(sp, [])
                    b.append(emb)
                    del b[:-3]
                    if len(b) == 3 and all(float(np.dot(b[i], b[j])) >= self.consist
                                           for i, j in itertools.combinations(range(3), 2)):
                        prof = np.mean(b, axis=0)
                        prof = prof / np.linalg.norm(prof)
                        hit_sim, hit = max(((float(np.dot(p, prof)), n)
                                            for n, p in self.profiles.items()), default=(-1.0, None))
                        if hit is not None and hit_sim >= self.dedupe:
                            target = hit          # 既存人物の別ラベルだった → 合流
                        else:
                            self.n_anon += 1
                            target = f"人物{self.n_anon}"
                            self.profiles[target] = prof   # 新規人物（以後凍結）
                        self.buf.pop(sp, None)
                        # 遡及置換は未確定キー(#ラベル)の昇格のみ。人物キーは絶対に書き換えない。
                        rename = ("#" + sp, target) if (prev is None or prev.startswith("#")) else None
                        self.sp_map[sp] = target
                        self._note("自動登録", label=sp, name=target, rename=rename)
                        return target
        # 声紋で決められない（重なり/短い相槌/蓄積中）→ ラベルの直近判定に追従
        key = prev if prev is not None else "#" + sp
        self.sp_map[sp] = key
        self._note(kind, label=sp, **info)
        return key

    def enroll(self, label: str, name: str) -> str | None:
        """「1=松井」「人物2=田中」: 話者に名前を付ける（声の登録 or 既存人物のリネーム）.

        実名を付けたプロファイルのみ voices.json に永続化される（匿名「人物N」は
        そのセッション限り）。戻り値: 旧表示キー（過去のrecords付け替え用）。
        十分な音声がまだ無ければ None。
        """
        label = str(label)
        if label in self.profiles:
            # 「人物1=松井」: 既存プロファイルのリネーム
            self.profiles[name] = self.profiles.pop(label)
            old = label
        else:
            cur = self.sp_map.get(label)
            if cur is not None and cur in self.profiles:
                # ラベルが（自動登録済みの）人物に対応済み → その人物に命名
                self.profiles[name] = self.profiles.pop(cur)
                old = cur
            else:
                # ラベルの直近声紋から新規登録
                embs = self.label_embs.get(label)
                if not embs:
                    return None
                prof = np.mean(embs, axis=0)
                self.profiles[name] = prof / np.linalg.norm(prof)
                old = cur if cur is not None else "#" + label
        for k, v in list(self.sp_map.items()):
            if v == old:
                self.sp_map[k] = name
        if old != label:   # 「人物N=名前」のリネーム以外は、ラベル自体も対応づける
            self.sp_map[label] = name
        self._persist()
        return old

    def _persist(self):
        named = {k: v.tolist() for k, v in self.profiles.items() if not self.ANON.match(k)}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(named, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def stats(self) -> str:
        parts = []
        if self.embed_ms:
            a = np.array(self.embed_ms)
            parts.append(f"声紋計算 {len(a)}回 平均{a.mean():.0f}ms 最大{a.max():.0f}ms")
        if self.counts:
            order = ["声紋一致", "補正", "自動登録", "蓄積中", "相槌追従",
                     "重なりスキップ", "声紋計算不可"]
            parts.append("判定内訳: " + " / ".join(
                f"{k}{self.counts[k]}" for k in order if self.counts.get(k)))
        return "、".join(parts) or "判定なし"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--model", default="stt-rt-v4")
    ap.add_argument("--wav", default=None, help="指定で実マイクの代わりにファイル擬似ライブ")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="保存先mdファイル（省略時 transcripts/日時.md）")
    ap.add_argument("--no-open", action="store_true", help="ブラウザを自動で開かない")
    ap.add_argument("--no-vp", action="store_true", help="声紋照合を無効化（Sonioxのラベルをそのまま使う）")
    ap.add_argument("--voices", default="voices.json", help="声紋プロファイルの保存先(既定 voices.json)")
    ap.add_argument("--vp-match", type=float, default=0.75,
                    help="即時判定のしきい値。別人が同一人物にされる→上げる/判定が遅い→下げる(既定0.75)")
    ap.add_argument("--vp-no-auto", action="store_true",
                    help="未知の声の自動登録（匿名「人物N」）を無効化")
    ap.add_argument("--vp-debug", action="store_true", help="発話ごとの声紋判定の内訳を表示")
    args = ap.parse_args()

    api_key = os.environ.get("SONIOX_API_KEY")
    if not api_key:
        raise SystemExit("環境変数 SONIOX_API_KEY を設定してください（https://console.soniox.com）")

    try:
        from websockets.sync.client import connect
    except ImportError:
        raise SystemExit("uv add websockets を実行してください")

    config = {
        "api_key": api_key,
        "model": args.model,
        "language_hints": [args.lang],
        "enable_speaker_diarization": True,
        "enable_endpoint_detection": True,
        "audio_format": "pcm_s16le",
        "sample_rate": SR,
        "num_channels": 1,
    }

    started = datetime.datetime.now()
    if args.out:
        out_path = args.out
    else:
        os.makedirs("transcripts", exist_ok=True)
        out_path = os.path.join("transcripts", started.strftime("%Y-%m-%d_%H%M") + ".md")
    html_path = os.path.splitext(out_path)[0] + ".html"

    # --- 状態 ---
    names: dict[str, str] = {}          # 表示キー -> 別名（声紋OFF時の命名用）
    colors: dict[str, str] = {}         # 表示キー -> ANSI色（出現順に割当）
    records: list[dict] = []            # 確定発話 {"ms", "speaker", "text"}
    state_lock = threading.Lock()

    tracker: VoiceProfiles | None = None
    if not args.no_vp:
        try:
            print("# 声紋モデルを読み込み中…", flush=True)
            tracker = VoiceProfiles(path=args.voices, thresh=args.vp_match,
                                    auto=not args.vp_no_auto)
            if tracker.profiles:
                print(f"# 声紋プロファイル: {', '.join(tracker.profiles)}（{args.voices}）", flush=True)
            else:
                print(f"# 声紋プロファイル: なし。未知の声は「人物N」として自動追跡、"
                      f"「1=松井」で実名化すると次回から自動表示（{args.voices}）", flush=True)
        except ImportError:
            print("# 声紋照合: OFF（uv add resemblyzer で有効化できます）", flush=True)

    pcm_buf = bytearray()               # 送信済み音声の全バッファ（声紋切り出し用, 16bit）
    buf_lock = threading.Lock()

    def disp_name(key) -> str:
        key = str(key)
        if key in names:
            return names[key]
        return f"話者{key[1:]}" if key.startswith("#") else key

    def key_for_label(sp) -> str:
        sp = str(sp)
        if tracker is not None and sp in tracker.sp_map:
            return tracker.sp_map[sp]
        return "#" + sp

    def color_of(key) -> str:
        key = str(key)
        if key not in colors:
            colors[key] = PALETTE[len(colors) % len(PALETTE)]
        return colors[key]

    def write_md():
        with state_lock:
            lines = [
                f"# 議事録 {started.strftime('%Y-%m-%d %H:%M')}",
                "",
                f"話者: " + (", ".join(f"{disp_name(s)}" for s in sorted(colors)) or "（未検出）"),
                "",
            ]
            for r in records:
                lines.append(f"- **[{fmt_ts(r['ms'])}] {disp_name(r['speaker'])}**: {r['text']}")
            tmp = out_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(tmp, out_path)

    def write_html(live: bool = True):
        import html as _html
        with state_lock:
            parts = []
            for r in records:
                sp = str(r["speaker"])
                idx = list(colors).index(sp) if sp in colors else 0
                c = HTML_PALETTE[idx % len(HTML_PALETTE)]
                parts.append(
                    f'<div class="u"><span class="ts">{fmt_ts(r["ms"])}</span>'
                    f'<span class="who" style="color:{c}">{_html.escape(disp_name(sp))}</span>'
                    f'{_html.escape(r["text"])}</div>'
                )
            doc = HTML_TMPL.format(
                refresh='<meta http-equiv="refresh" content="2">' if live else "",
                title=started.strftime("%Y-%m-%d %H:%M"),
                status='<span class="live">● ライブ（2秒ごと自動更新）</span>' if live else "終了",
                speakers=_html.escape(", ".join(disp_name(s) for s in sorted(colors)) or "（未検出）"),
                body="\n".join(parts) or '<p class="meta">（まだ発話なし）</p>',
            )
            tmp = html_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(doc)
            os.replace(tmp, html_path)

    def save(live: bool = True):
        write_md()
        write_html(live)

    def print_line(text: str):
        sys.stdout.write(CLEAR_LINE + text + "\n")
        sys.stdout.flush()

    def show_partial(sp, text: str):
        if not text.strip():
            sys.stdout.write(CLEAR_LINE)
        else:
            cols = os.get_terminal_size().columns if sys.stdout.isatty() else 120
            line = f"{disp_name(key_for_label(sp))}: {text.strip()}"
            sys.stdout.write(CLEAR_LINE + DIM + line[-(cols - 2):] + RESET)
        sys.stdout.flush()

    # --- 音声入力（マイク or ファイル）→ audio_q ---
    audio_q: "queue.Queue[bytes | None]" = queue.Queue()
    stop = threading.Event()

    def from_mic():
        import sounddevice as sd

        def cb(indata, frames, t, status):
            pcm = (np.clip(indata[:, 0], -1, 1) * 32767).astype("<i2").tobytes()
            audio_q.put(pcm)
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            device=args.device, callback=cb, blocksize=int(SR * 0.1)):
            while not stop.is_set():
                time.sleep(0.1)
        audio_q.put(None)

    def from_wav():
        import librosa
        y, _ = librosa.load(args.wav, sr=SR)
        step = int(SR * 0.12)
        for i in range(0, len(y), step):
            if stop.is_set():
                break
            pcm = (np.clip(y[i:i + step], -1, 1) * 32767).astype("<i2").tobytes()
            audio_q.put(pcm)
            time.sleep(0.12)
        audio_q.put(None)

    # --- 実行中コマンド ---
    def key_of(tok: str) -> str:
        """コマンド引数を表示キーへ: 登録名ならそのまま、それ以外は未登録ラベル扱い."""
        if tracker is not None and tok in tracker.profiles:
            return tok
        return "#" + tok

    def stdin_commands():
        while not stop.is_set():
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            mfix = re.match(r"^\s*fix\s+(\S+)\s*=\s*(\S+)\s*$", line)
            m = re.match(r"^\s*(\S+?)\s*=\s*(.+?)\s*$", line)
            if mfix:
                src, dst = key_of(mfix.group(1)), key_of(mfix.group(2))
                with state_lock:
                    for r in records:
                        if r["speaker"] == src:
                            r["speaker"] = dst
                    if tracker is not None:
                        for k, v in list(tracker.sp_map.items()):
                            if v == src:
                                tracker.sp_map[k] = dst
                save()
                print_line(f"# {disp_name(src)} を {disp_name(dst)} に統合しました（過去の発言も修正済み）")
            elif m:
                label, name = m.group(1), m.group(2)
                if tracker is not None:
                    old = tracker.enroll(label, name)
                    if old is None:
                        print_line(f"# 話者{label}の音声がまだ足りません（1秒以上話してから再実行）")
                        continue
                    with state_lock:
                        for r in records:
                            if r["speaker"] == old:
                                r["speaker"] = name
                    save()
                    print_line(f"# {name} の声を登録しました（過去の発言も置換、次回の会議から自動表示）")
                else:
                    with state_lock:
                        names["#" + label] = name
                    save()
                    print_line(f"# 話者{label} → {name}（過去の発言も置換済み）")
            elif line.strip():
                print_line("# コマンド: 「1=松井」(声を登録) / 「fix 3=1」(表示の統合) / Ctrl+Cで終了")

    print("# Sonioxに接続中…", flush=True)
    with connect(WS_URL) as ws:
        ws.send(json.dumps(config))
        threading.Thread(target=from_wav if args.wav else from_mic, daemon=True).start()
        threading.Thread(target=stdin_commands, daemon=True).start()

        def sender():
            while True:
                pcm = audio_q.get()
                if pcm is None:
                    ws.send("")   # 終端
                    break
                with buf_lock:
                    pcm_buf.extend(pcm)   # Sonioxの時刻軸と完全一致する位置で蓄積
                ws.send(pcm)
        threading.Thread(target=sender, daemon=True).start()

        save()
        print("# 開始。話してください（「1=松井」で声を登録 / Ctrl+Cで終了）", flush=True)
        print(f"# 保存先: {out_path}", flush=True)
        print(f"# ブラウザ表示: open {html_path}（ライブ中は2秒ごと自動更新）\n", flush=True)
        if not args.no_open:
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(html_path))

        cur_speaker = None
        cur_text = ""
        cur_ms: int | None = None
        cur_end: int | None = None
        recent_segs: list[tuple] = []   # (start, end, ラベル) 直近の確定発話（重なり検出用）

        def overlaps_other(start, end, label) -> bool:
            if start is None or end is None:
                return False
            return any(l != label and min(e, end) - max(s, start) > 0
                       for s, e, l in recent_segs)

        def flush():
            nonlocal cur_text, cur_ms, cur_end
            if cur_text.strip():
                label = str(cur_speaker)
                if tracker is not None:
                    if cur_ms is not None and cur_end is not None and cur_end > cur_ms:
                        with buf_lock:
                            seg = bytes(pcm_buf[cur_ms * 32: cur_end * 32])  # 16サンプル/ms×2byte
                        wav = np.frombuffer(seg, dtype="<i2").astype(np.float32) / 32768.0
                    else:
                        wav = np.zeros(0, dtype=np.float32)
                    sp_id = tracker.classify(wav, cur_speaker,
                                             overlapped=overlaps_other(cur_ms, cur_end, label))
                    d = tracker.last
                    if d and d["kind"] == "補正":
                        print_line(f"# 補正: ラベル{d['label']}の発言を{d['name']}に修正"
                                   f"（類似{d['sim']:.2f}、放置なら{disp_name(d['prev'])}になっていた）")
                    elif d and d["kind"] == "自動登録":
                        if d["rename"]:   # 「#ラベル→人物」の昇格のみ遡及置換（人物キーは不変）
                            old_key, new_key = d["rename"]
                            with state_lock:
                                for r in records:
                                    if r["speaker"] == old_key:
                                        r["speaker"] = new_key
                        print_line(f"# この声を「{d['name']}」として追跡します"
                                   f"（実名にするには {d['label']}=名前 と入力）")
                    elif args.vp_debug and d:
                        extra = f" 類似{d['sim']:.2f}({d['name']})" if "sim" in d else ""
                        print_line(f"# vp判定[{d['kind']}]{extra}")
                else:
                    sp_id = "#" + str(cur_speaker)
                if cur_ms is not None and cur_end is not None:
                    recent_segs.append((cur_ms, cur_end, label))
                    del recent_segs[:-12]
                with state_lock:
                    records.append({"ms": cur_ms, "speaker": sp_id, "text": cur_text.strip()})
                c = color_of(sp_id)
                print_line(f"{c}[{fmt_ts(cur_ms)}] {disp_name(sp_id)}{RESET}: {cur_text.strip()}")
                save()
            cur_text = ""
            cur_ms = None
            cur_end = None

        try:
            while True:
                res = json.loads(ws.recv())
                if res.get("error_code") is not None:
                    print_line(f"# エラー: {res['error_code']} - {res.get('error_message')}")
                    break
                partial = ""
                partial_sp = cur_speaker
                for token in res.get("tokens", []):
                    text = token.get("text") or ""
                    if text == "<end>":
                        flush()
                        continue
                    if not text:
                        continue
                    if token.get("is_final"):
                        sp = token.get("speaker")
                        if sp != cur_speaker:
                            flush()
                            cur_speaker = sp
                        if cur_ms is None:
                            cur_ms = token.get("start_ms")
                        if token.get("end_ms") is not None:
                            cur_end = token["end_ms"]
                        cur_text += text
                    else:
                        partial += text
                        partial_sp = token.get("speaker") or partial_sp
                show_partial(partial_sp if partial else cur_speaker, cur_text + partial)
                if res.get("finished"):
                    flush()
                    print_line("# 終了")
                    break
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            flush()
            save(live=False)
            if tracker is not None:
                print_line(f"# レイテンシ統計: {tracker.stats()}")
            print_line(f"# 議事録を保存しました: {out_path} / {html_path}")


if __name__ == "__main__":
    main()
