"""1マイク + Soniox リアルタイム議事録ツール（日本語・話者分離内蔵）.

1本のマイクで「誰が・何を」をライブ取得する本線ツール。
Sonioxのストリーミング(WebSocket)に音声を流し、speaker付きトークンが返るので、
多マイク・ゲート・同期なしで who-said-what が出る。

機能:
  - 話者ごとに色分けしたライブ表示（確定前のテキストは薄く表示）
  - Markdown議事録 + HTML を transcripts/ に自動保存（発話確定ごと＝クラッシュ安全）
  - HTMLはブラウザ自動オープン、ライブ中2秒ごと自動更新（--no-openで無効）
  - 声紋プロファイル方式の話者特定:
      実行中に「1=松井」と打つと、話者1の声がプロファイルとして voices.json に登録され、
      以降（と次回以降の会議）は声の照合だけで自動的に「松井」と表示される。
      未登録の声は Soniox のラベル（話者1/2…）のまま表示。
      台帳が固定なので誤判定が伝播せず、ロジックは「一番似ている登録者を選ぶ」だけ。
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

    def __init__(self, path: str = "voices.json", floor: float = 0.60, min_sec: float = 1.0,
                 margin: float = 0.05):
        from resemblyzer import VoiceEncoder, preprocess_wav  # 初回ロード数秒
        self._pre = preprocess_wav
        self._enc = VoiceEncoder("cpu", verbose=False)
        self.path = path
        self.floor = floor
        self.margin = margin   # 1位と2位の類似度差がこれ未満なら判定保留（似た声の誤マッチ防止）
        self.min_sec = min_sec
        self.profiles: dict[str, np.ndarray] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.profiles = {k: np.asarray(v, dtype=np.float64)
                                 for k, v in json.load(f).items()}
        self.sp_map: dict[str, str] = {}                    # Sonioxラベル -> 表示キー
        self.label_embs: dict[str, list[np.ndarray]] = {}   # ラベル -> 直近声紋（登録用）
        self.embed_ms: list[float] = []                     # レイテンシ統計

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
        sp = str(sp)
        if not overlapped and wav.size >= SR * self.min_sec:
            emb = self._embed(wav)
            if emb is not None:
                self.label_embs.setdefault(sp, []).append(emb)
                del self.label_embs[sp][:-10]    # 登録用に直近10発話だけ保持
                if self.profiles:
                    ranked = sorted(((float(np.dot(p, emb)), n)
                                     for n, p in self.profiles.items()), reverse=True)
                    sim, name = ranked[0]
                    second = ranked[1][0] if len(ranked) > 1 else -1.0
                    if sim >= self.floor and sim - second >= self.margin:
                        self.sp_map[sp] = name
                        return name
        # 声紋で決められない（重なり/短い相槌/未知の声/僅差）→ ラベルの直近判定に追従
        key = self.sp_map.get(sp, "#" + sp)
        self.sp_map[sp] = key
        return key

    def enroll(self, label: str, name: str) -> str | None:
        """ラベルの直近声紋を name のプロファイルとして登録・永続化.

        戻り値: そのラベルの旧表示キー（呼び出し側が過去のrecordsを付け替える用）。
        十分な音声がまだ無ければ None。
        """
        embs = self.label_embs.get(str(label))
        if not embs:
            return None
        prof = np.mean(embs, axis=0)
        self.profiles[name] = prof / np.linalg.norm(prof)
        old = self.sp_map.get(str(label), "#" + str(label))
        self.sp_map[str(label)] = name
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: v.tolist() for k, v in self.profiles.items()}, f, ensure_ascii=False)
        os.replace(tmp, self.path)
        return old

    def stats(self) -> str:
        if not self.embed_ms:
            return "声紋計算なし"
        a = np.array(self.embed_ms)
        return f"声紋計算 {len(a)}回 平均{a.mean():.0f}ms 最大{a.max():.0f}ms"


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
    ap.add_argument("--vp-floor", type=float, default=0.60,
                    help="登録者と判定する最低類似度。未知の人が登録者扱いされる→上げる/逆→下げる(既定0.60)")
    ap.add_argument("--vp-margin", type=float, default=0.05,
                    help="1位と2位の類似度差がこれ未満なら判定保留(既定0.05)")
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
            tracker = VoiceProfiles(path=args.voices, floor=args.vp_floor, margin=args.vp_margin)
            if tracker.profiles:
                print(f"# 声紋プロファイル: {', '.join(tracker.profiles)}（{args.voices}）", flush=True)
            else:
                print(f"# 声紋プロファイル: なし。「1=松井」で登録すると次回から自動表示（{args.voices}）", flush=True)
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
