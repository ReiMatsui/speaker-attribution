"""ストリーミングAPI版ライブ文字起こし（Deepgram, 日本語）.

チャンク式ではなく本物のストリーミング。マイク音声をそのままWebSocketで送り、
低遅延(〜200ms)で確定テキストが返る。ラグのばらつき・無音幻聴が原理的に出にくい。
1人=1マイク前提のライブ体感用（複数iPhoneは後で同じAPIに各接続を割り当てる）。

準備(Mac):
  uv add deepgram-sdk sounddevice
  export DEEPGRAM_API_KEY=...   # https://deepgram.com で無料キー取得

使い方:
  uv run python offshelf/live_mic_deepgram.py            # 実マイクでライブ(日本語)
  uv run python offshelf/live_mic_deepgram.py --lang en  # 英語

注意: クラウドに音声を送ります（プライバシー要件があるならローカルの live_mic.py を使用）。
"""
from __future__ import annotations

import argparse
import os
import queue
import threading

import numpy as np

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--model", default="nova-2", help="nova-2 / nova-3 など")
    ap.add_argument("--name", default="あなた")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise SystemExit("環境変数 DEEPGRAM_API_KEY を設定してください（https://deepgram.com で無料取得）")

    try:
        from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
    except ImportError:
        raise SystemExit("uv add deepgram-sdk を実行してください")
    import sounddevice as sd

    dg = DeepgramClient(key)
    conn = dg.listen.websocket.v("1")

    def on_open(self, *a, **k):
        print("# 接続しました。話してください（Ctrl+Cで終了）\n", flush=True)

    def join_words(words):
        # 日本語は空白なし、それ以外は空白区切り
        parts = [(getattr(w, "punctuated_word", None) or getattr(w, "word", "")) for w in words]
        sep = "" if args.lang == "ja" else " "
        return sep.join(p for p in parts if p)

    def on_transcript(self, result, *a, **k):
        if not getattr(result, "is_final", False):
            return
        try:
            alt = result.channel.alternatives[0]
            words = list(alt.words or [])
        except Exception:
            return
        if not words:
            t = (alt.transcript or "").strip()
            if t:
                print(f"話者?: {t}", flush=True)
            return
        # 連続する同一話者の語をまとめて「話者N: …」で出力
        cur, buf = None, []
        for w in words:
            sp = getattr(w, "speaker", None)
            if sp != cur and buf:
                print(f"話者{cur}: {join_words(buf)}", flush=True); buf = []
            cur = sp; buf.append(w)
        if buf:
            print(f"話者{cur}: {join_words(buf)}", flush=True)

    def on_error(self, error, *a, **k):
        print(f"# エラー: {error}", flush=True)

    conn.on(LiveTranscriptionEvents.Open, on_open)
    conn.on(LiveTranscriptionEvents.Transcript, on_transcript)
    conn.on(LiveTranscriptionEvents.Error, on_error)

    options = LiveOptions(
        model=args.model, language=args.lang,
        encoding="linear16", sample_rate=SR, channels=1,
        interim_results=True, smart_format=True, punctuate=True,
        diarize=True,   # 話者分離（1マイクで誰が話したか）
    )
    print("# Deepgramに接続中…", flush=True)
    if not conn.start(options):
        raise SystemExit("接続に失敗しました")

    audio_q: "queue.Queue[bytes]" = queue.Queue()

    def cb(indata, frames, t, status):
        # float32 -> int16 PCM bytes
        pcm = (np.clip(indata[:, 0], -1, 1) * 32767).astype(np.int16).tobytes()
        audio_q.put(pcm)

    stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            device=args.device, callback=cb, blocksize=int(SR * 0.05))
    stream.start()

    def sender():
        while True:
            pcm = audio_q.get()
            if pcm is None:
                break
            conn.send(pcm)

    th = threading.Thread(target=sender, daemon=True)
    th.start()

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\n# 終了", flush=True)
    finally:
        stream.stop()
        audio_q.put(None)
        try:
            conn.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
