"""Macマイクのライブ・デモ: しゃべると数秒後に文字が出る（準リアルタイム）.

1人=1マイクなので漏れ込みゲートは不要。マイク音をためて、数秒ごとにASRし、
新しく出た文だけを逐次表示する。「ライブで議事録が流れる」体感の最短版。

準備(Mac):
  uv add sounddevice
  # parakeet が未導入なら: uv add parakeet-mlx -U   （ffmpeg必須）

使い方:
  uv run python offshelf/live_mic.py --engine parakeet            # 実マイクでライブ
  uv run python offshelf/live_mic.py --engine parakeet --wav offshelf/ami_raw/mic0.wav  # ファイルで擬似ライブ
  Ctrl+C で終了。
"""
from __future__ import annotations

import argparse
import os
import tempfile
import threading
import time

import numpy as np
import soundfile as sf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SR = 16000


def asr_sentences(audio, engine, model_obj, lang="ja"):
    if engine == "parakeet":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            sf.write(tf.name, audio, SR); path = tf.name
        res = model_obj.transcribe(path); os.unlink(path)
        return [(s.start, s.end, s.text.strip()) for s in res.sentences if s.text.strip()]
    elif engine == "mlx-whisper":
        import mlx_whisper
        r = mlx_whisper.transcribe(audio, path_or_hf_repo=model_obj, language=lang,
                                   condition_on_previous_text=False)
        return [(s["start"], s["end"], s["text"].strip()) for s in r.get("segments", []) if s["text"].strip()]
    else:
        from spkattr.asr import transcribe_words
        words = transcribe_words(model_obj, audio, SR)
        sents = []; cur = []
        for w, s, e in words:
            if cur and s - cur[-1][2] > 0.8:
                sents.append((cur[0][1], cur[-1][2], " ".join(x[0] for x in cur))); cur = []
            cur.append((w, s, e))
        if cur:
            sents.append((cur[0][1], cur[-1][2], " ".join(x[0] for x in cur)))
        return sents


class AudioBuffer:
    """スレッドセーフに伸びていく録音バッファ."""
    def __init__(self):
        self.data = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()
        self.done = False

    def append(self, x):
        with self.lock:
            self.data = np.concatenate([self.data, x.astype(np.float32)])

    def length_s(self):
        with self.lock:
            return len(self.data) / SR

    def slice(self, a_s, b_s):
        with self.lock:
            return self.data[int(a_s * SR):int(b_s * SR)].copy()


def feed_from_wav(buf, path, realtime=True):
    import librosa
    y, _ = librosa.load(path, sr=SR)
    step = int(0.1 * SR)
    for i in range(0, len(y), step):
        buf.append(y[i:i + step])
        if realtime:
            time.sleep(0.1)
    buf.done = True


def feed_from_mic(buf, device=None):
    import sounddevice as sd

    def cb(indata, frames, t, status):
        buf.append(indata[:, 0].copy())
    stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            device=device, callback=cb)
    stream.start()
    return stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["mlx-whisper", "parakeet", "whisper"], default="mlx-whisper",
                    help="日本語は mlx-whisper か whisper。parakeetは英語/欧州語のみ")
    ap.add_argument("--lang", default="ja", help="言語コード（ja=日本語, en=英語）")
    ap.add_argument("--mlx-model", default="mlx-community/whisper-large-v3-turbo",
                    help="日本語OK・実績あり。軽くするなら mlx-community/whisper-small")
    ap.add_argument("--no-vad", dest="vad", action="store_false", help="無音スキップ(幻聴抑制)を無効化")
    ap.add_argument("--wav", default=None, help="指定すると実マイクの代わりにファイルで擬似ライブ")
    ap.add_argument("--chunk-s", type=float, default=4.0)
    ap.add_argument("--context-s", type=float, default=1.5)
    ap.add_argument("--name", default="あなた")
    ap.add_argument("--device", default=None, help="入力デバイス名/番号(任意)")
    ap.add_argument("--debug", action="store_true", help="音量・発話判定の診断行を表示")
    args = ap.parse_args()

    if args.engine == "parakeet" and args.lang != "en":
        print("# 注意: parakeetは日本語非対応です。--engine mlx-whisper を使ってください。", flush=True)

    print("# 準備中（モデル読み込み）…", flush=True)
    if args.engine == "parakeet":
        from parakeet_mlx import from_pretrained
        model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    elif args.engine == "mlx-whisper":
        model = args.mlx_model   # mlx_whisper は repo名を都度渡す
    else:
        from spkattr.asr import Transcriber
        model = Transcriber("base", language=args.lang)

    buf = AudioBuffer()
    if args.wav:
        threading.Thread(target=feed_from_wav, args=(buf, args.wav), daemon=True).start()
        print(f"# 擬似ライブ: {args.wav}", flush=True)
    else:
        stream = feed_from_mic(buf, args.device)
        print("# マイク開始。話してください（Ctrl+Cで終了）", flush=True)

    print(f"# {args.chunk_s:.0f}秒ごとに処理 / 遅延の目安 約{args.chunk_s:.0f}秒\n", flush=True)
    emitted = set()
    last_t = 0.0
    try:
        while True:
            now = buf.length_s()
            if now >= last_t + args.chunk_s:
                a = max(0.0, last_t - args.context_s)
                b = now
                seg = buf.slice(a, b)
                new = buf.slice(last_t, b)
                level = float(np.abs(new).max()) if len(new) else 0.0
                has_speech = True
                if args.vad:
                    from spkattr.vad import energy_vad
                    has_speech = len(new) > 0 and energy_vad(new, SR).mean() > 0.04
                # 診断: マイク音量と発話判定（--debug 時のみ）
                if args.debug:
                    mark = "🎤発話" if (level > 1e-3 and has_speech) else ("…無音" if level > 1e-3 else "⚠️無入力(マイク権限?)")
                    print(f"# [{now:5.1f}s] 音量 {level:.3f} {mark}", flush=True)
                if level > 1e-3 and has_speech:
                    try:
                        for s, e, text in asr_sentences(seg, args.engine, model, args.lang):
                            t = text.strip()
                            if not t or t in emitted:   # 同じ文の再掲だけ除く
                                continue
                            emitted.add(t)
                            print(f"[{now:6.1f}s] {args.name}: {t}", flush=True)
                    except Exception as ex:
                        print(f"# ASRエラー: {ex!r}", flush=True)
                last_t = now
            if buf.done and buf.length_s() <= last_t:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n# 終了")


if __name__ == "__main__":
    main()
