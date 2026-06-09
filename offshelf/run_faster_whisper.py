"""既製品その1: faster-whisper を「1人1マイク」に当てる.

各wav = 1人のマイク、とみなして別々に文字起こしする(=multichannel方式)。
チャンネルを分けて入れれば話者分離は自明に付く、という最も手軽な構成。
精度とRTF(実時間比)を体感するためのもの。漏れ込み除去はしない素の既製品。

使い方:
  uv run python offshelf/run_faster_whisper.py --dir samples --model base
  # 自分のwav群を渡す場合は --dir <フォルダ>（各wav=各話者）
"""
from __future__ import annotations

import argparse
import glob
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="samples", help="各wav=各話者マイク のフォルダ")
    ap.add_argument("--model", default="base", help="tiny/base/small/medium/large-v3")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    files = sorted(glob.glob(os.path.join(args.dir, "*.wav")))
    if not files:
        raise SystemExit(f"{args.dir} に wav がありません")

    total_audio = 0.0
    t0 = time.time()
    print(f"# faster-whisper / model={args.model} / {len(files)}ch\n")
    for f in files:
        name = os.path.basename(f).split(".")[0]
        segments, info = model.transcribe(f, language="en", vad_filter=True)
        total_audio += info.duration
        text = " ".join(s.text.strip() for s in segments).strip()
        print(f"[{name}] {text}\n")
    dt = time.time() - t0
    print(f"# 処理 {dt:.1f}s / 音声 {total_audio:.1f}s / RTF={dt/max(total_audio,1e-9):.2f}")


if __name__ == "__main__":
    main()
