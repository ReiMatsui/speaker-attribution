"""既製品その2: Moonshine（オンデバイス・低遅延ASR, Mac最適化）.

1本のwavを文字起こしして、リアルタイム志向モデルの速さ・精度を体感する。
Moonshineはエッジ/CPU向けで、Macローカルで完結（音声を外に出さない）。

インストール（Mac, uvプロジェクト内）:
  uv pip install useful-moonshine-onnx@git+https://github.com/usefulsensors/moonshine.git#subdirectory=moonshine-onnx
  # うまくいかない場合は本家 https://github.com/usefulsensors/moonshine の最新手順を参照

使い方:
  uv run python offshelf/run_moonshine.py --wav samples/spkA.wav
  # 1本マイクで全員の声を混ぜたものを渡すと「重なりにどれだけ強いか」を体感できる
"""
from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--model", default="moonshine/base", help="moonshine/tiny or moonshine/base")
    args = ap.parse_args()

    try:
        import moonshine_onnx as moonshine
    except ImportError:
        raise SystemExit(
            "moonshine が未インストールです。READMEの手順か docs/TRY_OFFSHELF.md を参照してください。")

    t0 = time.time()
    text = moonshine.transcribe(args.wav, args.model)
    dt = time.time() - t0
    print("[transcript]", text[0] if isinstance(text, list) else text)
    print(f"# 処理 {dt:.2f}s")


if __name__ == "__main__":
    main()
