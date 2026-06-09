"""既製品その3: Deepgram のmultichannelリアルタイム文字起こし（クラウドAPI）.

1人1マイク=multichannelとして送ると、チャンネルごとに別トランスクリプトが返る
（=話者分離が自明に付く）。本研究の構図にそのまま対応する商用ベースライン。

注意: クラウドに音声を送る。対面会議のプライバシーが要るなら、ローカル(Moonshine等)や
オンプレ(Speechmatics)を選ぶこと。無料枠あり。

準備:
  1. https://deepgram.com で無料APIキーを取得
  2. uv pip install deepgram-sdk
  3. export DEEPGRAM_API_KEY=...    （またはコード内に設定）

使い方（事前録音の多chファイル, 各chが各話者）:
  uv run python offshelf/run_deepgram_multichannel.py --wav multichannel.wav
"""
from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="多チャンネルwav（ch=話者）")
    args = ap.parse_args()

    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise SystemExit("環境変数 DEEPGRAM_API_KEY を設定してください")

    try:
        from deepgram import DeepgramClient, PrerecordedOptions, FileSource
    except ImportError:
        raise SystemExit("uv pip install deepgram-sdk を実行してください")

    dg = DeepgramClient(key)
    with open(args.wav, "rb") as f:
        payload: "FileSource" = {"buffer": f.read()}
    options = PrerecordedOptions(
        model="nova-2", multichannel=True, punctuate=True, language="en",
    )
    resp = dg.listen.prerecorded.v("1").transcribe_file(payload, options)
    chans = resp["results"]["channels"]
    for i, ch in enumerate(chans):
        alt = ch["alternatives"][0]
        print(f"[ch{i+1}] {alt['transcript']}\n")


if __name__ == "__main__":
    main()
