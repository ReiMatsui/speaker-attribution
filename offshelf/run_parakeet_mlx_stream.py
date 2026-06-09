"""parakeet-mlx の「リアルタイム・ストリーミング」を体感するデモ.

1本のwavを擬似的に小さなチャンクで送り込み、確定テキスト(finalized)と
暫定テキスト(draft)がどう更新されるか、各チャンクの処理時間（=計算遅延）を表示する。
parakeet-mlx は左右の文脈窓(context_size)を持つので、チャンクを短くしても
文脈を保てる＝我々が前回ぶつかった「チャンク化で文脈が切れる」問題への解になっているか
を、自分の耳と目で確かめるためのもの。Mac(Apple Silicon)専用。

準備:
  brew install ffmpeg
  uv add parakeet-mlx -U

使い方:
  # 1本マイク(全員混合)で重なりへの強さを見る:
  uv run python offshelf/make_test_audio.py   # mix_singlemic.wav を作る
  uv run python offshelf/run_parakeet_mlx_stream.py --wav offshelf/mix_singlemic.wav --chunk 1.0
  # 1人分のきれいな音声なら:
  uv run python offshelf/run_parakeet_mlx_stream.py --wav samples/spkB.wav --chunk 0.5
"""
from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--chunk", type=float, default=1.0, help="チャンク長(秒)=遅延の主因")
    ap.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v3")
    ap.add_argument("--left", type=int, default=256, help="左文脈フレーム")
    ap.add_argument("--right", type=int, default=256, help="右文脈フレーム(大きいほど精度↑遅延↑)")
    args = ap.parse_args()

    try:
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import load_audio
    except ImportError:
        raise SystemExit("parakeet-mlx 未インストール。Mac で `uv add parakeet-mlx -U`（ffmpeg必須）")

    model = from_pretrained(args.model)
    sr = model.preprocessor_config.sample_rate
    audio = load_audio(args.wav, sr)
    step = int(args.chunk * sr)

    print(f"# parakeet-mlx streaming / chunk={args.chunk}s / context=({args.left},{args.right})\n")
    worst = 0.0
    wall0 = time.time()
    with model.transcribe_stream(context_size=(args.left, args.right)) as tr:
        for i in range(0, len(audio), step):
            chunk = audio[i:i + step]
            t0 = time.time()
            tr.add_audio(chunk)
            dt = time.time() - t0
            worst = max(worst, dt)
            txt = tr.result.text.strip()
            audio_t = (i + step) / sr
            print(f"[{audio_t:5.1f}s] 計算 {dt*1000:4.0f}ms | {txt[-80:]}")
    total = time.time() - wall0
    audio_s = len(audio) / sr
    print(f"\n# 最終テキスト:\n{tr.result.text.strip()}")
    print(f"\n# 1チャンクの最悪計算遅延: {worst*1000:.0f}ms（チャンク長 {args.chunk*1000:.0f}ms より十分小さければ実時間で間に合う）")
    print(f"# 全体 {total:.1f}s / 音声 {audio_s:.1f}s / RTF={total/audio_s:.2f}")


if __name__ == "__main__":
    main()
