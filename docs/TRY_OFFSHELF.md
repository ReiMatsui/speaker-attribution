# 既製品を試すガイド（リアルタイムwho-said-whatの精度を体感する）

目的：自前実装に深入りする前に、**今すぐ使える既製品で精度と遅延を体感する**。
あなたの環境は Mac（Apple Silicon）想定なので、ローカル優先で並べる。

まず入力音声を作る（`samples/` の4話者から、1本マイク版と多ch版を生成）:

```bash
uv run python offshelf/make_test_audio.py
# → offshelf/mix_singlemic.wav（1本マイク） と offshelf/multichannel.wav（4ch）
```

---

## 試す優先順位（おすすめ順）

### ⭐ ①' parakeet-mlx × 1人1マイク（M4 Pro ローカルの本命・最有力）
NVIDIA Parakeet（精度トップクラス）をApple Silicon向けMLXで実行。高精度・高速
（Whisperの約30倍速）・ローカル完結・APIキー不要。`transcribe_stream` が左右の文脈窓を
持つので、**前回ぶつかった「チャンク化で文脈が切れる」問題に最初から対応**している。
M4 Pro なら今これが最有力。Mac(Apple Silicon)専用。

```bash
brew install ffmpeg
uv add parakeet-mlx -U
# 1人1マイクの精度を見る:
uv run python offshelf/run_parakeet_mlx.py --dir samples
# リアルタイム(ストリーミング)の遅延と精度を体感:
uv run python offshelf/make_test_audio.py
uv run python offshelf/run_parakeet_mlx_stream.py --wav offshelf/mix_singlemic.wav --chunk 1.0
```

### ① faster-whisper × 1人1マイク（一番手軽・まず最初に）
各wav=各話者として別々に文字起こし。チャンネルを分ければ話者分離は自明に付く。
漏れ込み除去なしの素の既製品。精度とRTFの基準になる。

```bash
uv pip install faster-whisper
uv run python offshelf/run_faster_whisper.py --dir samples --model base
# モデルを small / medium に上げると精度が上がる（遅くなる）
```

### ② Moonshine × 1本マイク（オンデバイス・低遅延、Mac最適）
1本に混ぜた音声を渡し、「1本マイクで重なりにどれだけ強いか」を体感。
Macローカルで完結し音声を外に出さない。話者識別も内蔵（v2）。

```bash
uv pip install useful-moonshine-onnx@git+https://github.com/usefulsensors/moonshine.git#subdirectory=moonshine-onnx
uv run python offshelf/run_moonshine.py --wav offshelf/mix_singlemic.wav
```
（インストール手順は変わりやすいので、うまくいかなければ
 https://github.com/usefulsensors/moonshine の最新READMEを確認）

### ③ whisper_streaming（本物のストリーミング遅延を体感）
前回我々が「対策」と考えたローリング文脈＋local agreementの既製実装。
真のリアルタイム遅延を体感したいとき。リポジトリ方式。

```bash
git clone https://github.com/ufal/whisper_streaming
cd whisper_streaming
uv pip install librosa soundfile faster-whisper
# マイク or ファイルから（READMEに各種起動例あり）
python3 whisper_online.py ../offshelf/mix_singlemic.wav --model base --language en --min-chunk-size 1
```

### ④ 商用API（最高精度のベースライン、要キー・要ネット）
クラウドに音声を送る点に注意（プライバシー）。無料枠あり。1人1マイク=multichannelに直結。

```bash
uv pip install deepgram-sdk
export DEEPGRAM_API_KEY=...   # https://deepgram.com で取得
uv run python offshelf/run_deepgram_multichannel.py --wav offshelf/multichannel.wav
```
AssemblyAI（ストリーミング話者分離 sub-300ms）や Speechmatics（オンプレ可）も同種。

### （GPUがあれば）⑤ NVIDIA streaming multitalker（1本マイクの本命）
1本マイクで重なっても誰の発話かを出すストリーミングモデル（enrollment不要）。
NeMo + GPU が必要なので、GPU環境がある時に。
モデル: `nvidia/multitalker-parakeet-streaming-0.6b-v1`

---

## 比較の見方

- **①と本研究の出力を比べる** → 漏れ込み除去ありなしで、漏れ込みが効く条件でどれだけ差が出るか。
- **②(1本) と ①(1人1マイク) を比べる** → マイクを分ける価値（重なりへの強さ）を体感。
- **④(商用) を上限の目安に** → ローカルでどこまで迫れるか。

評価を厳密にやるなら、`scripts/eval_*` と同じく LibriSpeech公式トランスクリプトで WER を測る。
ここでは「まず動かして感覚をつかむ」のが目的。

## どれを最初に？
**M4 Pro なら ①' parakeet-mlx を最初に。** ローカル・高精度・高速・APIキー不要で、
リアルタイム(`transcribe_stream`)まで一気に試せる。`run_parakeet_mlx.py`(精度)→
`run_parakeet_mlx_stream.py`(遅延)の順。
精度の上限の目安が欲しければ④商用API、軽さ重視なら①faster-whisper、を併用。
