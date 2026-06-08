# speaker-attribution

1人1マイク環境で、各マイクが拾った発話を **その持ち主の発話として正しく認識する** ことを目指す研究プロジェクト。

## 課題

理想は「マイク=その人」だが、現実には隣の人の声(特に相槌)が全員のマイクに漏れ込む(クロストーク)。
このため「このマイクで今鳴っている声は、本当に持ち主のものか?」が崩れる。

予備検証では、相手の声が -10dB 混ざるだけで本人の声紋スコアが 0.76 → 0.58 に低下した。
ここが本プロジェクトの中心課題。

## アプローチ

3段を1本に統合する。

1. **VAD** (`vad.py`) — 各マイクで発話区間を検出
2. **チャンネル間相関** (`crosstalk.py`) — 同じ波形が複数マイクに遅延付きで現れる=漏れ込み、と GCC-PHAT で判定。声量の個人差に依存しない物理的手がかり
3. **声紋照合** (`embedding.py`) — 残った区間を、事前登録した本人の声紋と照合して確定

評価用の多チャンネル・クロストーク付きデータは `synth.py` で合成できる(正解ラベル付き)。

## セットアップ (uv)

```bash
uv sync                              # 依存をインストール
uv run --extra dev pytest            # テスト

# 録音不要で実音声を試す: 別話者4名の公開サンプルを samples/ に同梱済み
uv run python scripts/demo_synth.py --use-embedding   # samples/ を自動使用

# サンプルを取り直す場合
uv run python scripts/fetch_samples.py

# 自前のwavで試す場合
uv run python scripts/demo_synth.py --real a.wav b.wav --use-embedding

# 実験の中身を「聴いて・見て」確認するHTMLレポートを生成
uv run python scripts/visualize.py            # report.html を生成 → ブラウザで開く
uv run python scripts/visualize.py --leak-db -6 --overlap-s 4   # 難しい条件
```

> 注: `samples/` には別話者の公開音声(VOiCES / LibriSpeech / Silero)を
> 16kHzモノラルに整えて同梱。`--real` を付けなければ自動で使われます。

## 構成

```
src/spkattr/
  vad.py        エネルギーベースVAD(Silero等に差し替え可)
  crosstalk.py  GCC-PHATによる漏れ込み判定
  embedding.py  Resemblyzerによる声紋登録・照合
  synth.py      多チャンネル・クロストーク合成 + 正解ラベル生成
  pipeline.py   3段の統合パイプライン
  evaluate.py   帰属精度・漏れ込み棄却率の評価
scripts/demo_synth.py  エンドツーエンドのデモ
tests/                 ユニットテスト
```

## ロードマップ

- [x] 統合プロトタイプ(VAD + 相関 + 声紋)
- [ ] 実データ評価(AMI会議コーパス: 1人1ヘッドセットマイク+正解ラベル)
- [ ] 重複時のロバスト化(target-speaker抽出 / TS-VAD)
- [ ] 失敗分析(相槌・似た声・同時大声)
- [ ] 必要に応じてカメラ(口の動き)による補助

## 評価指標

- `precision` / `recall` / `f1` : 本人発話の帰属精度
- `crosstalk_rejection` : 漏れ込みを正しく棄却できた割合
- 声紋の分離性(本人内 vs 他人間のコサイン類似度分布)
