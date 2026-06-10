# 引き継ぎプロンプト（新しいチャットの冒頭に貼る）

あなたは、対面会議をリアルタイムに「誰が・何を」話したか日本語で文字起こしする
研究/実装プロジェクトを引き継ぎます。作業フォルダは `~/speaker-attribution`
（Mac, Apple Silicon M4 Pro, uv管理, Python 3.13）。まず `docs/` 配下
（RESEARCH_LOG.md / SUMMARY.md / LANDSCAPE.md / この HANDOFF.md）と CLAUDE.md を読んでください。

## ゴール（不変）
対面会議を、低遅延・リアルタイムで「誰が何を言ったか」に文字起こしする（日本語）。

## ここまでの結論（重要）
- 当初は「1人1マイク＋自前の漏れ込み除去ゲート（VAD→チャンネル間相関 or 声紋）」を研究。
  合成データとAMI実会議で検証し、「強いASRほど漏れ込みで崩れる→ゲートが効く」ことを確認。
  ただし**良いヘッドセット＋強ASRなら生でもそこそこ動き、ゲートは"仕上げ掃除"**という実データ知見も得た。
- リアルタイム化を実機検証した結果、**ローカルWhisper（mlx-whisper/parakeet）は日本語ライブで品質が不安定**
  （幻聴「ご視聴ありがとうございました」、繰り返し暴走、ラグのムラ）。parakeetは日本語非対応。
- **最終的に Soniox のリアルタイムAPI（日本語・話者分離内蔵）を本線に採用**。
  1本のマイクで「話者1/話者2…」の who-said-what が、低遅延・高品質で取れることを実機確認済み。
  → 多マイク・自前ゲート・デバイス同期は不要になった。研究のゲートは「同時発話が激しい/各人マイク固定で
  堅くしたい」場合の**オプションの上乗せ**という位置づけ。

## 現在の本線ツール
- `offshelf/live_soniox.py` … 1マイク＋Soniox（日本語・話者分離）リアルタイム議事録ツール。**これが本命。**
  - 必要: `uv add websockets sounddevice` / `export SONIOX_API_KEY=...`（Sonioxは残高必要・少額チャージ）
  - 実行: `uv run python offshelf/live_soniox.py`（`--wav` でファイル擬似ライブも可、`--out` で保存先指定）
  - 機能（2026-06-10実装、モックサーバで検証済み・実機ライブ確認は未）:
    ① Markdown議事録を `transcripts/日時.md` に自動保存（発話確定ごとに原子的書き出し＝クラッシュ安全）
    ② 実行中に「1=松井」と打つと話者1を実名化（過去の発言も遡って置換）
    ③ 話者ごと色分けライブ表示、確定前テキストは薄く表示、`[MM:SS]` タイムスタンプ付き
    ④ HTML版も同時保存。`open transcripts/～.html` でブラウザ表示（ライブ中2秒ごと自動更新）
- 旧実装は `archive/offshelf/` に退避（2026-06-10）: `live_mic.py`（ローカルmlx-whisper, 不安定）,
  `live_mic_deepgram.py`（Deepgram代替, 日本語＋話者分離, 無料$200枠）,
  `meeting_server.py`（複数iPhone→Mac, HTTPS版）, run_*/make_*/transcribe_meeting_* と生成音声データ。
  研究コード `src/spkattr/` と `scripts/`、AMIデータ `offshelf/ami_raw/` は現役のまま。

## 次にやること（おすすめ順）
1. ~~`live_soniox.py` を"使える議事録ツール"に仕上げる~~ → **実装済み（2026-06-10）**。
   残り: 実マイク＋実会議でのライブ確認（保存・実名コマンド・表示の使い勝手）。
2. 必要なら**複数iPhone対応**（各自iPhone→Soniox、または1マイク＋Soniox分離で十分か比較）。
3. 同時発話が課題なら、研究の漏れ込みゲートを上乗せして比較（src/spkattr に実装あり）。

## 研究コード（src/spkattr/）— 必要時に使う
VAD / crosstalk(GCC-PHAT) / embedding(Resemblyzer) / pipeline / streaming(StreamingGate) / asr(Whisper) / synth / evaluate。
評価スクリプトは scripts/（eval_separation, eval_ami, eval_streaming_gate 等）。AMI実データDL済み。

## 重要な落とし穴（ハマりどころ）
- zshに `# コメント` 付きコマンドを貼ると引数エラー → コメントなしで1行ずつ実行。
- gitがマウントのロックで失敗することがある → `rm -f .git/*.lock` してからコミット。
- parakeetは日本語非対応（英語/欧州語）。kotoba-whisper-mlx は mlx_whisper と非互換で空出力。
  → 日本語ローカルは whisper-large-v3-turbo（不安定）、本命はSoniox。
- iOS Safariのマイクは**HTTPS必須**（meeting_serverは自己署名証明書を自動生成、各iPhoneで許可が必要）。
- macOSのターミナルに**マイク権限**が必要（無いと無音）。
- **APIキーはチャットに貼らない**（露出したら失効・再生成）。

## 方針
精度の数字より「実際に分離して議事録として使えるか」を優先。良ければ妥協して実用化を進める。
日本語で簡潔に、要点先出し。作業前に計画を共有し、長い調査や実装の前に確認を取る。
