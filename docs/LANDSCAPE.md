# 外部ツール・API・関連研究の整理（2026-06 時点）

リアルタイムで「誰が何を言ったか」を出すために、世の中で**今すぐ使えるもの**を棚卸しし、
本研究がどこを担うべきか（＝自前で作る価値がある部分）を切り分ける。

---

## 結論を先に（北極星チェック）

**ストリーミングASRと話者分離の大部分は、もう既製品で解ける。自前で作り直す必要はない。**

決定的な事実：私たちの「1人1マイク」は、商用API/既存ツールが言う **multichannel
（チャンネル＝話者を分けて入力）** とまったく同じ構図。各チャンネルを別々に文字起こしすれば、
話者分離は処理なしで付く（Deepgram/AssemblyAIが明言）。つまり：

- **リアルタイム化（前回ぶつかった壁）は、既存のストリーミングASRを各チャンネルに当てれば解決。**
  自前のローリング文脈実装は不要（whisper_streaming等が既に実装済み）。
- **既存ツールが面倒を見ないのは「チャンネル内の漏れ込み（クロストーク）」だけ。** APIは
  各チャンネルがきれいな前提。実際は隣の声が漏れる。**ここが本研究の差別化点**で、
  漏れ込みが大きい領域（卓上マイク・近接話者・相槌）で効く。

→ 戦略：**ストリーミングASRは既製品を採用。本研究は「その前段の漏れ込み除去層」に集中する。**

---

## A. ストリーミングASR（前回の遅延問題を解くもの）

Whisperは設計上バッチ専用。無理に使わず、以下のどれかを採用するのが筋。

- **whisper_streaming / SimulStreaming**（UFAL, Macháček）— まさに前回我々が「対策」として
  考えたローリング文脈＋local agreement方式の既製実装。Whisper系をそのまま低遅延化。
  長文で約3.3秒遅延の実績。オープンソース。<https://github.com/ufal/whisper_streaming>
- **WhisperLiveKit**（QuentinFuxa）— 同時STTの実装一式、すぐ使える。
- **NVIDIA Parakeet TDT / Canary**（RNN-Transducerで元から低遅延、超高速・高精度。Open ASR
  Leaderboard上位）。ただしGPU前提。
- **Moonshine v2** — CPU/エッジ志向、Mac上でオンデバイス・リアルタイム、しかも**pyannote
  による話者識別を内蔵**。対面会議のプライバシー要件（音声を外に出さない）に最適。
- 他: Kyutai（約1秒遅延）、Voxtral（ネイティブ・ストリーミング）。

教訓：**「Whisperをストリーミング化する」自前作業はやめ、既存ラッパーかストリーミング
ネイティブのモデルに載せ替える。** 前回CPUで重かったローリング文脈も、これらなら現実的。

## B. ストリーミングの話者帰属つきASR（＝1本マイク路線の本命）

1本マイクで「重なっても誰の発話か」を出す最新モデルが既に出ている。

- **NVIDIA `multitalker-parakeet-streaming-0.6b-v1` ＋ Streaming Sortformer**（2025）—
  ストリーミングのマルチトーカーASR。**enrollment不要**で重なりも扱う。到着順の話者キャッシュ
  （AOSC）で逐次更新。1本マイク路線をやるならまずこれを試すべき。
- **G-STAR**（arXiv 2603.10468）— 全体話者追跡つき帰属認識。

## C. 商用リアルタイムAPI（who-said-whatをそのまま提供）

- **AssemblyAI** — ストリーミング話者ダイアライゼーションを**sub-300ms遅延**で提供。
  multichannelストリーミング（チャンネル＝話者）にも対応＝我々の1人1マイクに直結。
- **Deepgram** — リアルタイムmultichannel（最大20ch、チャンネルごとに別トランスクリプト）。
  まさに1人1マイクの構図をそのまま受ける。
- **Speechmatics** — リアルタイム＋ダイアライゼーション、**オンプレ提供あり**（音声を外に出さない）。

注意：クラウドAPIは音声を外部送信する。対面会議のプライバシーを考えると、オンプレ
（Speechmatics）かローカルモデル（Moonshine等）が望ましい。

## D. オンデバイス／ローカル（プライバシー・対面向け）

- **Moonshine v2** — Mac上で完結、リアルタイム＋話者識別内蔵。
- **Meetily** — オフラインの会議文字起こし＋ダイアライゼーション。

---

## 本研究の立ち位置（何を作り、何を借りるか）

| 機能 | 既製品で足りるか | 本研究の関与 |
|---|---|---|
| ストリーミングASR | ◎ 足りる（whisper_streaming / Parakeet / Moonshine） | 借りる |
| チャンネル分離（1人1マイク） | ◎ multichannelで自明に分離 | 借りる |
| **チャンネル内の漏れ込み除去** | ✕ 既製品は非対応（各chクリーン前提） | **★ 自前で担う（差別化点）** |
| 相槌・割り込みの当て分け | △ 弱い | ★ cpWERで効果を測る価値あり |
| 1本マイクの重なり分離 | ○ NVIDIA streaming multitalkerが新登場 | 評価・比較対象 |

**要するに**：「リアルタイムwho-said-what」自体は既製品でかなり解ける。本研究のオリジナリティは
**1人1マイクのクロストーク（漏れ込み）を、ストリーミングASRの前段で落とす層**にある。
これを既製のストリーミングASRと組み合わせ、漏れ込みが効く領域で「漏れ込み除去あり/なし」を
比較するのが、最も意味のある貢献になる。

---

## 次の一手（推奨）

1. **whisper_streaming（またはMoonshine v2）を各チャンネルに当てて、リアルタイム版を最短で立てる。**
   前回の自前ローリング文脈は捨て、既製のlocal-agreement実装を使う。
2. その前段に本研究の漏れ込み除去ゲートを差し込み、**「ゲートあり/なし」をリアルタイム条件で比較**。
3. ベースラインとして商用multichannel API（Deepgram等）や1本マイク（NVIDIA streaming multitalker）を置く。
4. 評価は cpWER＋レイテンシ。漏れ込みが効く動作点（中程度）に寄せる。

## 出典
- Open STT/ストリーミング: modal.com/blog/open-source-stt, gladia.io, assemblyai.com/blog/whisper-alternatives
- whisper_streaming: github.com/ufal/whisper_streaming
- NVIDIA streaming multitalker: huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1
- リアルタイム・ダイアライゼーションAPI: assemblyai.com/blog/streaming-diarization-major-upgrade, developers.deepgram.com/docs/multichannel
- オンデバイス: onresonant.com（Moonshine vs Parakeet vs Whisper）, meetily.ai
