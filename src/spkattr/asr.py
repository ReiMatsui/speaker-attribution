"""話者ラベル付き文字起こし.

パイプラインが「本人の発話」と判定した区間だけをASR(Whisper)にかけ、
漏れ込みと判定した区間は文字起こししない(ゲート)。
出力は (start, end, speaker, text) のリスト = 時系列の書き起こし。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Utterance:
    start: float
    end: float
    speaker: str
    text: str


def merge_own_spans(segments, n_ch, owner_names, max_gap_s=0.6):
    """is_own_voice な窓を、チャンネルごとに連続スパンへ統合する.

    返り値: {channel: [(start, end), ...]}
    隣接する本人区間の間が max_gap_s 以下なら1つにつなぐ。
    """
    per_ch = {ch: [] for ch in range(n_ch)}
    for seg in segments:
        if seg.is_own_voice:
            per_ch[seg.channel].append((seg.start, seg.end))
    spans = {}
    for ch, ivs in per_ch.items():
        ivs.sort()
        merged = []
        for s, e in ivs:
            if merged and s - merged[-1][1] <= max_gap_s:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        spans[ch] = merged
    return spans


class Transcriber:
    """Whisper をラップしたASR."""

    def __init__(self, model_name: str = "base", device: str = "cpu"):
        import whisper
        self.model = whisper.load_model(model_name, device=device)

    def transcribe_span(self, audio: np.ndarray, sr: int, robust: bool = True) -> str:
        # Whisper は 16kHz float32 を想定
        a = audio.astype(np.float32)
        if np.abs(a).max() > 0:
            a = a / np.abs(a).max() * 0.95
        opts = dict(fp16=False, language="en")
        if robust:
            # ハルシネーション・繰り返し・漏れ込み拾いを抑える設定
            opts.update(
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                temperature=0.0,
            )
        result = self.model.transcribe(a, **opts)
        return result.get("text", "").strip()


def transcribe_words(transcriber, audio: np.ndarray, sr: int, robust: bool = True):
    """全体を一度に文字起こしし、単語ごとの (word, start, end) を返す.

    チャンネルを断片に切らず全体を渡すので、Whisperが文脈を使えてASR精度が出る。
    """
    a = audio.astype(np.float32)
    if np.abs(a).max() > 0:
        a = a / np.abs(a).max() * 0.95
    opts = dict(fp16=False, language="en", word_timestamps=True)
    if robust:
        opts.update(condition_on_previous_text=False, no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4, temperature=0.0)
    r = transcriber.model.transcribe(a, **opts)
    out = []
    for seg in r.get("segments", []):
        for w in seg.get("words", []):
            out.append((w["word"].strip(), float(w["start"]), float(w["end"])))
    return out


def _own_mask_from_segments(segments, n_ch, hop_s=0.25):
    """segments から、チャンネルごとの「本人発話」判定を時刻→boolで引ける表にする."""
    per_ch = {ch: {} for ch in range(n_ch)}  # ch -> {hop_index: own_bool}
    for seg in segments:
        idx = int(round(seg.start / hop_s))
        # 同一窓が重複したら「本人」を優先(OR)
        per_ch[seg.channel][idx] = per_ch[seg.channel].get(idx, False) or seg.is_own_voice
    return per_ch, hop_s


def attribute_words(channels: np.ndarray, sr: int, owner_names, segments,
                    transcriber: "Transcriber", robust: bool = True) -> list[Utterance]:
    """全体ASR→単語ごとに本人時間かで取捨選択する方式.

    各チャンネルを全体文字起こしし、各単語の中心時刻でそのチャンネルが
    「本人発話」と判定されていれば採用、漏れ込み時間なら捨てる。
    """
    n_ch, n = channels.shape
    own, hop_s = _own_mask_from_segments(segments, n_ch)
    utts: list[Utterance] = []
    for ch in range(n_ch):
        words = transcribe_words(transcriber, channels[ch], sr, robust)
        kept = []
        for word, s, e in words:
            c = (s + e) / 2
            idx = int(round(c / hop_s))
            # 近傍3hopのいずれかで本人ならば採用(境界に寛容)
            is_own = any(own[ch].get(idx + d, False) for d in (-1, 0, 1))
            if is_own:
                kept.append((word, s, e))
        if kept:
            text = " ".join(w for w, _, _ in kept)
            utts.append(Utterance(kept[0][1], kept[-1][2], owner_names[ch], text))
    utts.sort(key=lambda u: u.start)
    return utts


def transcribe(channels: np.ndarray, sr: int, owner_names, segments,
               transcriber: "Transcriber", pad_s: float = 0.4,
               merge_gap_s: float = 2.5, ramp_s: float = 0.03,
               robust: bool = True) -> list[Utterance]:
    """連続マスク方式で話者付きトランスクリプトを返す(推奨).

    断片に切らず、本人発話と判定した時間以外を無音化した「全長の連続音声」を
    チャンネルごとに作り、それを一括でASRする。Whisperが文脈を保てるため
    断片化による単語落ちを防げる(検証で WER 30%→13%)。

    pad_s        : 本人区間の前後に足す余白(取りこぼし防止)。
    merge_gap_s  : この間隔以下の本人区間を1つにつなぐ(発話を分断しない)。
    ramp_s       : マスク境界のスムージング(クリック音防止)。
    """
    n_ch, n = channels.shape
    spans = merge_own_spans(segments, n_ch, owner_names, max_gap_s=merge_gap_s)
    ramp = max(1, int(ramp_s * sr))
    kernel = np.ones(ramp) / ramp
    utts: list[Utterance] = []
    for ch in range(n_ch):
        if not spans.get(ch):
            continue
        mask = np.zeros(n)
        for s, e in spans[ch]:
            a = int(max(0, (s - pad_s) * sr))
            b = int(min(n, (e + pad_s) * sr))
            mask[a:b] = 1.0
        mask = np.clip(np.convolve(mask, kernel, mode="same"), 0.0, 1.0)
        text = transcriber.transcribe_span(channels[ch] * mask, sr, robust=robust)
        if text:
            s0 = spans[ch][0][0]
            e0 = spans[ch][-1][1]
            utts.append(Utterance(s0, e0, owner_names[ch], text))
    utts.sort(key=lambda u: u.start)
    return utts
