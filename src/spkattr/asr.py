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

    def transcribe_span(self, audio: np.ndarray, sr: int) -> str:
        # Whisper は 16kHz float32 を想定
        a = audio.astype(np.float32)
        if np.abs(a).max() > 0:
            a = a / np.abs(a).max() * 0.95
        result = self.model.transcribe(a, fp16=False)
        return result.get("text", "").strip()


def transcribe(channels: np.ndarray, sr: int, owner_names, segments,
               transcriber: "Transcriber", pad_s: float = 0.1) -> list[Utterance]:
    """本人と判定したスパンだけ文字起こしして話者付きトランスクリプトを返す."""
    n_ch, n = channels.shape
    spans = merge_own_spans(segments, n_ch, owner_names)
    utts: list[Utterance] = []
    for ch, ivs in spans.items():
        for s, e in ivs:
            a = int(max(0, (s - pad_s) * sr))
            b = int(min(n, (e + pad_s) * sr))
            if b - a < int(0.2 * sr):
                continue
            text = transcriber.transcribe_span(channels[ch, a:b], sr)
            if text:
                utts.append(Utterance(s, e, owner_names[ch], text))
    utts.sort(key=lambda u: u.start)
    return utts
