"""声紋(声の特徴)による本人照合.

Resemblyzer の VoiceEncoder で発話を 256 次元の埋め込みに変換し、
事前登録(enrollment)した各話者の埋め込みとのコサイン類似度で本人を判定する。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav


@dataclass
class Enrollment:
    """登録済み話者の声紋データベース."""

    encoder: VoiceEncoder
    profiles: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def create(cls, device: str = "cpu") -> "Enrollment":
        return cls(encoder=VoiceEncoder(device))

    def register(self, name: str, wav: np.ndarray) -> None:
        """話者 name の声を登録する(16kHz mono を想定)."""
        emb = self.encoder.embed_utterance(preprocess_wav(wav, source_sr=16000))
        self.profiles[name] = emb

    def embed(self, wav: np.ndarray) -> np.ndarray:
        return self.encoder.embed_utterance(preprocess_wav(wav, source_sr=16000))

    def identify(self, wav: np.ndarray) -> tuple[str | None, float, dict[str, float]]:
        """発話を登録話者と照合.

        返り値 (best_name, best_score, all_scores)
        """
        if not self.profiles:
            raise ValueError("登録話者がいません。先に register() してください。")
        emb = self.embed(wav)
        scores = {name: float(np.dot(emb, prof)) for name, prof in self.profiles.items()}
        best = max(scores, key=scores.get)
        return best, scores[best], scores

    def verify(self, wav: np.ndarray, name: str, thresh: float = 0.6) -> tuple[bool, float]:
        """発話が指定話者 name 本人かを閾値判定(speaker verification)."""
        emb = self.embed(wav)
        score = float(np.dot(emb, self.profiles[name]))
        return score >= thresh, score


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
