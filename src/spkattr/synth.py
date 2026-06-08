"""1人録音から多チャンネルのクロストーク付きデータを合成する.

各話者に専用マイクを割り当て、そのチャンネルには本人のクリーン音声を入れる。
他話者の声は「減衰 + 遅延」をかけて漏れ込みとして加算する。
正解ラベル(どの時刻に誰が話したか)も同時に生成できるので評価に使える。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SynthConfig:
    sr: int = 16000
    leak_db: float = -12.0          # 漏れ込みの減衰量(dB)。小さいほど漏れ大
    delay_ms: float = 1.5           # マイク間の到達遅延
    noise_db: float = -40.0         # 背景ノイズ
    seed: int = 0


def _delay(x: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 0:
        return x
    return np.concatenate([np.zeros(samples), x])[: len(x)]


def synthesize(sources: list[np.ndarray], cfg: SynthConfig | None = None):
    """sources[i] = 話者 i のクリーン音声(同一長, 16kHz, mono).

    返り値:
      channels : shape (n_spk, n_samples) のマルチチャンネル波形
      labels   : shape (n_spk, n_samples) の bool。各話者が発話中か(VAD正解)
    各話者のチャンネル i = 本人 + Σ_j!=i (減衰・遅延した話者 j)。
    """
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    n = max(len(s) for s in sources)
    srcs = [np.pad(s, (0, n - len(s))) for s in sources]
    n_spk = len(srcs)

    leak_gain = 10 ** (cfg.leak_db / 20)
    delay_samp = int(cfg.sr * cfg.delay_ms / 1000)
    noise_gain = 10 ** (cfg.noise_db / 20)

    channels = np.zeros((n_spk, n))
    for i in range(n_spk):
        ch = srcs[i].copy()                       # 本人(クリーン)
        for j in range(n_spk):
            if i == j:
                continue
            jitter = rng.integers(0, delay_samp + 1)
            ch = ch + leak_gain * _delay(srcs[j], delay_samp + int(jitter))
        ch = ch + noise_gain * rng.standard_normal(n)
        channels[i] = ch

    # 正解VADラベル: 各クリーン音源のエネルギー包絡から作る
    labels = np.zeros((n_spk, n), dtype=bool)
    win = int(cfg.sr * 0.025)
    for i, s in enumerate(srcs):
        env = np.convolve(s ** 2, np.ones(win) / win, mode="same")
        thr = np.percentile(env, 60) * 0.1 + 1e-9
        labels[i] = env > thr
    return channels, labels
