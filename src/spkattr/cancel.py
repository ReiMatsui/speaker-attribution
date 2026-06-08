"""漏れ込み(クロストーク)を音そのものから除去する手法.

ゲートは「いつ書き起こすか」を決めるだけで、本人の発話中もマイクには他人の声が
混ざったまま。ここを実際にクリーン化してからASRに渡すのが狙い。

提供する手法:
- tf_channel_mask : 時間周波数ごとに「最もエネルギーの大きいマイク=その音の持ち主」
                    とみなし、各マイクで自分が支配的なT-F点だけ残す(ソフトマスク)。
                    1人1マイクの幾何(本人は本人マイクで最大)を利用。学習不要。
- lsq_cancel      : 他チャンネルを参照にした最小二乗FIRで、相関する漏れ込み成分を減算。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft, istft


def tf_channel_mask(channels: np.ndarray, sr: int, power: float = 2.0,
                    floor: float = 0.0, nperseg: int = 512,
                    noverlap: int = 384) -> np.ndarray:
    """時間周波数の支配チャンネル・ソフトマスクで各チャンネルをクリーン化."""
    n_ch, n = channels.shape
    specs = []
    for ch in range(n_ch):
        _, _, Z = stft(channels[ch], sr, nperseg=nperseg, noverlap=noverlap)
        specs.append(Z)
    mag = np.stack([np.abs(Z) ** power for Z in specs])  # (C,F,T)
    denom = mag.sum(axis=0) + 1e-12

    cleaned = np.zeros_like(channels)
    for ch in range(n_ch):
        w = mag[ch] / denom
        if floor > 0:
            w = np.clip(w, floor, 1.0)
        _, x = istft(specs[ch] * w, sr, nperseg=nperseg, noverlap=noverlap)
        x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
        cleaned[ch] = x
    return cleaned


def lsq_cancel(channels: np.ndarray, taps: int = 32) -> np.ndarray:
    """他チャンネルを参照にした最小二乗FIRで漏れ込みを減算.

    目標チャンネル y を、他チャンネルの遅延コピー(±taps)から線形予測し、
    予測できた成分(=相関する漏れ込み)を引く。本人成分は他チャンネルから
    予測しにくいので残りやすい。
    """
    n_ch, n = channels.shape
    out = channels.copy()
    half = taps // 2
    for k in range(n_ch):
        # 参照行列を構築(他チャンネルの遅延版)
        cols = []
        for j in range(n_ch):
            if j == k:
                continue
            for d in range(-half, half):
                cols.append(np.roll(channels[j], d))
        X = np.stack(cols, axis=1)  # (n, taps*(C-1))
        y = channels[k]
        # 最小二乗 h = argmin ||y - X h||  (漏れ込み成分の推定)
        h, *_ = np.linalg.lstsq(X, y, rcond=None)
        leak_est = X @ h
        out[k] = y - leak_est
    return out
