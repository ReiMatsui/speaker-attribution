"""チャンネルごとの発話区間検出 (VAD).

外部依存を増やさないため、まずはエネルギーベースの軽量VADを実装する。
フレーム単位のエネルギーを計算し、各チャンネルのノイズフロアに対する
相対しきい値で発話/非発話を判定する。後でSilero VAD等に差し替え可能。
"""
from __future__ import annotations

import numpy as np


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """信号を (フレーム数, frame_len) に切り出す."""
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n = 1 + (len(x) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x: np.ndarray, sr: int, frame_ms: float = 25.0,
                    hop_ms: float = 10.0) -> tuple[np.ndarray, int]:
    """各フレームの対数エネルギー(dB)とhop長を返す."""
    frame_len = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    frames = frame_signal(x, frame_len, hop)
    energy = (frames ** 2).mean(axis=1) + 1e-12
    return 10 * np.log10(energy), hop


def energy_vad(x: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0,
               rel_thresh_db: float = 15.0, min_speech_ms: float = 120.0) -> np.ndarray:
    """エネルギーベースVAD.

    返り値: フレーム単位の bool 配列 (True=発話). hop_ms 間隔.

    rel_thresh_db: ノイズフロア(下位パーセンタイル)からこのdB以上で発話とみなす。
    min_speech_ms: これより短い発話の塊は除去する(チャタリング抑制)。
    """
    edb, hop = frame_energy_db(x, sr, frame_ms, hop_ms)
    noise_floor = np.percentile(edb, 10)
    active = edb > (noise_floor + rel_thresh_db)

    # 短すぎる発話区間を除去
    min_frames = max(1, int(min_speech_ms / hop_ms))
    active = _remove_short_runs(active, min_frames)
    return active


def _remove_short_runs(mask: np.ndarray, min_frames: int) -> np.ndarray:
    out = mask.copy()
    i = 0
    n = len(mask)
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if (j - i) < min_frames:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out
