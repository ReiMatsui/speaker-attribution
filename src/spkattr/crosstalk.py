"""チャンネル間相関による漏れ込み(クロストーク)判定.

中心アイデア:
ある音は、発生源に最も近いマイクで「最も早く・最も強く」入る。
他のマイクには、わずかに遅れた減衰コピーとして漏れ込む。
よって各時間窓で、チャンネル間の到達時間差(TDOA)と相互相関の強さを見れば、
「この窓の音は誰のマイクが持ち主か」を声量の個人差に依存せず推定できる。

到達時間差の推定には GCC-PHAT を使う。
"""
from __future__ import annotations

import numpy as np


def gcc_phat(a: np.ndarray, b: np.ndarray, sr: int, max_tau: float | None = None):
    """GCC-PHAT で a と b の到達時間差と相関の鋭さを返す.

    返り値 (tau, peak):
      tau < 0 は a が b より早く到達(a が先行)していることを表す。
      tau > 0 は a が b より遅れていることを表す。
      peak は相関の鋭さ(peak-to-average ratio)。同一音源なら大きく(>3程度)、
      無相関なら 1 付近。声量に依存しないので漏れ込み判定に使える。
    """
    n = 1
    while n < len(a) + len(b):
        n *= 2
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12  # PHAT 重み
    cc = np.fft.irfft(R, n)
    cc = np.concatenate((cc[-(n // 2):], cc[: n // 2 + 1]))

    max_shift = n // 2
    if max_tau is not None:
        max_shift = min(max_shift, int(sr * max_tau))
    center = n // 2
    window = cc[center - max_shift: center + max_shift + 1]
    aw = np.abs(window)
    peak_idx = int(np.argmax(aw))
    shift = peak_idx - max_shift
    tau = shift / sr
    # 相関の鋭さ: ピーク / 平均(peak-to-average ratio)。声量に依存しない。
    peak = float(aw[peak_idx] / (aw.mean() + 1e-12))
    return tau, peak


def resolve_owner(window_per_ch: np.ndarray, sr: int,
                  corr_thresh: float = 3.0, max_tau: float = 0.005) -> dict:
    """1つの時間窓について、各チャンネルが「持ち主の発話か/漏れ込みか」を判定.

    window_per_ch: shape (n_ch, win_len) のマルチチャンネル波形。
    返り値 dict:
      owner       : 持ち主と推定されたチャンネル index (発話なしなら None)
      is_leak     : 各チャンネルが漏れ込みか否かの bool 配列
      energy_db   : 各チャンネルのエネルギー(dB)

    判定ロジック:
      1. エネルギー最大のチャンネルを暫定の持ち主とする。
      2. 他チャンネルとの GCC-PHAT を取り、相関が高く(同一音)かつ
         その音が暫定持ち主に対して「遅れている」(tau>=0方向)なら漏れ込みと判定。
    """
    n_ch = window_per_ch.shape[0]
    energy = (window_per_ch ** 2).mean(axis=1) + 1e-12
    energy_db = 10 * np.log10(energy)

    owner = int(np.argmax(energy))
    is_leak = np.zeros(n_ch, dtype=bool)

    ref = window_per_ch[owner]
    for ch in range(n_ch):
        if ch == owner:
            continue
        tau, peak = gcc_phat(ref, window_per_ch[ch], sr, max_tau=max_tau)
        # 同一音源(相関が高い)で、持ち主refより遅れている => 漏れ込み
        if peak > corr_thresh and tau <= 0:
            is_leak[ch] = True
        elif peak > corr_thresh and tau > 0:
            # ref の方が遅れている => 本来の持ち主は ch かもしれない
            # よりエネルギーが近ければ持ち主を ch に譲る余地を残す
            is_leak[ch] = False

    return {"owner": owner, "is_leak": is_leak, "energy_db": energy_db}
