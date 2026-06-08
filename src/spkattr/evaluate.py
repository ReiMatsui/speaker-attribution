"""帰属結果の評価指標.

- attribution_accuracy: 各窓で「持ち主本人の発話」を正しく拾えたか
- crosstalk_rejection : 漏れ込み(他人の声)をどれだけ正しく棄却できたか
"""
from __future__ import annotations

import numpy as np


def labels_to_windows(labels: np.ndarray, sr: int, win_ms: float, hop_ms: float):
    """サンプル単位の正解VADを、パイプラインと同じ時間窓の bool に変換."""
    n_ch, n = labels.shape
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    rows = []
    for s0 in range(0, max(1, n - win + 1), hop):
        s1 = min(s0 + win, n)
        rows.append(labels[:, s0:s1].mean(axis=1) > 0.3)
    return np.array(rows)  # (n_win, n_ch)


def evaluate(segments, truth_windows, owner_names, sr, win_ms=500.0, hop_ms=250.0):
    """segments(パイプライン出力) を truth_windows と突き合わせ.

    truth_windows[w, ch] = その窓でチャンネル ch の持ち主が実際に話していたか。
    """
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    n_win = len(truth_windows)
    n_ch = len(owner_names)

    # パイプライン出力を (n_win, n_ch) の「本人発話と判定」行列へ
    pred = np.zeros((n_win, n_ch), dtype=bool)
    for seg in segments:
        w = int(round(seg.start * sr / hop))
        if 0 <= w < n_win and seg.is_own_voice:
            pred[w, seg.channel] = True

    truth = truth_windows[:n_win]
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())   # 漏れ込み等を本人と誤判定
    fn = int((~pred & truth).sum())   # 本人発話を取りこぼし
    tn = int((~pred & ~truth).sum())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    crosstalk_rejection = tn / (tn + fp + 1e-9)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "crosstalk_rejection": crosstalk_rejection,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
