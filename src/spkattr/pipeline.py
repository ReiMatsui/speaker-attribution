"""VAD + チャンネル間相関 + 声紋照合 を統合した話者帰属パイプライン.

入力: マルチチャンネル波形 (n_ch, n_samples), 各チャンネルの持ち主名, enrollment
出力: 時間窓ごとの帰属結果(誰が話したか、漏れ込みを除いたか)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import vad as vadmod
from .crosstalk import resolve_owner
from .embedding import Enrollment


@dataclass
class Segment:
    start: float
    end: float
    channel: int
    owner_name: str
    is_speech: bool        # VADで発話と判定
    is_own_voice: bool     # 漏れ込みでなく持ち主本人と判定
    verify_score: float    # 声紋照合スコア(本人プロファイルとの類似度)


def run(channels: np.ndarray, sr: int, owner_names: list[str],
        enroll: Enrollment | None = None,
        win_ms: float = 500.0, hop_ms: float = 250.0,
        verify_thresh: float = 0.45, active_ratio: float = 0.2) -> list[Segment]:
    """パイプライン本体.

    手順(各時間窓ごと):
      1. VAD: 各チャンネルで発話が鳴っているか
      2. crosstalk: 発話中チャンネル群から漏れ込みを除き、持ち主を特定
      3. embedding: 持ち主と判定された窓を声紋照合で本人確認(enroll があれば)
    """
    n_ch, n = channels.shape
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    segments: list[Segment] = []

    # 1. VAD: チャンネル全体で一度だけ計算する(ノイズ床を全体から推定).
    #    窓ごとに計算するとノイズ床推定が壊れるため。
    vad_hop_ms = 10.0
    vad_frames = []  # 各チャンネルのフレーム単位 bool
    for ch in range(n_ch):
        vad_frames.append(vadmod.energy_vad(channels[ch], sr, hop_ms=vad_hop_ms))
    vad_per_ms = 1.0 / vad_hop_ms  # フレーム/ms

    starts = range(0, max(1, n - win + 1), hop)
    for s0 in starts:
        s1 = min(s0 + win, n)
        wseg = channels[:, s0:s1]

        # この窓に対応するVADフレーム範囲
        f0 = int((s0 / sr) * 1000 * vad_per_ms)
        f1 = int((s1 / sr) * 1000 * vad_per_ms)
        active = np.zeros(n_ch, dtype=bool)
        for ch in range(n_ch):
            seg = vad_frames[ch][f0:f1]
            active[ch] = len(seg) > 0 and seg.mean() > active_ratio

        if not active.any():
            continue

        # 2. crosstalk: 漏れ込み判定(activeなチャンネルだけで)
        res = resolve_owner(wseg, sr)
        is_leak = res["is_leak"]

        # 3. 各 active チャンネルを評価
        for ch in range(n_ch):
            if not active[ch]:
                continue
            own = not is_leak[ch]
            score = float("nan")
            if own and enroll is not None and owner_names[ch] in enroll.profiles:
                ok, score = enroll.verify(wseg[ch], owner_names[ch], verify_thresh)
                own = own and ok
            segments.append(Segment(
                start=s0 / sr, end=s1 / sr, channel=ch,
                owner_name=owner_names[ch], is_speech=True,
                is_own_voice=bool(own), verify_score=score,
            ))
    return segments
