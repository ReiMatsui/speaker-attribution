"""ゲートのストリーミング化（リアルタイム逐次処理）.

バッチ版 pipeline.run は全体を見て判定するが、リアルタイムでは無理。
StreamingGate は音声をチャンク（hop）単位で受け取り、各チャンネルについて
「今このhopは本人発話か漏れか」を、わずかな先読み(lookahead)だけで逐次判定する。

判定はバッチ版と同じ物理（チャンネル間相関 resolve_owner）を、
ローカルな窓 [t - left_context, t + lookahead] に対して行う。
→ 1語が確定するまでの遅延 ≒ lookahead + 計算時間。計算は軽い。

使い方（概念）:
    g = StreamingGate(n_ch, sr)
    for chunk in stream:            # chunk: (n_ch, hop) の音声
        out = g.push(chunk)         # out: 直近で確定した (t0,t1, own[bool per ch]) のリスト
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import vad as vadmod
from .crosstalk import resolve_owner


@dataclass
class StreamingGate:
    n_ch: int
    sr: int
    hop_s: float = 0.25
    lookahead_s: float = 0.25
    left_context_s: float = 0.75
    active_ratio: float = 0.05
    corr_thresh: float = 3.0

    rel_thresh_db: float = 15.0
    floor_history_s: float = 4.0                          # ノイズ床推定に使う履歴長

    buf: np.ndarray = field(default=None, repr=False)   # (n_ch, samples) 直近の音声
    t0: float = 0.0                                       # buf 先頭の絶対時刻(秒)
    emitted_until: float = 0.0                            # ここまで判定確定した絶対時刻
    _ehist: list = field(default=None, repr=False)       # 各chのフレームdB履歴

    def __post_init__(self):
        self.buf = np.zeros((self.n_ch, 0), dtype=np.float32)
        self._ehist = [list() for _ in range(self.n_ch)]

    def push(self, chunk):
        """chunk: (n_ch, hop_samples)。確定したhopの判定リストを返す.

        返り値: list of dict {start, end, own(np.bool[n_ch])}
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        self.buf = np.concatenate([self.buf, chunk], axis=1)
        hop = int(self.hop_s * self.sr)
        look = int(self.lookahead_s * self.sr)
        leftc = int(self.left_context_s * self.sr)

        results = []
        # buf の絶対時刻範囲は [t0, t0 + buf_len/sr]
        # 次に確定させるhopの開始(絶対秒)
        while True:
            hop_start_t = self.emitted_until
            hop_start_idx = int(round((hop_start_t - self.t0) * self.sr))
            need_end_idx = hop_start_idx + hop + look      # 判定に必要な末尾
            if need_end_idx > self.buf.shape[1]:
                break  # まだ先読み分が溜まっていない
            a = max(0, hop_start_idx - leftc)
            b = hop_start_idx + hop + look
            window = self.buf[:, a:b]
            own = self._decide(window, hop_start_idx - a, hop)
            results.append({"start": hop_start_t, "end": hop_start_t + self.hop_s, "own": own})
            self.emitted_until += self.hop_s

        # メモリ節約: 古い部分を捨てる
        keep_from_t = self.emitted_until - self.left_context_s
        drop_idx = int((keep_from_t - self.t0) * self.sr)
        if drop_idx > self.sr:  # 1秒以上たまったら切り詰め
            self.buf = self.buf[:, drop_idx:]
            self.t0 += drop_idx / self.sr
        return results

    def _decide(self, window, hop_off, hop):
        """window 内の [hop_off, hop_off+hop] が各chで本人発話か判定.

        VADは「走査中に保持しているノイズ床」を基準に判定する（ローカル窓で
        床を推定すると、窓全体が発話のとき壊れるため）。
        """
        n_ch = self.n_ch
        own = np.zeros(n_ch, dtype=bool)
        active = np.zeros(n_ch, dtype=bool)
        max_hist = int(self.floor_history_s * 100)  # フレーム数(10ms刻み)
        for ch in range(n_ch):
            edb, _ = vadmod.frame_energy_db(window[ch], self.sr)
            f0 = int((hop_off / self.sr) * 100)
            f1 = int(((hop_off + hop) / self.sr) * 100)
            hop_edb = edb[f0:f1]
            # 履歴を更新してノイズ床(下位10%)を推定
            self._ehist[ch].extend(hop_edb.tolist())
            if len(self._ehist[ch]) > max_hist:
                self._ehist[ch] = self._ehist[ch][-max_hist:]
            floor = np.percentile(self._ehist[ch], 10)
            if len(hop_edb):
                frac = (hop_edb > floor + self.rel_thresh_db).mean()
                active[ch] = frac > self.active_ratio
        if not active.any():
            return own
        res = resolve_owner(window, self.sr, corr_thresh=self.corr_thresh)
        is_leak = res["is_leak"]
        for ch in range(n_ch):
            own[ch] = active[ch] and not is_leak[ch]
        return own

    @property
    def latency_s(self):
        """1hopが確定するまでの本質的な遅延（先読み分。計算時間は別途加算）。"""
        return self.hop_s + self.lookahead_s
