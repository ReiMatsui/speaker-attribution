import numpy as np

from spkattr.crosstalk import gcc_phat, resolve_owner
from spkattr.synth import SynthConfig, synthesize
from spkattr.vad import energy_vad


def test_gcc_phat_detects_delay():
    sr = 16000
    rng = np.random.default_rng(0)
    x = rng.standard_normal(8000)
    delay = 16  # samples
    y = np.concatenate([np.zeros(delay), x])[: len(x)]
    tau, peak = gcc_phat(x, y, sr, max_tau=0.01)
    assert peak > 3.0  # 同一音源なので相関が鋭い
    # x が先行(y が遅延)→ gcc_phat(x,y) の tau は -delay/sr 付近
    assert abs(tau + delay / sr) < 1e-3


def test_resolve_owner_picks_loud_clean_channel():
    sr = 16000
    rng = np.random.default_rng(1)
    src = rng.standard_normal(8000)
    # ch0 = 本人(強), ch1 = 漏れ込み(遅延+減衰)
    delay = 24
    leak = np.concatenate([np.zeros(delay), src])[: len(src)] * 0.25
    window = np.stack([src, leak])
    res = resolve_owner(window, sr)
    assert res["owner"] == 0
    assert res["is_leak"][1]  # ch1 は漏れ込みと判定


def test_synthesize_shapes():
    sr = 16000
    sources = [np.random.randn(sr).astype(float) for _ in range(3)]
    ch, labels = synthesize(sources, SynthConfig(sr=sr))
    assert ch.shape == (3, sr)
    assert labels.shape == (3, sr)


def test_energy_vad_runs():
    sr = 16000
    x = np.zeros(sr)
    x[4000:8000] = np.random.randn(4000)
    a = energy_vad(x, sr)
    assert a.any()
