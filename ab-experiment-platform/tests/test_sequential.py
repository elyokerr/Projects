import numpy as np

from src.abtest.sequential import msprt_stream


def test_msprt_crosses_under_real_effect():
    r = msprt_stream(base_rate=0.20, lift=0.05, n_max=20000, alpha=0.05,
                     tau=0.05, seed=41)
    assert r.crossed is True and r.stop_index is not None


def test_msprt_rarely_crosses_under_null():
    crosses = [msprt_stream(base_rate=0.20, lift=0.0, n_max=20000, alpha=0.05,
                            tau=0.05, seed=s).crossed for s in range(200)]
    assert np.mean(crosses) <= 0.10


def test_naive_peeking_inflates_fpr():
    from src.abtest.sequential import naive_peeking_fpr
    fpr = naive_peeking_fpr(base_rate=0.20, n_max=20000, look_every=500,
                            n_sims=200, alpha=0.05, seed=99)
    assert fpr > 0.15
