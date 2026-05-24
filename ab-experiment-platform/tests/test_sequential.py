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


def test_obf_bounds_decreasing_and_final_near_196():
    from src.abtest.sequential import obrien_fleming_bounds
    b = obrien_fleming_bounds(n_looks=5, alpha=0.05)
    assert len(b) == 5
    assert all(b[i] > b[i + 1] for i in range(len(b) - 1))
    assert 1.96 <= b[-1] <= 2.5
