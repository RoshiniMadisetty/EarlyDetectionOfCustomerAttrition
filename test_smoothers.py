import numpy as np
import pytest

from silentchurn import smoothers as sm


def rand_X(n=60, L=120, seed=0):
    return np.random.default_rng(seed).gamma(2.0, 1.0, size=(n, L)).astype(np.float32)


@pytest.mark.parametrize("taps", [sm.sma_taps(7), sm.lowess_taps(14), sm.lowess_taps(28),
                                  sm.spline_taps(28, 1), sm.spline_taps(28, 10), sm.spline_taps(28, 100)])
def test_taps_sum_to_one_so_constants_are_preserved(taps):
    assert taps.sum() == pytest.approx(1.0, abs=1e-6)
    X = np.full((3, 60), 4.2, dtype=np.float32)
    Y = sm.fir_apply(X, taps)
    assert np.allclose(Y[:, len(taps):], 4.2, atol=1e-4)


@pytest.mark.parametrize("taps", [sm.lowess_taps(14), sm.spline_taps(28, 10)])
def test_local_linear_and_spline_endpoint_reproduce_a_straight_line(taps):
    t = np.arange(120, dtype=np.float32)
    X = np.vstack([0.5 * t + 3, -0.2 * t + 40]).astype(np.float32)
    Y = sm.fir_apply(X, taps)
    assert np.allclose(Y[:, 40:], X[:, 40:], atol=1e-3)  # no lag on a linear trend (unlike a moving average)


def test_sma_matches_numpy_and_lags_a_linear_trend():
    X = rand_X()
    Y = sm.fir_apply(X, sm.sma_taps(7))
    ref = np.array([X[:, t - 6:t + 1].mean(axis=1) for t in range(6, 120)]).T
    assert np.allclose(Y[:, 6:], ref, atol=1e-5) and np.isnan(Y[:, :6]).all()
    t = np.arange(120, dtype=np.float32)[None, :]
    assert np.allclose(sm.fir_apply(t, sm.sma_taps(7))[0, 20:], t[0, 20:] - 3.0, atol=1e-4)  # lag of (w-1)/2


def test_ema_recursion():
    X = np.array([[1, 2, 3, 4]], dtype=np.float32)
    Y = sm.ema(X, 0.5)
    assert np.allclose(Y[0], [1, 1.5, 2.25, 3.125])


def test_lowess_matches_direct_weighted_regression():
    w = 14
    X = rand_X(5, 60, seed=3)
    Y = sm.fir_apply(X, sm.lowess_taps(w))
    t = 40
    d = np.arange(w)
    wt = (1 - (d / w) ** 3) ** 3
    for i in range(5):
        y = X[i, t - d]
        coef = np.polyfit(-d, y, 1, w=np.sqrt(wt))
        assert Y[i, t] == pytest.approx(coef[1], abs=1e-3)


def test_spline_filter_matches_direct_scipy_fit():
    from scipy.interpolate import make_smoothing_spline
    X = rand_X(4, 60, seed=5)
    Y = sm.fir_apply(X, sm.spline_taps(28, 10))
    t = 50
    xs = np.arange(28, dtype=float)
    for i in range(4):
        seg = X[i, t - 27:t + 1].astype(float)
        assert Y[i, t] == pytest.approx(float(make_smoothing_spline(xs, seg, lam=10)(xs[-1])), abs=1e-3)


def test_spline_smoothness_is_ordered_by_lambda():
    e = [sm.spline_edf(28, l) for l in (1, 10, 100)]
    assert e[0] > e[1] > e[2] > 2


def test_causality_test_passes_for_all_valid_smoothers_and_fails_for_the_leaky_one():
    X = rand_X()
    for s in sm.make_smoothers(include_leaky=True):
        ok = sm.is_causal(s.fn, X)
        assert ok == s.causal, f"{s.name}: is_causal={ok}, declared causal={s.causal}"


def test_registry_names_unique_and_leaky_flagged():
    sms = sm.make_smoothers()
    names = [s.name for s in sms]
    assert len(set(names)) == len(names)
    assert [s.name for s in sms if not s.causal] == ["SMA-centered w=7 (LEAKY)"]


def test_cusum_is_causal_and_flags_a_sustained_drop():
    L = 120
    rng = np.random.default_rng(0)
    X = np.maximum(3 + rng.normal(0, 0.8, size=(400, L)), 0).astype(np.float32)
    X[:200, L - 20:] -= 2.5                      # 200 events lose most engagement in the last 20 days
    X = np.maximum(X, 0)
    S, first = sm.cusum_down(X, L)
    assert (first[:200] > 0).mean() > 0.9 and (first[200:] > 0).mean() < 0.2
    det = first[:200][first[:200] > 0]
    assert np.median(det) <= 20 and (det > 20).mean() < 0.15   # alarms cluster after the drop starts; early ones are false alarms
    assert sm.is_causal(lambda Z: sm.cusum_down(Z, L)[0], X)
