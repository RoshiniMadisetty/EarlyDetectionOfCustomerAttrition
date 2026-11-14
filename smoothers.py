"""Causal smoothers for the daily engagement matrix X (rows = events, columns = days, last column = day before expiry).

Every method here is an END-POINT estimate: the value at day t uses only days <= t.
  * SMA(w)          trailing simple moving average
  * EMA(alpha)      exponential moving average (recursive)
  * LOWESS(w)       local-linear regression with tricube weights on the trailing w days, evaluated at day t
                    (one-sided LOESS, degree 1, NO robustness iterations - not full Cleveland LOWESS)
  * SPLINE(w, lam)  cubic smoothing spline fitted to the trailing w days (scipy make_smoothing_spline), evaluated at the
                    last day. For fixed (w, lam) this is a fixed linear filter, so it is applied as a convolution.
  * SMA-centered    LEAKY REFERENCE ONLY: uses future days. It exists to show how much a non-causal smoother flatters
                    detection; it is never part of a valid comparison.
``is_causal`` is the automated leakage test: scrambling every day after t must not change any output at or before t.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np


def fir_apply(X: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """y_t = sum_i taps[i] * x_{t-i}; the first len(taps)-1 columns (incomplete window) are NaN."""
    w = len(taps)
    X = np.asarray(X, dtype=np.float32)
    Y = np.zeros_like(X)
    for i, c in enumerate(taps):
        if c == 0:
            continue
        if i == 0:
            Y += np.float32(c) * X
        else:
            Y[:, i:] += np.float32(c) * X[:, :-i]
    Y[:, : w - 1] = np.nan
    return Y


def sma_taps(w: int) -> np.ndarray:
    return np.full(w, 1.0 / w)


def lowess_taps(w: int) -> np.ndarray:
    """Local-linear, tricube-weighted end-point filter over the last w days (lag d = 0..w-1)."""
    d = np.arange(w, dtype=float)
    wt = (1 - (d / w) ** 3) ** 3
    x = -d
    s0, s1, s2 = wt.sum(), (wt * x).sum(), (wt * x * x).sum()
    return wt * (s2 - s1 * x) / (s0 * s2 - s1 * s1)


def spline_matrix(w: int, lam: float) -> np.ndarray:
    from scipy.interpolate import make_smoothing_spline
    x = np.arange(w, dtype=float)
    return np.array([make_smoothing_spline(x, np.eye(w)[j], lam=lam)(x) for j in range(w)]).T


def spline_taps(w: int, lam: float) -> np.ndarray:
    S = spline_matrix(w, lam)
    return S[w - 1, ::-1].copy()  # taps[i] = weight on the value i days before the current day


def spline_edf(w: int, lam: float) -> float:
    return float(np.trace(spline_matrix(w, lam)))


def ema(X: np.ndarray, alpha: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    Y = np.empty_like(X)
    Y[:, 0] = X[:, 0]
    a = np.float32(alpha)
    for t in range(1, X.shape[1]):
        Y[:, t] = a * X[:, t] + (1 - a) * Y[:, t - 1]
    return Y


def sma_centered(X: np.ndarray, w: int) -> np.ndarray:
    """LEAKY: centred moving average (uses future days)."""
    X = np.asarray(X, dtype=np.float32)
    half = w // 2
    c = np.cumsum(np.pad(X, ((0, 0), (1, 0))), axis=1)
    n = X.shape[1]
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + (w - half), 0, n)
    return (c[:, hi] - c[:, lo]) / (hi - lo).astype(np.float32)


@dataclass
class Smoother:
    name: str
    family: str
    param: str
    fn: Callable[[np.ndarray], np.ndarray]
    causal: bool = True


def make_smoothers(include_leaky: bool = True) -> List[Smoother]:
    out = [Smoother("raw", "raw", "none", lambda X: np.asarray(X, dtype=np.float32))]
    for w in (3, 7, 14, 28):
        t = sma_taps(w)
        out.append(Smoother(f"SMA w={w}", "SMA", f"w={w}", lambda X, t=t: fir_apply(X, t)))
    for a in (0.3, 0.15, 0.07, 0.035):
        out.append(Smoother(f"EMA a={a}", "EMA", f"alpha={a}", lambda X, a=a: ema(X, a)))
    for w in (7, 14, 28):
        t = lowess_taps(w)
        out.append(Smoother(f"LOWESS w={w}", "LOWESS", f"w={w}", lambda X, t=t: fir_apply(X, t)))
    for lam in (1, 10, 100):
        t = spline_taps(28, lam)
        out.append(Smoother(f"Spline w=28 lam={lam}", "Spline", f"lam={lam}", lambda X, t=t: fir_apply(X, t)))
    if include_leaky:
        out.append(Smoother("SMA-centered w=7 (LEAKY)", "SMA-centered", "w=7", lambda X: sma_centered(X, 7), causal=False))
    return out


def is_causal(fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, n_cuts: int = 5, seed: int = 0, atol: float = 1e-5) -> bool:
    """Automated leakage test: outputs at days <= cut must not change when every later day is scrambled."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)[:200]
    base = fn(X)
    for cut in rng.integers(X.shape[1] // 3, X.shape[1] - 2, size=n_cuts):
        X2 = X.copy()
        X2[:, cut + 1:] = rng.uniform(0, 10, size=X2[:, cut + 1:].shape).astype(np.float32)
        Y2 = fn(X2)
        if not np.allclose(base[:, : cut + 1], Y2[:, : cut + 1], atol=atol, equal_nan=True):
            return False
    return True


def cusum_down(X: np.ndarray, L: int, ref_days=(90, 61), alarm_days: int = 60, k: float = 0.5, h: float = 5.0):
    """One-sided (downward) CUSUM change-point detector, causal.

    Reference mean/sd come from days ref_days[0]..ref_days[1] before expiry; the statistic starts at 0 at day
    ``alarm_days`` before expiry:  S_t = max(0, S_{t-1} - z_t - k)  with z the standardised daily value.
    Returns (S, first_alarm_k): S has shape (n, L) (zeros before the start), first_alarm_k = days before expiry of the
    first S > h (0 = no alarm). Thresholds k=0.5, h=5 are the textbook defaults, not tuned on churn labels.
    """
    X = np.asarray(X, dtype=np.float32)
    ref = X[:, L - ref_days[0]: L - ref_days[1] + 1]
    mu, sd = ref.mean(axis=1), np.maximum(ref.std(axis=1), 0.5)
    S = np.zeros_like(X)
    prev = np.zeros(X.shape[0], dtype=np.float32)
    for j in range(L - alarm_days, L):
        z = (X[:, j] - mu) / sd
        prev = np.maximum(0, prev - z - np.float32(k))
        S[:, j] = prev
    m = S[:, L - alarm_days:] > h
    first = np.where(m.any(axis=1), alarm_days - m.argmax(axis=1), 0)
    return S, first
