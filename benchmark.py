"""Step 4 (RQ1): which smoother detects churners EARLIEST at a matched false-positive rate?

Signal per event: drop_t = baseline - smoothed_t, baseline = mean engagement over days 90..61 before expiry.
Alarm (per smoother, per group, per alpha):
  * the alarm day is the EARLIEST day k in [K..1] (k = days before expiry) with drop_k > tau;
  * tau is set on RETAINED events of the group so that a fraction alpha of them ever alarm in the window (matched FPR);
  * lead time = k at the first alarm; churners that never alarm are undetected (reported, not imputed).
Metrics: TPR at lead >= 7/14/21/28 days, overall detection rate, median lead among detected, achieved FPR.
Uncertainty: customer-level bootstrap (a customer who appears in both cohorts is resampled as one unit); tau is
re-calibrated in every resample, churners are never used to set tau.
Smoothers must be causal; the leaky reference is reported separately and marked.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .smoothers import Smoother, is_causal

K_MAX = 60
LEADS = (7, 14, 21, 28)
GROUPS = ("all", "auto_renew", "manual_renew", "silent_only")
MIN_BASELINE = 0.5  # log(1+minutes): ~0.65 min/day average listening in the reference window


def baseline(X: np.ndarray, L: int) -> np.ndarray:
    return np.asarray(X[:, L - 90: L - 60], dtype=np.float32).mean(axis=1)


def drop_matrix(S: np.ndarray, B: np.ndarray, L: int, K: int = K_MAX) -> np.ndarray:
    """(events x K) drop signal for days k = K..1 (column 0 is k=K, the earliest day)."""
    W = S[:, L - K:]
    if np.isnan(W).any():
        raise ValueError("NaN inside the alarm window (smoother window longer than the lead-in?)")
    return B[:, None] - W


def first_alarm_lead(D: np.ndarray, tau: float) -> np.ndarray:
    """Days before expiry of the first alarm (0 = never alarmed). D columns run k=K..1."""
    K = D.shape[1]
    m = D > tau
    det = m.any(axis=1)
    return np.where(det, K - m.argmax(axis=1), 0)


def calibrate_tau(M_retained: np.ndarray, alpha: float, w: Optional[np.ndarray] = None) -> float:
    """(1-alpha) weighted quantile of the per-event maximum drop among retained events."""
    order = np.argsort(M_retained)
    ms = M_retained[order]
    ws = np.ones_like(ms, dtype=np.float64) if w is None else np.asarray(w, dtype=np.float64)[order]
    cum = np.cumsum(ws)
    if cum[-1] <= 0:
        return np.nan
    return float(ms[min(np.searchsorted(cum, (1 - alpha) * cum[-1], side="left"), len(ms) - 1)])


def _wmedian(x: np.ndarray, w: np.ndarray) -> float:
    if len(x) == 0 or w.sum() <= 0:
        return np.nan
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(cw, 0.5 * cw[-1]), len(x) - 1)])


def _metrics(M, Dc, tau, wR, wC, MR):
    lead = first_alarm_lead(Dc, tau)
    det = lead > 0
    tot = wC.sum()
    out = {f"tpr_L{L}": (wC[det & (lead >= L)].sum() / tot if tot > 0 else np.nan) for L in LEADS}
    out["tpr_any"] = wC[det].sum() / tot if tot > 0 else np.nan
    out["median_lead_detected"] = _wmedian(lead[det].astype(float), wC[det])
    out["fpr_achieved"] = wR[MR > tau].sum() / wR.sum() if wR.sum() > 0 else np.nan
    return out


def run_benchmark(X, ev: pd.DataFrame, has_log: np.ndarray, smoothers: List[Smoother], L: int,
                  alphas=(0.05, 0.10), B: int = 200, seed: int = 0, min_baseline: float = MIN_BASELINE,
                  check_causal: bool = True) -> Dict[str, pd.DataFrame]:
    """Returns {'results': tidy table, 'counts': analysable counts}."""
    base_all = baseline(X, L)
    ok = np.asarray(has_log, bool) & (base_all >= min_baseline)
    sel = np.where(ok)[0]
    y = ev["is_churn"].to_numpy()[sel]
    auto = ev["is_auto_renew"].to_numpy()[sel]
    silent = ev["silent"].to_numpy()[sel]
    cust = pd.factorize(ev["msno"].to_numpy()[sel])[0]
    ncust = int(cust.max()) + 1
    rng = np.random.default_rng(seed)
    W = np.empty((B + 1, len(sel)), dtype=np.uint8)
    W[0] = 1
    for b in range(B):
        W[b + 1] = np.minimum(rng.multinomial(ncust, np.full(ncust, 1.0 / ncust))[cust], 255)

    masks = {"all": np.ones(len(sel), bool), "auto_renew": auto == 1, "manual_renew": auto == 0}
    counts = pd.DataFrame({"events_total": [len(ev)], "analysable": [len(sel)],
                           "analysable_churners": [int((y == 1).sum())], "analysable_silent_churners": [int(((y == 1) & (silent == 1)).sum())],
                           "no_logs_in_window": [int((~np.asarray(has_log, bool)).sum())],
                           "low_baseline": [int((np.asarray(has_log, bool) & (base_all < min_baseline)).sum())]})
    rows = []
    Xs = X[sel]
    Bsel = base_all[sel]
    for sm in smoothers:
        if check_causal and sm.causal and not is_causal(sm.fn, Xs):
            raise AssertionError(f"LEAKAGE: smoother '{sm.name}' is declared causal but changes when future days change")
        D = drop_matrix(sm.fn(Xs), Bsel, L)
        M = D.max(axis=1)
        for g in GROUPS:
            if g == "silent_only":
                R_idx, C_idx = np.where(y == 0)[0], np.where((y == 1) & (silent == 1))[0]
            else:
                R_idx, C_idx = np.where(masks[g] & (y == 0))[0], np.where(masks[g] & (y == 1))[0]
            if len(C_idx) < 20 or len(R_idx) < 100:
                continue
            o = np.argsort(M[R_idx])
            R_sorted = R_idx[o]
            MR = M[R_sorted]
            Dc = D[C_idx]
            for alpha in alphas:
                res = []
                for b in range(B + 1):
                    wR = W[b][R_sorted].astype(np.float64)
                    wC = W[b][C_idx].astype(np.float64)
                    cum = np.cumsum(wR)
                    tau = float(MR[min(np.searchsorted(cum, (1 - alpha) * cum[-1], side="left"), len(MR) - 1)]) if cum[-1] > 0 else np.nan
                    res.append(_metrics(M, Dc, tau, wR, wC, MR))
                est = res[0]
                for metric in est:
                    arr = np.array([r[metric] for r in res[1:]], dtype=float)
                    lo, hi = (np.nanpercentile(arr, [2.5, 97.5]) if B > 0 and np.isfinite(arr).any() else (np.nan, np.nan))
                    rows.append({"method": sm.name, "family": sm.family, "param": sm.param, "causal": sm.causal,
                                 "group": g, "alpha": alpha, "metric": metric, "estimate": est[metric], "lo": lo, "hi": hi,
                                 "n_churn": len(C_idx), "n_retained": len(R_idx)})
    return {"results": pd.DataFrame(rows), "counts": counts}


def best_per_family(results: pd.DataFrame, group="all", alpha=0.10, metric="tpr_L14") -> pd.DataFrame:
    d = results[(results["group"] == group) & (results["alpha"] == alpha) & (results["metric"] == metric) & results["causal"]]
    d = d[d["family"] != "raw"].sort_values("estimate", ascending=False).drop_duplicates("family")
    return d[["family", "method", "estimate", "lo", "hi"]].reset_index(drop=True)


def alarm_leads_for(sm: Smoother, X, ev, has_log, L, alpha=0.10, min_baseline=MIN_BASELINE) -> np.ndarray:
    """Per-event first-alarm lead (days) for one smoother, tau calibrated on all analysable retained events.
    Events that are not analysable get -1."""
    base_all = baseline(X, L)
    ok = np.asarray(has_log, bool) & (base_all >= min_baseline)
    sel = np.where(ok)[0]
    D = drop_matrix(sm.fn(X[sel]), base_all[sel], L)
    y = ev["is_churn"].to_numpy()[sel]
    tau = calibrate_tau(D.max(axis=1)[y == 0], alpha)
    out = np.full(len(ev), -1, dtype=int)
    out[sel] = first_alarm_lead(D, tau)
    return out
