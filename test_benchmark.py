import numpy as np
import pandas as pd
import pytest

from silentchurn import benchmark as bm
from silentchurn import smoothers as sm

L = 120


def planted(n_ret=1500, n_ch=300, seed=1, noise=1.0, decay_days=25):
    """Retained: stationary noisy series. Churners: gradual decay over the last `decay_days` days."""
    rng = np.random.default_rng(seed)
    n = n_ret + n_ch
    level = rng.uniform(2.0, 4.0, size=(n, 1))
    X = level + noise * rng.normal(size=(n, L))
    X[X < 0] = 0
    y = np.r_[np.zeros(n_ret, int), np.ones(n_ch, int)]
    ramp = np.clip((np.arange(L) - (L - decay_days)) / decay_days, 0, 1)
    X[n_ret:] = np.maximum(X[n_ret:] - level[n_ret:] * 0.9 * ramp[None, :], 0)
    auto = rng.integers(0, 2, size=n)
    ev = pd.DataFrame({"msno": [f"c{i}" for i in range(n)], "is_churn": y, "is_auto_renew": auto,
                       "silent": ((y == 1) & (rng.random(n) < 0.7)).astype(int)})
    return X.astype(np.float32), ev, np.ones(n, bool)


def test_first_alarm_lead_definition():
    K = 60
    D = np.zeros((3, K), dtype=np.float32)
    D[0, K - 17:] = 5.0          # crosses at k=17 -> lead 17
    D[1, :] = 5.0                # alarms from the first day -> lead 60
    D[2, :] = 0.0                # never
    assert bm.first_alarm_lead(D, 1.0).tolist() == [17, 60, 0]


def test_calibrate_tau_gives_requested_fpr():
    M = np.random.default_rng(0).normal(size=20000)
    tau = bm.calibrate_tau(M, 0.10)
    assert (M > tau).mean() == pytest.approx(0.10, abs=0.002)
    w = np.where(M > 1, 3, 1)  # weights change the quantile
    assert bm.calibrate_tau(M, 0.10, w) > tau


def test_matched_fpr_and_detection_of_planted_decay():
    X, ev, ok = planted()
    out = bm.run_benchmark(X, ev, ok, sm.make_smoothers(include_leaky=False), L, alphas=(0.10,), B=20, seed=3)
    r = out["results"]
    f = r[(r.group == "all") & (r.metric == "fpr_achieved")]
    assert (f["estimate"].between(0.08, 0.11)).all()               # FPR matched for every method
    tpr = r[(r.group == "all") & (r.metric == "tpr_any")].set_index("method")["estimate"]
    assert tpr["EMA a=0.15"] > 0.5                                  # planted decay is detectable
    assert tpr["SMA w=7"] > tpr["raw"]                              # smoothing beats raw on noisy data at matched FPR
    ci = r[(r.metric == "tpr_L14") & (r.group == "all")]
    assert (ci["lo"] <= ci["hi"]).all()


def test_all_groups_present_and_counts():
    X, ev, ok = planted()
    out = bm.run_benchmark(X, ev, ok, [sm.make_smoothers(False)[5]], L, alphas=(0.05, 0.10), B=10)
    r = out["results"]
    assert set(r["group"]) == set(bm.GROUPS)
    c = out["counts"].iloc[0]
    assert c["analysable"] == len(ev) and c["analysable_churners"] == 300


def test_low_baseline_and_no_log_events_are_excluded_and_counted():
    X, ev, ok = planted()
    X[:50] = 0.0                 # inactive baseline
    ok = ok.copy(); ok[50:80] = False
    out = bm.run_benchmark(X, ev, ok, [sm.make_smoothers(False)[1]], L, alphas=(0.1,), B=5)
    c = out["counts"].iloc[0]
    assert c["no_logs_in_window"] == 30 and c["low_baseline"] == 50 and c["analysable"] == len(ev) - 80


def test_declared_causal_but_leaky_smoother_is_rejected():
    X, ev, ok = planted(n_ret=300, n_ch=60)
    bad = sm.Smoother("cheat", "SMA", "w=7", lambda Z: sm.sma_centered(Z, 7), causal=True)
    with pytest.raises(AssertionError, match="LEAKAGE"):
        bm.run_benchmark(X, ev, ok, [bad], L, alphas=(0.1,), B=2)


def test_leaky_reference_is_allowed_only_when_flagged():
    X, ev, ok = planted(n_ret=400, n_ch=80)
    out = bm.run_benchmark(X, ev, ok, [sm.make_smoothers()[-1]], L, alphas=(0.1,), B=2)
    assert (out["results"]["causal"] == False).all() and len(out["results"]) > 0


def test_bootstrap_resamples_customers_not_events():
    # two events of the same customer must always receive the same weight
    X, ev, ok = planted(n_ret=200, n_ch=40)
    ev = ev.copy(); ev.loc[1, "msno"] = ev.loc[0, "msno"]
    cust = pd.factorize(ev["msno"])[0]
    assert cust[0] == cust[1]
    out = bm.run_benchmark(X, ev, ok, [sm.make_smoothers(False)[1]], L, alphas=(0.1,), B=3)
    assert len(out["results"]) > 0


def test_best_per_family_and_alarm_leads():
    X, ev, ok = planted()
    sms = sm.make_smoothers(include_leaky=False)
    out = bm.run_benchmark(X, ev, ok, sms, L, alphas=(0.10,), B=5)
    best = bm.best_per_family(out["results"])
    assert set(best["family"]) == {"SMA", "EMA", "LOWESS", "Spline"}
    leads = bm.alarm_leads_for(sms[5], X, ev, ok, L)
    assert leads.shape == (len(ev),) and (leads[ev.is_churn.to_numpy() == 0] > 0).mean() == pytest.approx(0.10, abs=0.02)
