import numpy as np
import pandas as pd
import pytest

from silentchurn import model as md

L = 120


def make(n=6000, seed=2, shift=0.0):
    rng = np.random.default_rng(seed)
    level = rng.uniform(1.5, 4.0, size=(n, 1))
    X = np.maximum(level + rng.normal(0, 1.2, size=(n, L)), 0)
    X[rng.random((n, L)) < 0.15] = 0
    base_p = 0.08
    y = (rng.random(n) < base_p + shift).astype(int)
    ramp = np.clip((np.arange(L) - (L - 25)) / 25, 0, 1)
    X[y == 1] = np.maximum(X[y == 1] - level[y == 1] * 0.8 * ramp[None, :], 0)
    start = pd.Timestamp("2017-02-01")
    ev = pd.DataFrame({"event_id": range(n), "msno": [f"c{i}" for i in range(n)], "is_churn": y,
                       "is_auto_renew": rng.integers(0, 2, n), "plan_days": 30,
                       "expiry_date": start + pd.to_timedelta(rng.integers(0, 59, n), unit="D"),
                       "silent": y})
    return X.astype(np.float32), ev, np.ones(n, bool)


def test_features_do_not_use_post_cutoff_days():
    X, ev, ok = make(500)
    for h in (1, 7, 14):
        assert md.check_feature_leakage(X, ev, L, h)


def test_leakage_test_detects_a_leaky_feature(monkeypatch):
    X, ev, ok = make(500)
    real = md.features_from_series
    def leaky(X, ev, L, h=7):
        f = real(X, ev, L, h)
        f["drop7"] = f["drop7"] + X[:, -1]          # peeks at the day before expiry
        return f
    monkeypatch.setattr(md, "features_from_series", leaky)
    assert not md.check_feature_leakage(X, ev, L, 7)


def test_psi_basics():
    r = np.random.default_rng(0)
    a = r.normal(size=5000)
    assert md.psi(a, r.normal(size=5000)) < 0.05
    assert md.psi(a, r.normal(1.5, 1, size=5000)) > md.PSI_ALERT
    b = r.integers(0, 2, 3000)
    assert md.psi(b, b) == pytest.approx(0, abs=1e-9)


def test_reliability_and_ece_on_calibrated_scores():
    r = np.random.default_rng(1)
    p = r.uniform(0.01, 0.4, 40000)
    y = (r.random(40000) < p).astype(int)
    t = md.reliability_table(y, p)
    assert len(t) == 10 and (t["obs_lo"] <= t["observed"]).all() and (t["observed"] <= t["obs_hi"]).all()
    assert md.ece(y, p) < 0.02
    assert md.ece(y, np.clip(p * 2, 0, 1)) > 0.08          # a miscalibrated score is flagged


def test_run_model_end_to_end():
    X, ev, ok = make()
    out = md.run_model(X, ev, ok, L)
    m = out["metrics"]
    assert m["auc_monitor"] > 0.8 and m["ece_monitor"] < 0.03
    assert out["n_train"] > 1000 and out["n_monitor"] > 1000
    s = out["split_frame"]
    assert set(s["split"]) <= {"train", "monitor", "excluded"}
    assert s.loc[s.split == "train", "expiry_date"].max() <= pd.Timestamp("2017-02-14")
    assert s.loc[s.split == "monitor", "expiry_date"].min() > pd.Timestamp("2017-02-14")
    assert set(out["importance"]["feature"]) == set(md.FEATURES)
    assert out["importance"].iloc[0]["feature"] in ("drop7", "drop28", "trend", "zero_frac14", "days_since_listen")
    assert {"week", "feature", "psi"} <= set(out["drift"].columns) and out["drift"]["week"].nunique() >= 4
    assert out["perf_week"]["auc"].notna().sum() >= 3


def test_monitor_set_excludes_customers_seen_in_training():
    X, ev, ok = make(3000)
    ev = ev.copy()
    # make 100 customers appear both early (train) and late (monitor window)
    dup = ev.iloc[:100].copy(); dup["msno"] = ev.iloc[:100]["msno"].to_numpy()
    dup["event_id"] = range(len(ev), len(ev) + 100)
    dup["expiry_date"] = pd.Timestamp("2017-03-20")
    ev.loc[:99, "expiry_date"] = pd.Timestamp("2017-02-05")
    ev2 = pd.concat([ev, dup], ignore_index=True)
    X2 = np.vstack([X, X[:100]])
    out = md.run_model(X2, ev2, np.ones(len(ev2), bool), L)
    s = out["split_frame"]
    shared = set(s.loc[s.split == "train", "msno"]) & set(s.loc[s.split == "monitor", "msno"])
    assert not shared and out["n_excluded_shared_customers"] >= 100


def test_drift_detected_when_engagement_shifts_late():
    X, ev, ok = make(8000, seed=4)
    late = (ev["expiry_date"] >= pd.Timestamp("2017-03-08")).to_numpy()
    X[late] = np.maximum(X[late] - 1.0, 0)                  # engagement level drops for late cohorts
    out = md.run_model(X, ev, ok, L)
    d = out["drift"]
    early = d[(d.week <= 2) & (d.feature == "drop7")]["psi"].max()
    lateps = d[(d.week >= 5) & (d.feature == "drop7")]["psi"].max()
    assert lateps > early and lateps > 0.05


def test_slice_table_flags_small_groups():
    r = np.random.default_rng(0)
    df = pd.DataFrame({"y": r.integers(0, 2, 2000), "score": r.random(2000), "g": r.choice(["a", "b", "tiny"], 2000, p=[.5, .49, .01])})
    t = md.slice_table(df, "g")
    assert set(t["level"]) == {"a", "b", "tiny"} and not t.set_index("level").loc["tiny", "reliable"]
    assert t.set_index("level").loc["a", "auc"] == t.set_index("level").loc["a", "auc"]  # not NaN
