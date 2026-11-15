"""Step 5 (Unit 4): ONE simple model whose OUTPUT is visualised - not the contribution.

* Features use only days with k >= h before expiry (cutoff = expiry - h days); ``check_feature_leakage`` is the
  automated test (scrambling every later day must not change a single feature).
* Time-honest split: train on events expiring up to ``train_end`` (default 2017-02-14); everything later is the
  monitoring set (out of time), minus customers who also appear in the training set.
* Outputs: scores, reliability table with Wilson CIs, ECE/Brier/AUC, permutation importance, weekly PSI and weekly
  AUC/calibration for drift, and subgroup (bias) slices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .benchmark import baseline, MIN_BASELINE
from .ingest import wilson_ci

FEATURES = ["drop7", "drop28", "trend", "zero_frac14", "days_since_listen", "is_auto_renew", "log_plan_days"]
PSI_WARN, PSI_ALERT = 0.10, 0.25   # conventional rules of thumb, not statistical tests


def features_from_series(X: np.ndarray, ev: pd.DataFrame, L: int, h: int = 7) -> pd.DataFrame:
    end = L - h + 1                       # columns 0..L-h  <=>  k >= h
    A = np.asarray(X[:, :end], dtype=np.float32)
    mb = np.asarray(X[:, L - 90: L - 60], dtype=np.float32).mean(axis=1)
    m7, m28 = A[:, -7:].mean(axis=1), A[:, -28:].mean(axis=1)
    rev = (A[:, -60:] > 0)[:, ::-1]
    f = pd.DataFrame({
        "drop7": mb - m7, "drop28": mb - m28, "trend": m7 - m28,
        "zero_frac14": (A[:, -14:] == 0).mean(axis=1),
        "days_since_listen": np.where(rev.any(axis=1), rev.argmax(axis=1), 60).astype(np.float32),
        "is_auto_renew": ev["is_auto_renew"].to_numpy().astype(np.float32),
        "log_plan_days": np.log1p(ev["plan_days"].fillna(30).clip(0, 420).to_numpy()).astype(np.float32),
    })
    return f


def check_feature_leakage(X, ev, L, h=7, seed=0) -> bool:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)[:300]
    ev = ev.iloc[:300]
    f1 = features_from_series(X, ev, L, h)
    X2 = X.copy()
    X2[:, L - h + 1:] = rng.uniform(0, 12, size=X2[:, L - h + 1:].shape).astype(np.float32)
    f2 = features_from_series(X2, ev, L, h)
    return bool(np.allclose(f1.to_numpy(), f2.to_numpy(), atol=1e-6))


def psi(base: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    base, cur = np.asarray(base, float), np.asarray(cur, float)
    if len(np.unique(base)) <= bins:
        cats = np.unique(base)
        pb = np.array([(base == c).mean() for c in cats]); pc = np.array([(cur == c).mean() for c in cats])
        pc = np.append(pc, (~np.isin(cur, cats)).mean()); pb = np.append(pb, 0.0)
    else:
        edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        pb = np.histogram(base, edges)[0] / len(base)
        pc = np.histogram(cur, edges)[0] / len(cur)
    pb, pc = np.clip(pb, eps, None), np.clip(pc, eps, None)
    return float(np.sum((pc - pb) * np.log(pc / pb)))


def reliability_table(y, p, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("bin")
    t = g.agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"), events=("y", "sum")).reset_index()
    ci = [wilson_ci(int(e), int(n)) for e, n in zip(t["events"], t["n"])]
    t["obs_lo"], t["obs_hi"] = [c[0] for c in ci], [c[1] for c in ci]
    return t


def ece(y, p, bins: int = 10) -> float:
    t = reliability_table(y, p, bins)
    return float(np.sum(t["n"] * (t["mean_pred"] - t["observed"]).abs()) / t["n"].sum())


def fit_model(F_train: pd.DataFrame, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    return m.fit(F_train[FEATURES].to_numpy(), np.asarray(y_train))


def permutation_importance(model, F, y, n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    X = F[FEATURES].to_numpy().copy()
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for j, name in enumerate(FEATURES):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        rows.append({"feature": name, "auc_drop": float(np.mean(drops)), "auc_drop_sd": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)


def slice_table(df: pd.DataFrame, col: str, min_n: int = 300, min_pos: int = 30) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    rows = []
    for lvl, d in df.groupby(col, dropna=False):
        pos = int(d["y"].sum())
        auc = roc_auc_score(d["y"], d["score"]) if len(d) >= min_n and pos >= min_pos and pos < len(d) else np.nan
        rows.append({"slice_col": col, "level": str(lvl), "n": len(d), "churners": pos, "churn_rate": d["y"].mean(),
                     "mean_score": d["score"].mean(), "calibration_gap": d["score"].mean() - d["y"].mean(),
                     "auc": auc, "reliable": bool(len(d) >= min_n and pos >= min_pos)})
    return pd.DataFrame(rows)


def run_model(X, ev: pd.DataFrame, has_log, L: int, h: int = 7, train_end="2017-02-14", min_baseline=MIN_BASELINE,
              exclude_shared: bool = False) -> dict:
    from sklearn.metrics import roc_auc_score, brier_score_loss
    if not check_feature_leakage(X, ev, L, h):
        raise AssertionError("LEAKAGE: features changed when post-cutoff days were scrambled")
    base = baseline(X, L)
    ok = np.asarray(has_log, bool) & (base >= min_baseline)
    F = features_from_series(X, ev, L, h)
    d = ev.reset_index(drop=True).copy()
    d["y"] = d["is_churn"].to_numpy()
    tr = ok & (d["expiry_date"] <= pd.Timestamp(train_end)).to_numpy()
    train_customers = set(d.loc[tr, "msno"])
    late = ok & (d["expiry_date"] > pd.Timestamp(train_end)).to_numpy()
    shared = late & d["msno"].isin(train_customers).to_numpy()
    # exclude_shared=True drops later events of customers seen in training (independence), but on real data this removes
    # most renewing customers and changes the population mix (a drift ARTEFACT), so the default keeps them and reports the count.
    mon = late & ~shared if exclude_shared else late
    model = fit_model(F[tr], d.loc[tr, "y"])
    score = np.full(len(d), np.nan)
    score[ok] = model.predict_proba(F.loc[ok, FEATURES].to_numpy())[:, 1]
    d["score"] = score
    d["split"] = np.where(tr, "train", np.where(mon, "monitor", "excluded"))
    m = d[mon]
    out = {"scores": d[["event_id", "split", "score"]], "features": F, "model": model, "n_train": int(tr.sum()),
           "n_monitor": int(mon.sum()), "n_excluded_shared_customers": int(shared.sum()) if exclude_shared else 0,
           "n_monitor_events_from_training_customers": int((mon & shared).sum()), "exclude_shared": bool(exclude_shared)}
    out["metrics"] = {
        "auc_monitor": float(roc_auc_score(m["y"], m["score"])), "brier_monitor": float(brier_score_loss(m["y"], m["score"])),
        "ece_monitor": ece(m["y"], m["score"]), "auc_train": float(roc_auc_score(d.loc[tr, "y"], d.loc[tr, "score"])),
        "ece_train": ece(d.loc[tr, "y"], d.loc[tr, "score"]), "base_rate_train": float(d.loc[tr, "y"].mean()),
        "base_rate_monitor": float(m["y"].mean()), "mean_score_monitor": float(m["score"].mean())}
    out["reliability_monitor"] = reliability_table(m["y"], m["score"])
    out["reliability_train"] = reliability_table(d.loc[tr, "y"], d.loc[tr, "score"])
    out["importance"] = permutation_importance(model, F[mon], m["y"].to_numpy())
    # weekly drift (weeks since 2017-02-01) vs the training window
    wk = ((d["expiry_date"] - pd.Timestamp("2017-02-01")).dt.days // 7).clip(lower=0)
    drift, perf = [], []
    for w in sorted(wk[mon].unique()):
        idx = mon & (wk == w).to_numpy()
        for f in FEATURES + ["score"]:
            ref = d.loc[tr, "score"].to_numpy() if f == "score" else F.loc[tr, f].to_numpy()
            cur = d.loc[idx, "score"].to_numpy() if f == "score" else F.loc[idx, f].to_numpy()
            drift.append({"week": int(w), "week_start": (pd.Timestamp("2017-02-01") + pd.Timedelta(days=7 * int(w))).date(),
                          "feature": f, "psi": psi(ref, cur), "n": int(idx.sum())})
        yy, ss = d.loc[idx, "y"].to_numpy(), d.loc[idx, "score"].to_numpy()
        perf.append({"week": int(w), "week_start": (pd.Timestamp("2017-02-01") + pd.Timedelta(days=7 * int(w))).date(), "n": int(idx.sum()),
                     "churners": int(yy.sum()), "base_rate": float(yy.mean()), "mean_score": float(ss.mean()),
                     "calibration_gap": float(ss.mean() - yy.mean()),
                     "auc": float(roc_auc_score(yy, ss)) if 20 <= yy.sum() < len(yy) and len(yy) >= 200 else np.nan})
    out["drift"], out["perf_week"] = pd.DataFrame(drift), pd.DataFrame(perf)
    out["split_frame"] = d
    return out


def build_slices(split_frame: pd.DataFrame, members: pd.DataFrame, top_n: int = 6, min_n: int = 300, min_pos: int = 30) -> pd.DataFrame:
    """Subgroup (bias) table on the monitoring set: renewal type, registration channel, city, gender, age band."""
    d = split_frame[split_frame["split"] == "monitor"].copy()
    m = members.drop_duplicates("msno").set_index("msno").reindex(d["msno"])
    d["renewal_type"] = np.where(d["is_auto_renew"] == 1, "Auto-renew", "Manual")
    d["gender"] = m["gender"].fillna("unknown").replace("", "unknown").to_numpy()
    bd = m["bd"].to_numpy(dtype=float)
    d["age_band"] = pd.cut(pd.Series(bd), [14, 20, 30, 40, 60], labels=["15-20", "21-30", "31-40", "41-60"]).astype(object).fillna("unknown").to_numpy()
    for col in ("registered_via", "city"):
        s = m[col].to_numpy()
        top = pd.Series(s).value_counts().head(top_n).index
        d[col] = pd.Series(s).where(pd.Series(s).isin(top), "other").astype(str).to_numpy()
    return pd.concat([slice_table(d, c, min_n, min_pos) for c in ("renewal_type", "registered_via", "city", "gender", "age_band")], ignore_index=True)
