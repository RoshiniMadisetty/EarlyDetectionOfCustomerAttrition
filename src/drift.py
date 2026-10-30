"""Experiment E6: temporal drift and monitoring.

* static strategy : train once on the first n months, score every later month
* rolling strategy: for month i, retrain on the n months ending `label_lag` months
                    before i (labels of the latest months are not known yet)
* PSI of every feature, per month, against the first n (training) months

The model here is trained WITHOUT class weighting so its raw probabilities are
meaningful for log loss.

Outputs (outputs/): drift_metrics.csv, drift_psi.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .config import FEATURE_FAMILY, FEATURES, OUTPUTS
from . import models as M

N_TREES = 300


def psi(expected, actual, bins=10):
    """Population stability index with quantile bins from `expected`; NaN is its own bin."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    e_nan, a_nan = np.isnan(e).mean(), np.isnan(a).mean()
    ee, aa = e[~np.isnan(e)], a[~np.isnan(a)]
    if len(ee) == 0 or len(aa) == 0:
        return np.nan
    edges = np.unique(np.quantile(ee, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([ee.min() - 1, ee.max() + 1])
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(ee, edges)[0] / len(e)
    pa = np.histogram(aa, edges)[0] / len(a)
    pe, pa = np.append(pe, e_nan), np.append(pa, a_nan)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def score_month(model, d):
    X, y = M.xy(d, FEATURES)
    p = np.clip(model.predict_proba(X)[:, 1], M.EPS, 1 - M.EPS)
    return {"n": len(y), "churn_rate": float(y.mean()), "mean_predicted": float(p.mean()),
            "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "logloss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p))}


def main(a):
    out = Path(a.out)
    df = M.load_dataset(a.dataset, horizon=a.horizon, common_cohort=(a.cohort == "common"))
    months = sorted(df["expire_month"].unique())
    n, lag = a.n_train_months, a.label_lag
    if len(months) < n + lag + 1:
        raise ValueError(f"need at least {n + lag + 1} months, have {len(months)}")
    by_m = {m: df[df["expire_month"] == m] for m in months}
    pool = pd.concat([by_m[m] for m in months[:n]])
    rows = []

    static, _ = M.train_xgb(*M.xy(pool, FEATURES), n_estimators=N_TREES, weight_positive=False)
    for i in range(n + lag, len(months)):
        m = months[i]
        label = str(pd.Timestamp(m).date())
        rows.append({"horizon": a.horizon, "month": label, "strategy": "static",
                     "months_since_training": i - n + 1, **score_month(static, by_m[m])})
        win = pd.concat([by_m[x] for x in months[i - lag - n: i - lag]])
        roll, _ = M.train_xgb(*M.xy(win, FEATURES), n_estimators=N_TREES, weight_positive=False)
        rows.append({"horizon": a.horizon, "month": label, "strategy": "rolling",
                     "months_since_training": lag + 1, **score_month(roll, by_m[m])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "drift_metrics.csv", index=False)

    psi_rows = []
    for m in months:
        label = str(pd.Timestamp(m).date())
        for f in FEATURES:
            v = psi(pool[f], by_m[m][f])
            status = ("n/a" if np.isnan(v) else "stable" if v < 0.10
                      else "moderate" if v < 0.25 else "major")
            psi_rows.append({"horizon": a.horizon, "month": label, "feature": f,
                             "family": FEATURE_FAMILY[f], "psi": v, "status": status,
                             "in_training_window": m in months[:n]})
    pd.DataFrame(psi_rows).to_csv(out / "drift_psi.csv", index=False)
    print(metrics[["month", "strategy", "pr_auc", "roc_auc", "logloss"]].to_string(index=False))
    print(f"wrote drift_metrics.csv and drift_psi.csv to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(OUTPUTS / "dataset.parquet"))
    ap.add_argument("--out", default=str(OUTPUTS))
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--n-train-months", type=int, default=3)
    ap.add_argument("--label-lag", type=int, default=1)
    ap.add_argument("--cohort", choices=["common", "per_horizon"], default="common")
    main(ap.parse_args())
