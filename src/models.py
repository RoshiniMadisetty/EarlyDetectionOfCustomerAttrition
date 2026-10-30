"""Models, calibration, uncertainty and metrics shared by all experiment scripts."""
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from xgboost import XGBClassifier

from .config import CATEGORICAL, FEATURES

SEED = 42
EPS = 1e-3  # probability clip so log loss stays finite


# ----------------------------------------------------------------------------- data
def load_dataset(path, horizon=None, common_cohort=True) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if horizon is not None:
        df = df[df["horizon"] == horizon]
    if common_cohort:
        df = df[df["in_common_cohort"]]
    df = df.copy()
    df["expire_month"] = pd.to_datetime(df["expire_month"])
    return df.reset_index(drop=True)


def split_by_month(df: pd.DataFrame):
    """Out-of-time split on expiry month: last month = test, one before = validation,
    everything earlier = train. Needs at least three months."""
    months = sorted(df["expire_month"].unique())
    if len(months) < 3:
        raise ValueError(f"need >= 3 expiry months for a time-based split, got {len(months)}")
    test_m, val_m, train_m = months[-1], months[-2], months[:-2]
    tr = df[df["expire_month"].isin(train_m)]
    va = df[df["expire_month"] == val_m]
    te = df[df["expire_month"] == test_m]
    info = {"train_months": [str(pd.Timestamp(m).date()) for m in train_m],
            "val_month": str(pd.Timestamp(val_m).date()),
            "test_month": str(pd.Timestamp(test_m).date())}
    return tr, va, te, info


def xy(df, features):
    return df[features].astype("float32"), df["is_churn"].astype(int).to_numpy()


# ----------------------------------------------------------------------------- models
def make_logreg(features, seed=SEED):
    cat = [c for c in features if c in CATEGORICAL]
    num = [c for c in features if c not in CATEGORICAL]
    parts = []
    if cat:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value=-1)),
            ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=50, sparse_output=False))]), cat))
    parts.append(("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("q", QuantileTransformer(n_quantiles=200, output_distribution="normal",
                                  subsample=200_000, random_state=seed))]), num))
    return Pipeline([("prep", ColumnTransformer(parts)),
                     ("clf", LogisticRegression(max_iter=2000, C=0.5,
                                                class_weight="balanced"))])


def make_rf(n_train, seed=SEED):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, max_features="sqrt", n_jobs=-1,
            class_weight="balanced_subsample", random_state=seed,
            max_samples=min(1.0, 200_000 / max(n_train, 1))))])


XGB_PARAMS = dict(learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, reg_lambda=1.0, tree_method="hist")


def train_xgb(Xtr, ytr, Xva=None, yva=None, seed=SEED, n_estimators=None, params=None,
              weight_positive=True):
    """XGBoost with class weighting. Early stopping on the validation month when
    n_estimators is None; otherwise a fixed number of trees (no validation needed)."""
    p = dict(XGB_PARAMS)
    if params:
        p.update(params)
    pos = max(int(ytr.sum()), 1)
    spw = (len(ytr) - pos) / pos if weight_positive else 1.0
    if n_estimators is None:
        model = XGBClassifier(n_estimators=800, early_stopping_rounds=40, eval_metric="aucpr",
                              scale_pos_weight=spw, random_state=seed, n_jobs=-1, **p)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        best = int(model.best_iteration) + 1
    else:
        model = XGBClassifier(n_estimators=n_estimators, eval_metric="aucpr",
                              scale_pos_weight=spw, random_state=seed, n_jobs=-1, **p)
        model.fit(Xtr, ytr, verbose=False)
        best = n_estimators
    return model, best


def fit_predict(kind, tr, va, te, features, seed=SEED):
    """Fit one model on the training months; return raw probabilities for val and test."""
    Xtr, ytr = xy(tr, features)
    Xva, yva = xy(va, features)
    Xte, _ = xy(te, features)
    extra = {}
    if kind == "logreg":
        model = make_logreg(features, seed).fit(Xtr, ytr)
    elif kind == "rf":
        model = make_rf(len(Xtr), seed).fit(Xtr, ytr)
    elif kind == "xgb":
        model, best = train_xgb(Xtr, ytr, Xva, yva, seed=seed)
        extra["best_iteration"] = best
    else:
        raise ValueError(kind)
    return model, model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1], extra


# ----------------------------------------------------------------------------- calibration
def wilson(k, n, z=1.96):
    n = np.maximum(n, 1)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


class RiskCalibrator:
    """Isotonic calibration + split-conformal prediction sets + per-bin Wilson intervals.

    The validation month is split at random into half A (fits the isotonic map) and
    half B (conformal quantile and interval bins), so none of the uncertainty
    statements are computed on data that fitted the calibration map.
    Risk bands use fixed thresholds taken from the validation month.
    """

    def __init__(self, n_bins=20, alpha=0.10, seed=SEED):
        self.n_bins, self.alpha, self.seed = n_bins, alpha, seed

    def fit(self, p_raw, y):
        p_raw, y = np.asarray(p_raw, float), np.asarray(y, int)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(y))
        a, b = perm[: len(y) // 2], perm[len(y) // 2:]
        self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p_raw[a], y[a])
        pb = self.transform(p_raw[b])
        yb = y[b]
        # conformal: nonconformity = 1 - probability assigned to the true class
        s = np.where(yb == 1, 1 - pb, pb)
        n = len(s)
        level = min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q = float(np.quantile(s, level, method="higher"))
        # interval bins (equal-frequency on calibrated score)
        edges = np.unique(np.quantile(pb, np.linspace(0, 1, self.n_bins + 1)))
        self.edges = edges
        idx = self._bin(pb)
        nb = np.bincount(idx, minlength=len(edges) - 1).astype(float)
        kb = np.bincount(idx, weights=yb, minlength=len(edges) - 1)
        self.bin_low, self.bin_high = wilson(kb, nb)
        # risk bands from the whole validation month
        pall = self.transform(p_raw)
        self.band_edges = [float(np.quantile(pall, qq)) for qq in (0.50, 0.80, 0.95)]
        return self

    def transform(self, p_raw):
        return np.clip(self.iso.predict(np.asarray(p_raw, float)), EPS, 1 - EPS)

    def _bin(self, p):
        if len(self.edges) <= 2:
            return np.zeros(len(p), dtype=int)
        return np.clip(np.searchsorted(self.edges[1:-1], p, side="right"), 0, len(self.edges) - 2)

    def interval(self, p_cal):
        idx = self._bin(np.asarray(p_cal))
        return self.bin_low[idx], self.bin_high[idx]

    def prediction_set(self, p_cal):
        """Label of the conformal set: churn / stay / both / empty.
        'both' (q >= 0.5) and 'empty' (neither label conforms) are both treated as
        'abstain' in the dashboard: the model cannot commit at the chosen confidence."""
        p = np.asarray(p_cal)
        has1 = (1 - p) <= self.q
        has0 = p <= self.q
        out = np.full(len(p), "empty", dtype=object)
        out[has1 & ~has0] = "churn"
        out[has0 & ~has1] = "stay"
        out[has1 & has0] = "both"
        return out

    def coverage(self, p_cal, y):
        p, y = np.asarray(p_cal), np.asarray(y)
        covered = np.where(y == 1, (1 - p) <= self.q, p <= self.q)
        return float(covered.mean())

    def risk_band(self, p_cal):
        return np.array(["Low", "Medium", "High", "Critical"], dtype=object)[
            np.searchsorted(self.band_edges, np.asarray(p_cal), side="right")]

    def to_meta(self):
        return {"alpha": self.alpha, "conformal_q": self.q, "band_edges": self.band_edges,
                "bin_edges": self.edges.tolist()}


# ----------------------------------------------------------------------------- metrics
def top_k_stats(y, p, frac):
    n = len(y)
    m = max(1, int(math.ceil(frac * n)))
    order = np.argsort(-np.asarray(p), kind="stable")[:m]
    tp = float(np.asarray(y)[order].sum())
    pos = float(np.sum(y))
    return {"precision": tp / m, "recall": tp / pos if pos else float("nan"),
            "lift": (tp / m) / (pos / n) if pos else float("nan")}


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(y) * abs(y[c].mean() - p[c].mean()) for c in chunks if len(c)))


def evaluate(y, p_raw, p_cal=None):
    y = np.asarray(y)
    row = {"n": int(len(y)), "n_pos": int(y.sum()), "churn_rate": float(y.mean()),
           "pr_auc": float(average_precision_score(y, p_raw)),
           "roc_auc": float(roc_auc_score(y, p_raw))}
    if p_cal is not None:
        row.update(logloss_cal=float(log_loss(y, p_cal, labels=[0, 1])),
                   brier_cal=float(brier_score_loss(y, p_cal)), ece_cal=ece(y, p_cal))
    for f in (0.05, 0.10):
        s = top_k_stats(y, p_raw, f)
        row[f"precision_at_{int(f * 100)}pct"] = s["precision"]
        row[f"recall_at_{int(f * 100)}pct"] = s["recall"]
        row[f"lift_at_{int(f * 100)}pct"] = s["lift"]
    return row


def bootstrap_ci(y, p, metric, n_boot=200, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(metric(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ----------------------------------------------------------------------------- artifacts
def save_artifacts(dirpath: Path, model, calibrator, meta: dict):
    dirpath.mkdir(parents=True, exist_ok=True)
    model.save_model(str(dirpath / "xgb.json"))
    joblib.dump(calibrator, dirpath / "calibrator.joblib")
    (dirpath / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def load_artifacts(dirpath: Path):
    model = XGBClassifier()
    model.load_model(str(dirpath / "xgb.json"))
    cal = joblib.load(dirpath / "calibrator.joblib")
    meta = json.loads((dirpath / "meta.json").read_text())
    return model, cal, meta
