"""Experiment E3: explanation timeline (SHAP by horizon) and explanation stability.

SHAP values are exact TreeSHAP values from XGBoost (pred_contribs), expressed in
log-odds of the class-weighted raw model score. Their ranking and relative size are
meaningful; their absolute scale is NOT the calibrated churn probability.

Outputs (outputs/): shap_importance.csv, shap_family_importance.csv,
shap_rank_by_horizon.csv, shap_horizon_shift.csv, shap_stability.csv,
shap_values_long.parquet, shap_base.csv ; figures/ : PNGs
"""
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from .config import ARTIFACTS, FAMILY_ORDER, FEATURE_FAMILY, FEATURES, OUTPUTS
from . import models as M


def shap_matrix(model, X):
    booster = model.get_booster()
    contrib = booster.predict(xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)
    return contrib[:, :-1], contrib[:, -1]


def top_k_jaccard(a, b, k=10):
    ta, tb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return len(ta & tb) / len(ta | tb)


def main(a):
    out = Path(a.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df_all = pd.read_parquet(a.dataset, columns=["horizon"])
    horizons = a.horizons or sorted(df_all["horizon"].unique(), reverse=True)
    rng = np.random.default_rng(M.SEED)

    imp_rows, stab_rows, long_parts, base_parts = [], [], [], []
    imp_vec = {}
    for h in horizons:
        model, cal, meta = M.load_artifacts(Path(a.artifacts) / f"h{h}")
        df_h = M.load_dataset(a.dataset, horizon=h, common_cohort=(a.cohort == "common"))
        tr, va, te, _ = M.split_by_month(df_h)

        def sample(d, n):
            return d.iloc[rng.choice(len(d), size=min(n, len(d)), replace=False)]

        te_s, va_s = sample(te, a.sample), sample(va, a.sample)
        Xte = te_s[FEATURES].astype("float32")
        Xva = va_s[FEATURES].astype("float32")
        sv_te, base_te = shap_matrix(model, Xte)
        sv_va, _ = shap_matrix(model, Xva)
        m_te, m_va = np.abs(sv_te).mean(0), np.abs(sv_va).mean(0)
        imp_vec[h] = m_te

        order = (-m_te).argsort().argsort() + 1
        for j, f in enumerate(FEATURES):
            imp_rows.append({"horizon": h, "feature": f, "family": FEATURE_FAMILY[f],
                             "mean_abs_shap": m_te[j], "rank": int(order[j]),
                             "share": m_te[j] / m_te.sum()})

        # stability 1: validation month vs test month importance
        stab_rows.append({"horizon": h, "comparison": "val_month_vs_test_month",
                          "spearman": spearmanr(m_va, m_te)[0],
                          "top10_jaccard": top_k_jaccard(m_va, m_te)})
        # stability 2: retrain with different seeds (fixed number of trees)
        Xtr, ytr = M.xy(tr, FEATURES)
        vecs = []
        for s in range(a.n_stab_seeds):
            m_s, _ = M.train_xgb(Xtr, ytr, seed=M.SEED + 100 + s,
                                 n_estimators=meta["best_iteration"])
            sv_s, _ = shap_matrix(m_s, Xte)
            vecs.append(np.abs(sv_s).mean(0))
        rho = [spearmanr(u, v)[0] for u, v in combinations(vecs, 2)]
        jac = [top_k_jaccard(u, v) for u, v in combinations(vecs, 2)]
        stab_rows.append({"horizon": h, "comparison": f"across_{a.n_stab_seeds}_seeds",
                          "spearman": float(np.mean(rho)), "top10_jaccard": float(np.mean(jac))})

        # long-format export for Power BI (a subset of customers)
        ex = rng.choice(len(te_s), size=min(a.export_n, len(te_s)), replace=False)
        ids = te_s.iloc[ex][["event_id", "msno"]].reset_index(drop=True)
        n_ex = len(ids)
        long_parts.append(pd.DataFrame({
            "horizon": h,
            "event_id": np.repeat(ids["event_id"].to_numpy(), len(FEATURES)),
            "msno": np.repeat(ids["msno"].to_numpy(), len(FEATURES)),
            "feature": np.tile(FEATURES, n_ex),
            "family": np.tile([FEATURE_FAMILY[f] for f in FEATURES], n_ex),
            "shap_value": sv_te[ex].ravel(),
            "feature_value": Xte.iloc[ex].to_numpy().ravel()}))
        base_parts.append(pd.DataFrame({"horizon": h, "event_id": ids["event_id"],
                                        "base_value": base_te[ex],
                                        "raw_logit": base_te[ex] + sv_te[ex].sum(1)}))

        # SHAP beeswarm
        try:
            import shap
            plt.figure()
            shap.summary_plot(sv_te, Xte, max_display=15, show=False)
            plt.title(f"SHAP summary, {h} days before expiry")
            plt.tight_layout()
            plt.savefig(fig_dir / f"shap_summary_h{h}.png", dpi=150)
            plt.close()
        except Exception as e:  # plotting is optional
            print(f"   (skipped beeswarm for h={h}: {e})")
        print(f"h={h}: top features -> " + ", ".join(
            np.array(FEATURES)[np.argsort(-m_te)[:5]]))

    imp = pd.DataFrame(imp_rows)
    imp.to_csv(out / "shap_importance.csv", index=False)
    fam = imp.groupby(["horizon", "family"], as_index=False)["mean_abs_shap"].sum()
    fam["share"] = fam["mean_abs_shap"] / fam.groupby("horizon")["mean_abs_shap"].transform("sum")
    fam.to_csv(out / "shap_family_importance.csv", index=False)
    imp.pivot(index="feature", columns="horizon", values="rank").reset_index() \
       .to_csv(out / "shap_rank_by_horizon.csv", index=False)
    pd.DataFrame(stab_rows).to_csv(out / "shap_stability.csv", index=False)

    shift = []
    for h1, h2 in combinations(sorted(imp_vec, reverse=True), 2):
        shift.append({"horizon_a": h1, "horizon_b": h2,
                      "spearman": spearmanr(imp_vec[h1], imp_vec[h2])[0],
                      "top10_jaccard": top_k_jaccard(imp_vec[h1], imp_vec[h2])})
    pd.DataFrame(shift).to_csv(out / "shap_horizon_shift.csv", index=False)
    pd.concat(long_parts).to_parquet(out / "shap_values_long.parquet", index=False)
    pd.concat(base_parts).to_csv(out / "shap_base.csv", index=False)

    # signal timeline figure: share of explanation by feature family at each horizon
    piv = fam.pivot(index="horizon", columns="family", values="share") \
             .reindex(columns=FAMILY_ORDER).sort_index(ascending=False)
    ax = piv.plot(kind="bar", stacked=True, figsize=(7, 4), colormap="viridis")
    ax.set_xlabel("Warning horizon (days before expiry)")
    ax.set_ylabel("Share of mean |SHAP|")
    ax.set_title("Which signals drive the warning, by horizon")
    ax.legend(title="Feature family", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "signal_timeline.png", dpi=150)
    plt.close()
    print(f"wrote explanation outputs to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(OUTPUTS / "dataset.parquet"))
    ap.add_argument("--out", default=str(OUTPUTS))
    ap.add_argument("--artifacts", default=str(ARTIFACTS))
    ap.add_argument("--horizons", type=int, nargs="+")
    ap.add_argument("--cohort", choices=["common", "per_horizon"], default="common")
    ap.add_argument("--sample", type=int, default=20000, help="test rows used for importance")
    ap.add_argument("--export-n", type=int, default=2000, help="customers exported per horizon")
    ap.add_argument("--n-stab-seeds", type=int, default=5)
    main(ap.parse_args())
