"""Experiments E1, E2 and E5: horizon sweep, feature-family ablation, calibration.

For each horizon h (days before expiry):
  * split by expiry month (train earlier months, validate on the next, test on the last)
  * fit Logistic Regression, Random Forest and XGBoost on the training months
  * calibrate on the validation month (isotonic); build conformal sets + intervals
  * score the test month; PR-AUC and ROC-AUC use raw scores (ranking), log loss,
    Brier and ECE use calibrated probabilities
  * XGBoost: repeat over seeds, bootstrap CIs, cumulative feature-family ablation
    (overall and per segment)

Outputs (outputs/): metrics_by_horizon.csv, seed_runs.csv, ablation.csv,
reliability.csv, conformal_summary.csv, predictions.parquet, splits.json
Artifacts (artifacts/h<h>/): xgb.json, calibrator.joblib, meta.json

  python -m src.run_experiments --dataset outputs/dataset.parquet
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import ARTIFACTS, FAMILIES, FAMILY_ORDER, FEATURES, OUTPUTS, SEGMENTS
from . import models as M


def run_horizon(df_h, h, kinds, n_seeds, do_ablation, art_dir, min_per_class=20):
    tr, va, te, info = M.split_by_month(df_h)
    yva = va["is_churn"].to_numpy()
    yte = te["is_churn"].to_numpy()
    for name, d in (("train", tr), ("val", va), ("test", te)):
        if d["is_churn"].nunique() < 2:
            raise ValueError(f"h={h}: {name} split has a single class; widen --months")
    print(f"h={h}: train={len(tr):,} val={len(va):,} test={len(te):,} "
          f"(test churn {yte.mean():.3f}) {info['test_month']}")

    metrics_rows, preds = [], None
    xgb_bundle = None
    for kind in kinds:
        model, pv, pt, extra = M.fit_predict(kind, tr, va, te, FEATURES)
        cal = M.RiskCalibrator().fit(pv, yva)
        pt_cal = cal.transform(pt)
        row = M.evaluate(yte, pt, pt_cal)
        row.update(model=kind, horizon=h, n_train=len(tr), n_val=len(va),
                   baseline_pr_auc=float(yte.mean()), **extra)
        if kind == "xgb":
            lo, hi = M.bootstrap_ci(yte, pt, average_precision_score)
            row.update(pr_auc_ci_low=lo, pr_auc_ci_high=hi)
            lo, hi = M.bootstrap_ci(yte, pt, roc_auc_score)
            row.update(roc_auc_ci_low=lo, roc_auc_ci_high=hi)
            xgb_bundle = (model, cal, pt, pt_cal, extra)
        metrics_rows.append(row)
        print(f"   {kind:7s} PR-AUC={row['pr_auc']:.4f}  ROC-AUC={row['roc_auc']:.4f}  "
              f"logloss={row['logloss_cal']:.4f}")

    # ---------------------------------------------------------------- xgb extras
    seed_rows, abl_rows, rel_rows, conf_rows = [], [], [], []
    if xgb_bundle:
        model, cal, pt, pt_cal, extra = xgb_bundle
        best = extra["best_iteration"]
        Xtr, ytr = M.xy(tr, FEATURES)
        Xte, _ = M.xy(te, FEATURES)
        seed_rows.append({"horizon": h, "seed": M.SEED, "pr_auc": average_precision_score(yte, pt),
                          "roc_auc": roc_auc_score(yte, pt)})
        for s in range(1, n_seeds):
            m_s, _ = M.train_xgb(Xtr, ytr, seed=M.SEED + s, n_estimators=best)
            p_s = m_s.predict_proba(Xte)[:, 1]
            seed_rows.append({"horizon": h, "seed": M.SEED + s,
                              "pr_auc": average_precision_score(yte, p_s),
                              "roc_auc": roc_auc_score(yte, p_s)})

        # reliability (equal-frequency bins on the test month)
        order = np.argsort(pt_cal)
        for i, c in enumerate(np.array_split(order, 10)):
            lo, hi = M.wilson(np.array([yte[c].sum()]), np.array([len(c)]))
            rel_rows.append({"horizon": h, "bin": i + 1, "n": len(c),
                             "mean_predicted": float(pt_cal[c].mean()),
                             "observed_rate": float(yte[c].mean()),
                             "obs_ci_low": float(lo[0]), "obs_ci_high": float(hi[0])})

        sets = cal.prediction_set(pt_cal)
        conf_rows.append({"horizon": h, "alpha": cal.alpha,
                          "target_coverage": 1 - cal.alpha,
                          "empirical_coverage": cal.coverage(pt_cal, yte),
                          "share_churn_only": float((sets == "churn").mean()),
                          "share_stay_only": float((sets == "stay").mean()),
                          "share_both": float((sets == "both").mean()),
                          "share_empty": float((sets == "empty").mean()),
                          "share_abstain": float(np.isin(sets, ["both", "empty"]).mean())})

        lo, hi = cal.interval(pt_cal)
        preds = te[["event_id", "msno", "horizon", "expire_month", "expire_date", "cutoff_date",
                    "segment", "renewal_value", "is_churn"]].copy()
        preds["p_raw"] = pt
        preds["p_cal"] = pt_cal
        preds["p_low"] = lo
        preds["p_high"] = hi
        preds["conformal_set"] = sets
        preds["decision"] = np.select(
            [sets == "churn", sets == "stay"], ["confident_churn", "confident_stay"], "abstain")
        preds["risk_band"] = cal.risk_band(pt_cal)
        preds["risk_rank_pct"] = (pd.Series(-pt_cal).rank(method="first").to_numpy()
                                  / len(preds))

        meta = {"horizon": h, "features": FEATURES, "split": info,
                "best_iteration": best, "n_train": len(tr), "calibrator": cal.to_meta()}
        M.save_artifacts(art_dir / f"h{h}", model, cal, meta)

        # cumulative feature-family ablation (XGBoost, early stopping on the val month)
        if do_ablation:
            seg_te = te["segment"].to_numpy()
            used, prev_ap = [], None
            for fam in FAMILY_ORDER:
                used += FAMILIES[fam]
                Xa, ya = M.xy(tr, used)
                Xv, yv = M.xy(va, used)
                Xt, _ = M.xy(te, used)
                m_a, _ = M.train_xgb(Xa, ya, Xv, yv)
                p_a = m_a.predict_proba(Xt)[:, 1]
                for seg in ["all"] + SEGMENTS:
                    mask = np.ones(len(yte), bool) if seg == "all" else (seg_te == seg)
                    npos, nneg = int(yte[mask].sum()), int((1 - yte[mask]).sum())
                    if npos < min_per_class or nneg < min_per_class:
                        continue
                    abl_rows.append({
                        "horizon": h, "families_up_to": fam, "n_features": len(used),
                        "segment": seg, "n_test": int(mask.sum()), "n_pos": npos,
                        "churn_rate": npos / mask.sum(),
                        "pr_auc": average_precision_score(yte[mask], p_a[mask]),
                        "roc_auc": roc_auc_score(yte[mask], p_a[mask])})
    return metrics_rows, seed_rows, abl_rows, rel_rows, conf_rows, preds, info


def main(a):
    df_all = pd.read_parquet(a.dataset)
    horizons = a.horizons or sorted(df_all["horizon"].unique(), reverse=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    M_rows, S_rows, A_rows, R_rows, C_rows, P, splits = [], [], [], [], [], [], {}
    for h in horizons:
        df_h = M.load_dataset(a.dataset, horizon=h, common_cohort=(a.cohort == "common"))
        res = run_horizon(df_h, h, a.models, a.n_seeds, not a.skip_ablation, Path(a.artifacts))
        M_rows += res[0]; S_rows += res[1]; A_rows += res[2]; R_rows += res[3]
        C_rows += res[4]; P.append(res[5]); splits[int(h)] = res[6]

    pd.DataFrame(M_rows).to_csv(out / "metrics_by_horizon.csv", index=False)
    pd.DataFrame(S_rows).to_csv(out / "seed_runs.csv", index=False)
    pd.DataFrame(R_rows).to_csv(out / "reliability.csv", index=False)
    pd.DataFrame(C_rows).to_csv(out / "conformal_summary.csv", index=False)
    if A_rows:
        abl = pd.DataFrame(A_rows)
        abl = abl.sort_values(["horizon", "segment", "n_features"])
        abl["pr_auc_gain"] = abl.groupby(["horizon", "segment"])["pr_auc"].diff()
        abl.to_csv(out / "ablation.csv", index=False)
    pd.concat([p for p in P if p is not None]).to_parquet(out / "predictions.parquet", index=False)
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    print(f"\nwrote results to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(OUTPUTS / "dataset.parquet"))
    ap.add_argument("--out", default=str(OUTPUTS))
    ap.add_argument("--artifacts", default=str(ARTIFACTS))
    ap.add_argument("--horizons", type=int, nargs="+")
    ap.add_argument("--models", nargs="+", default=["logreg", "rf", "xgb"],
                    choices=["logreg", "rf", "xgb"])
    ap.add_argument("--cohort", choices=["common", "per_horizon"], default="common")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--skip-ablation", action="store_true")
    main(ap.parse_args())
