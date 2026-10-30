"""Collect every result table into powerbi/data/*.csv for Power BI Desktop.

Adds a text key `row_key` = "<event_id>_<horizon>" to the customer-level tables,
because Power BI relationships need a single-column key.

Also builds the Unit 1 EDA tables:
  eda_daily_usage.csv  mean listening per day relative to the cutoff, churned vs retained
  eda_sample.csv       random sample of events with features and label
  correlation_matrix.csv  Spearman correlations among features and the label

  python -m src.export_powerbi --dataset outputs/dataset.parquet --data-dir data/parquet
"""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import DATA_PARQUET, FAMILIES, FEATURE_FAMILY, FEATURES, OUTPUTS, POWERBI

DESCRIPTIONS = {
    "anchor_payment_method_id": "Payment method of the expiring subscription",
    "anchor_plan_days": "Length of the expiring plan in days",
    "anchor_list_price": "List price of the expiring plan",
    "anchor_paid": "Amount actually paid for the expiring plan",
    "anchor_auto_renew": "1 if auto-renew was on for the expiring plan",
    "anchor_discount": "List price minus amount paid",
    "anchor_discount_ratio": "Discount as a share of list price",
    "n_cancel_since_anchor": "Cancel transactions since the expiring plan was bought",
    "n_cancel_30d": "Cancel transactions in the 30 days before the cutoff",
    "n_cancel_before": "All cancel transactions before the cutoff",
    "n_plan_changes": "Times the plan length changed between purchases",
    "n_pay_method_changes": "Times the payment method changed between purchases",
    "n_distinct_plan_days": "Distinct plan lengths used so far",
    "n_distinct_pay_methods": "Distinct payment methods used so far",
    "manual_tx_share": "Share of past purchases made without auto-renew",
    "days_since_anchor_tx": "Days from the plan purchase to the cutoff",
    "plan_progress": "Share of the plan already elapsed at the cutoff",
    "tenure_days": "Days since registration at the cutoff",
    "n_renewals": "Paid renewals before the expiring plan",
    "city": "City code", "registered_via": "Registration channel code",
    "gender_code": "1 male, 0 female, blank unknown", "bd": "Age (invalid values blanked)",
    "secs_7": "Listening seconds, last 7 days", "secs_14": "Listening seconds, last 14 days",
    "secs_30": "Listening seconds, last 30 days", "unq_7": "Unique songs, last 7 days",
    "unq_14": "Unique songs, last 14 days", "unq_30": "Unique songs, last 30 days",
    "plays_30": "Song plays, last 30 days",
    "completion_ratio_30": "Share of plays completed (100%), last 30 days",
    "skip_ratio_30": "Share of plays stopped before 25%, last 30 days",
    "avg_secs_per_active_day_30": "Mean seconds on days with any listening",
    "active_days_7": "Active days, last 7", "active_days_14": "Active days, last 14",
    "active_days_30": "Active days, last 30",
    "days_since_last_active": "Days since last listening (30 = none in window)",
    "max_inactivity_gap_30": "Longest run of inactive days in the 30-day window",
    "usage_ratio_7_vs_prev": "Daily listening, last 7 days vs the previous 23",
    "usage_ratio_14_vs_prev": "Daily listening, last 14 days vs the previous 16",
    "usage_slope_30": "Slope of daily seconds over 30 days (sec/day per day)",
    "usage_vs_baseline": "Daily listening vs the user's own prior 90-day baseline",
    "active_days_ratio_vs_baseline": "Active-day rate vs the user's own prior 90-day baseline",
}


def eda_tables(dataset, data_dir, horizon, out, sample_n, seed):
    con = duckdb.connect()
    ds = Path(dataset).as_posix()
    logs = (Path(data_dir) / "user_logs*.parquet").as_posix()
    con.execute(f"""
        CREATE TABLE ev AS
        SELECT event_id, msno, cutoff_date AS C, is_churn, segment
        FROM read_parquet('{ds}') WHERE horizon = {horizon} AND in_common_cohort
    """)
    daily = con.execute(f"""
        WITH n AS (SELECT segment, is_churn, count(*) AS n_events FROM ev GROUP BY 1, 2),
        d AS (
            SELECT ev.segment, ev.is_churn, l.date - ev.C AS day_offset,
                   SUM(l.total_secs) AS secs_sum, COUNT(*) AS active
            FROM ev JOIN read_parquet('{logs}') l
              ON l.msno = ev.msno AND l.date <= ev.C AND l.date > ev.C - 30
            GROUP BY 1, 2, 3)
        SELECT d.segment, CASE WHEN d.is_churn = 1 THEN 'Churned' ELSE 'Retained' END AS outcome,
               d.day_offset, d.secs_sum / n.n_events AS mean_secs,
               d.active * 1.0 / n.n_events AS share_active, n.n_events
        FROM d JOIN n USING (segment, is_churn) ORDER BY 1, 2, 3
    """).fetchdf()
    daily.insert(0, "horizon", horizon)
    daily.to_csv(out / "eda_daily_usage.csv", index=False)

    df = pd.read_parquet(ds)
    df = df[(df["horizon"] == horizon) & df["in_common_cohort"]]
    samp = df.sample(min(sample_n, len(df)), random_state=seed)
    cols = ["event_id", "msno", "horizon", "expire_month", "segment", "is_churn"] + FEATURES
    samp[cols].to_csv(out / "eda_sample.csv", index=False)
    corr = samp[FEATURES + ["is_churn"]].astype(float).corr(method="spearman")
    long = corr.stack().reset_index()
    long.columns = ["feature_x", "feature_y", "spearman"]
    long["family_x"] = long["feature_x"].map(FEATURE_FAMILY).fillna("label")
    long["family_y"] = long["feature_y"].map(FEATURE_FAMILY).fillna("label")
    long.to_csv(out / "correlation_matrix.csv", index=False)


def main(a):
    src, dst = Path(a.outputs), Path(a.powerbi) / "data"
    dst.mkdir(parents=True, exist_ok=True)

    preds = pd.read_parquet(src / "predictions.parquet")
    preds["row_key"] = preds["event_id"].astype(str) + "_" + preds["horizon"].astype(str)
    for c in ("expire_month", "expire_date", "cutoff_date"):
        preds[c] = pd.to_datetime(preds[c]).dt.strftime("%Y-%m-%d")
    preds.to_csv(dst / "predictions.csv", index=False)

    ds = pd.read_parquet(a.dataset)
    keys = preds[["event_id", "horizon"]]
    feat = keys.merge(ds[["event_id", "horizon"] + FEATURES], on=["event_id", "horizon"], how="left")
    feat.insert(0, "row_key", feat["event_id"].astype(str) + "_" + feat["horizon"].astype(str))
    feat.to_csv(dst / "customer_features.csv", index=False)

    long = pd.read_parquet(src / "shap_values_long.parquet")
    long.insert(0, "row_key", long["event_id"].astype(str) + "_" + long["horizon"].astype(str))
    long.to_csv(dst / "shap_values_long.csv", index=False)
    base = pd.read_csv(src / "shap_base.csv")
    base.insert(0, "row_key", base["event_id"].astype(str) + "_" + base["horizon"].astype(str))
    base.to_csv(dst / "shap_base.csv", index=False)

    for name in ["metrics_by_horizon", "ablation", "reliability", "conformal_summary",
                 "seed_runs", "shap_importance", "shap_family_importance",
                 "shap_rank_by_horizon", "shap_horizon_shift", "shap_stability",
                 "value_of_lead_time", "value_of_lead_time_best",
                 "drift_metrics", "drift_psi"]:
        f = src / f"{name}.csv"
        if f.exists():
            pd.read_csv(f).to_csv(dst / f"{name}.csv", index=False)
        else:
            print(f"  (missing {name}.csv - run the matching step first)")

    horizons = sorted(preds["horizon"].unique(), reverse=True)
    pd.DataFrame({"horizon": horizons,
                  "label": [f"{h} days before expiry" for h in horizons],
                  "sort_order": range(1, len(horizons) + 1)}).to_csv(dst / "horizons.csv", index=False)
    pd.DataFrame({"feature": FEATURES, "family": [FEATURE_FAMILY[f] for f in FEATURES],
                  "description": [DESCRIPTIONS.get(f, "") for f in FEATURES]}) \
      .to_csv(dst / "feature_catalog.csv", index=False)

    rep = Path(a.dataset).parent / "cohort_report.json"
    if rep.exists():
        pd.DataFrame(json.loads(rep.read_text())["per_horizon"]).to_csv(
            dst / "cohort_report.csv", index=False)

    eda_tables(a.dataset, a.data_dir, a.eda_horizon, dst, a.eda_sample, 42)
    print(f"wrote {len(list(dst.glob('*.csv')))} CSV files to {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(OUTPUTS / "dataset.parquet"))
    ap.add_argument("--outputs", default=str(OUTPUTS))
    ap.add_argument("--data-dir", default=str(DATA_PARQUET))
    ap.add_argument("--powerbi", default=str(POWERBI))
    ap.add_argument("--eda-horizon", type=int, default=14)
    ap.add_argument("--eda-sample", type=int, default=20000)
    main(ap.parse_args())
