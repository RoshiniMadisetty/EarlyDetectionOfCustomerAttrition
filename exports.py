"""Step 6: tables shaped for Power BI + theme + DAX measures.

Output directory layout (all CSV, UTF-8, comma separated, ISO dates, '.' decimals):
  dim_event.csv          dashboard sample (one row per event; anonymous integer ids, no customer ids)
  fact_series.csv        daily series of the sample events, with Python moving averages / spline / LOWESS / CUSUM
  fact_cohort_curves.csv cohort means with normal-approx CI over ALL analysable events (not only the sample)
  fact_leadtime.csv      RQ1 benchmark (method x group x alpha x metric, with bootstrap CI)
  fact_calibration.csv, fact_score_hist.csv, fact_importance.csv, fact_drift.csv, fact_perf_week.csv, fact_slices.csv
  fact_sampling.csv      sample vs full churn rate (limitations)
  dim_method.csv, data_dictionary.csv, theme.json, dax_measures.txt
The Python series for the dashboard are computed on MINUTES (not log minutes) so that DAX moving averages computed in
Power BI on the `minutes` column can be compared number-for-number with the `sma7_py` column.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import smoothers as sm
from .benchmark import baseline, best_per_family, alarm_leads_for, cusum_threshold, MIN_BASELINE

# Okabe-Ito (colour-blind-safe categorical) and ColorBrewer YlOrRd 5-class (sequential risk scale)
CATEGORICAL = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#000000", "#999999"]
RISK_SCALE = ["#FFFFB2", "#FECC5C", "#FD8D3C", "#F03B20", "#BD0026"]
SCORE_BANDS = [0, 0.02, 0.05, 0.10, 0.20, 1.0001]
SCORE_LABELS = ["0-2%", "2-5%", "5-10%", "10-20%", "20%+"]
REPRESENTATIVE = {"raw": "raw", "SMA": "SMA w=7", "EMA": "EMA a=0.15", "LOWESS": "LOWESS w=14", "Spline": "Spline w=28 lam=10"}


def theme_json() -> dict:
    return {
        "name": "SilentChurn colour-blind-safe",
        "dataColors": CATEGORICAL,
        "background": "#FFFFFF", "foreground": "#1F2937", "tableAccent": "#0072B2",
        "good": "#009E73", "neutral": "#E69F00", "bad": "#D55E00",
        "minimum": RISK_SCALE[0], "center": RISK_SCALE[2], "maximum": RISK_SCALE[4], "null": "#9CA3AF",
        "textClasses": {
            "callout": {"fontSize": 28, "fontFace": "Segoe UI Semibold", "color": "#111827"},
            "title": {"fontSize": 14, "fontFace": "Segoe UI Semibold", "color": "#111827"},
            "header": {"fontSize": 12, "fontFace": "Segoe UI Semibold", "color": "#111827"},
            "label": {"fontSize": 11, "fontFace": "Segoe UI", "color": "#374151"},
        },
        "visualStyles": {"*": {"*": {"*": [{"fontSize": 11, "fontFamily": "Segoe UI"}]}}},
    }


DAX = r"""-- ===== Page 1: Foundations =====
Sample Churn Rate = AVERAGE ( fact_sampling[sample_churn_rate] )
Sample CI Lower = AVERAGE ( fact_sampling[sample_ci_lo] )
Sample CI Upper = AVERAGE ( fact_sampling[sample_ci_hi] )
Full Churn Rate = AVERAGE ( fact_sampling[full_churn_rate] )
Cohort Mean = AVERAGE ( fact_cohort_curves[mean] )
Cohort Lower = AVERAGE ( fact_cohort_curves[lo] )
Cohort Upper = AVERAGE ( fact_cohort_curves[hi] )
Events = DISTINCTCOUNT ( dim_event[event_id] )
Churned Events = CALCULATE ( [Events], dim_event[outcome] = "Churned" )
Churn Rate (dashboard sample) = DIVIDE ( [Churned Events], [Events] )
Avg Minutes per Day = AVERAGE ( fact_series[minutes] )
Avg Log Minutes = AVERAGE ( fact_series[x_log1p_minutes] )

-- ===== Page 2: Smoothing lab (Unit 1 calculation in Power BI) =====
-- t is the day index (1 = oldest day, increasing to the day before expiry), so a TRAILING window is t0-6 .. t0.
MA7 DAX =
VAR t0 = MAX ( fact_series[t] )
VAR cnt =
    CALCULATE ( COUNTROWS ( fact_series ),
        FILTER ( ALL ( fact_series[t] ), fact_series[t] <= t0 && fact_series[t] > t0 - 7 ) )
RETURN
    IF ( cnt = 7,
        CALCULATE ( AVERAGE ( fact_series[minutes] ),
            FILTER ( ALL ( fact_series[t] ), fact_series[t] <= t0 && fact_series[t] > t0 - 7 ) ),
        BLANK () )

MA14 DAX =
VAR t0 = MAX ( fact_series[t] )
VAR cnt =
    CALCULATE ( COUNTROWS ( fact_series ),
        FILTER ( ALL ( fact_series[t] ), fact_series[t] <= t0 && fact_series[t] > t0 - 14 ) )
RETURN
    IF ( cnt = 14,
        CALCULATE ( AVERAGE ( fact_series[minutes] ),
            FILTER ( ALL ( fact_series[t] ), fact_series[t] <= t0 && fact_series[t] > t0 - 14 ) ),
        BLANK () )

-- Exponential moving average in DAX would need recursion; use the precomputed Python column instead.
MA7 Python = AVERAGE ( fact_series[sma7_py] )
MA7 Difference (DAX - Python) = [MA7 DAX] - [MA7 Python]
Max Abs MA7 Difference =
MAXX ( VALUES ( fact_series[t] ), ABS ( [MA7 Difference (DAX - Python)] ) )
EMA Python = AVERAGE ( fact_series[ema015_py] )
LOWESS Python = AVERAGE ( fact_series[lowess14_py] )
Spline Python = AVERAGE ( fact_series[spline28_py] )
Raw Minutes = AVERAGE ( fact_series[minutes] )
CUSUM Statistic = AVERAGE ( fact_series[cusum] )
CUSUM Threshold = AVERAGE ( fact_series[cusum_threshold] )
CUSUM Alarm Marker = IF ( MAX ( fact_series[cusum_first_alarm] ) = 1, MAX ( fact_series[minutes] ), BLANK () )

-- ===== Page 3: Detection lead time (fact_leadtime filtered by slicers: metric, group, alpha) =====
TPR = AVERAGE ( fact_leadtime[estimate] )
TPR Lower (bootstrap 2.5%) = AVERAGE ( fact_leadtime[lo] )
TPR Upper (bootstrap 97.5%) = AVERAGE ( fact_leadtime[hi] )
Churners in Group = MAX ( fact_leadtime[n_churn] )
Retained in Group = MAX ( fact_leadtime[n_retained] )

-- ===== Page 4: Model output and uncertainty =====
Mean Predicted = AVERAGE ( fact_calibration[mean_pred] )
Observed Rate = AVERAGE ( fact_calibration[observed] )
Observed Lower = AVERAGE ( fact_calibration[obs_lo] )
Observed Upper = AVERAGE ( fact_calibration[obs_hi] )
Importance (AUC drop) = AVERAGE ( fact_importance[auc_drop] )
Slice Calibration Gap = AVERAGE ( fact_slices[calibration_gap] )
Slice AUC = AVERAGE ( fact_slices[auc] )
Score Histogram Count = SUM ( fact_score_hist[count] )

-- ===== Page 5: Drift and monitoring =====
PSI = AVERAGE ( fact_drift[psi] )
PSI Warning Line = 0.10
PSI Alert Line = 0.25
Weekly AUC = AVERAGE ( fact_perf_week[auc] )
Weekly Base Rate = AVERAGE ( fact_perf_week[base_rate] )
Weekly Mean Score = AVERAGE ( fact_perf_week[mean_score] )
Weekly Calibration Gap = AVERAGE ( fact_perf_week[calibration_gap] )
Weekly Events = SUM ( fact_perf_week[n] )
CUSUM Alarm Share = AVERAGE ( fact_cusum_week[share_with_cusum_alarm] )
Latest Score PSI =
CALCULATE ( [PSI], fact_drift[feature] = "score",
    FILTER ( ALL ( fact_drift[week] ), fact_drift[week] = MAX ( fact_drift[week] ) ) )
"""

DICTIONARY = [
    ("dim_event", "event_id", "Integer key of one (customer, expiry) event; anonymous."),
    ("dim_event", "period", "Expiry cohort: 2017-02 (train.csv) or 2017-03 (train_v2.csv)."),
    ("dim_event", "outcome", "Churned / Retained using Kaggle's official labels."),
    ("dim_event", "churn_type", "Silent churn = churned and no cancel row within 60 days before expiry; Explicit cancel = churned with such a row; Retained."),
    ("dim_event", "renewal_type", "Auto-renew or Manual, from the last non-cancelled transaction before the cutoff."),
    ("dim_event", "baseline_log_minutes", "Mean log(1+minutes) per day over days 90..61 before expiry."),
    ("dim_event", "model_score / score_band / model_split", "Logistic model probability at cutoff = expiry - 7 days; split train/monitor/excluded."),
    ("dim_event", "lead_days_<family>", "Days before expiry of the first alarm for the best-in-family smoother at 10% FPR (0 = no alarm, -1 = not analysable)."),
    ("fact_series", "t", "Day index 1..L increasing with time (L = day before expiry). Use for DAX trailing windows."),
    ("fact_series", "days_before_expiry", "L - t + 1 (1 = day before expiry)."),
    ("fact_series", "day_offset", "-days_before_expiry (-120 ... -1): use as the X axis so time runs left to right."),
    ("fact_leadtime", "lead_days", "Required lead time in days for the tpr_L* metrics (numeric axis); blank for other metrics."),
    ("fact_slices", "slice_label", "slice_col: level, ready for an axis."),
    ("fact_cusum_week", "share_with_cusum_alarm", "Share of analysable events in that expiry week with a CUSUM alarm."),
    ("fact_series", "minutes", "Minutes listened that day (0 when no log row)."),
    ("fact_series", "sma7_py / ema015_py / lowess14_py / spline28_py", "Python causal smoothers on minutes (end-point estimates)."),
    ("fact_series", "cusum / cusum_threshold / cusum_first_alarm", "CUSUM statistic, its alarm threshold (calibrated so ~10% of retained customers ever alarm) and a 1 on the day of the first alarm."),
    ("fact_cohort_curves", "mean / lo / hi", "Cohort mean with normal-approximation 95% CI over analysable events; log(1+minutes) scale."),
    ("fact_leadtime", "metric", "tpr_L7/14/21/28 = share of churners alarmed at least that many days before expiry; tpr_any; median_lead_detected; fpr_achieved."),
    ("fact_leadtime", "lo / hi", "Customer-level bootstrap 2.5% / 97.5%."),
    ("fact_calibration", "split", "monitor (out of time) or train."),
    ("fact_drift", "psi", "Population stability index vs the training window; 0.10 / 0.25 are conventional rules of thumb."),
    ("fact_slices", "reliable", "False when the slice has too few events/churners for a trustworthy AUC."),
]


def _band(s):
    return pd.cut(s, SCORE_BANDS, labels=SCORE_LABELS, right=False).astype(object)


def _age_band(bd):
    return pd.cut(bd, [14, 20, 30, 40, 60], labels=["15-20", "21-30", "31-40", "41-60"]).astype(object).where(bd.between(15, 60), "unknown")


def export_powerbi(out_dir, ev: pd.DataFrame, X: np.ndarray, has_log: np.ndarray, L: int, bench: pd.DataFrame,
                   model_out: dict, members: Optional[pd.DataFrame] = None, sampling: Optional[pd.DataFrame] = None,
                   n_silent=250, n_explicit=100, n_retained=400, seed: int = 7) -> Dict[str, pd.DataFrame]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ev = ev.reset_index(drop=True).copy()
    base = baseline(X, L)
    ok = np.asarray(has_log, bool) & (base >= MIN_BASELINE)
    scores = model_out["scores"].set_index("event_id")
    ev["model_score"] = scores.loc[ev["event_id"], "score"].to_numpy()
    ev["model_split"] = scores.loc[ev["event_id"], "split"].to_numpy()
    ev["churn_type"] = np.where(ev["is_churn"] == 0, "Retained", np.where(ev["silent"] == 1, "Silent churn", "Explicit cancel"))
    ev["renewal_type"] = np.where(ev["is_auto_renew"] == 1, "Auto-renew", "Manual")
    ev["outcome"] = np.where(ev["is_churn"] == 1, "Churned", "Retained")
    ev["baseline_log_minutes"] = base

    # ---- representative smoothers and first-alarm leads (alpha = 10%) ----
    reg = {s.name: s for s in sm.make_smoothers(include_leaky=False)}
    best = best_per_family(bench, "all", 0.10, "tpr_L14")
    chosen = {r.family: r.method for r in best.itertuples()}
    for fam, name in chosen.items():
        ev[f"lead_days_{fam.lower()}"] = alarm_leads_for(reg[name], X, ev, has_log, L)
    cus_h, S_all, _ = cusum_threshold(X, ev, has_log, L)
    cus_first = sm.cusum_first(S_all, L, cus_h)
    ev["lead_days_cusum"] = np.where(ok, cus_first, -1)

    # ---- dashboard sample ----
    def pick(mask, n):
        idx = np.where(mask & ok)[0]
        return rng.choice(idx, size=min(n, len(idx)), replace=False)
    sel = np.sort(np.concatenate([pick(ev["churn_type"].eq("Silent churn").to_numpy(), n_silent),
                                  pick(ev["churn_type"].eq("Explicit cancel").to_numpy(), n_explicit),
                                  pick(ev["churn_type"].eq("Retained").to_numpy(), n_retained)]))
    dim = ev.iloc[sel].copy()
    if members is not None:
        m = members.drop_duplicates("msno").set_index("msno")
        j = m.reindex(dim["msno"])
        dim["city"] = j["city"].to_numpy(); dim["registered_via"] = j["registered_via"].to_numpy()
        dim["gender"] = j["gender"].fillna("unknown").replace("", "unknown").to_numpy()
        dim["age_band"] = _age_band(j["bd"].reset_index(drop=True)).to_numpy()
    dim["score_band"] = _band(dim["model_score"].fillna(-1).clip(0, 1)).where(dim["model_score"].notna(), "n/a")
    keep = ["event_id", "period", "expiry_date", "outcome", "churn_type", "renewal_type", "plan_days", "baseline_log_minutes",
            "model_score", "score_band", "model_split"] + [c for c in dim.columns if c.startswith("lead_days_")] + \
           [c for c in ("city", "registered_via", "gender", "age_band") if c in dim.columns]
    dim_event = dim[keep].copy()
    dim_event["expiry_date"] = pd.to_datetime(dim_event["expiry_date"]).dt.strftime("%Y-%m-%d")

    # ---- fact_series (minutes scale) ----
    M = np.expm1(X[sel].astype(np.float64)).clip(min=0).astype(np.float32)
    S7, EMA, LOW, SPL = (sm.fir_apply(M, sm.sma_taps(7)), sm.ema(M, 0.15), sm.fir_apply(M, sm.lowess_taps(14)),
                         sm.fir_apply(M, sm.spline_taps(28, 10)))
    CUS, cfirst = S_all[sel], cus_first[sel]
    n = len(sel)
    t = np.tile(np.arange(1, L + 1), n)
    ids = np.repeat(ev["event_id"].to_numpy()[sel], L)
    exp = np.repeat(pd.to_datetime(ev["expiry_date"]).to_numpy()[sel], L)
    cal = pd.to_datetime(exp) - pd.to_timedelta(L - t + 1, unit="D")
    first_flag = np.zeros((n, L), dtype=np.int8)
    for i, k in enumerate(cfirst):
        if k > 0:
            first_flag[i, L - k] = 1
    fs = pd.DataFrame({"event_id": ids, "t": t, "days_before_expiry": L - t + 1, "day_offset": -(L - t + 1), "calendar_date": cal.strftime("%Y-%m-%d"),
                       "minutes": M.ravel(), "x_log1p_minutes": X[sel].ravel(), "sma7_py": S7.ravel(), "ema015_py": EMA.ravel(),
                       "lowess14_py": LOW.ravel(), "spline28_py": SPL.ravel(), "cusum": CUS.ravel(), "cusum_threshold": cus_h, "cusum_first_alarm": first_flag.ravel()})

    # ---- cohort curves over ALL analysable events ----
    cohorts = {}
    ct, rt = ev["churn_type"].to_numpy(), ev["renewal_type"].to_numpy()
    for c in ("Silent churn", "Explicit cancel", "Retained"):
        cohorts[(c, "All")] = ok & (ct == c)
        for r in ("Auto-renew", "Manual"):
            cohorts[(c, r)] = ok & (ct == c) & (rt == r)
    rows = []
    for fam, name in REPRESENTATIVE.items():
        S = reg[name].fn(X)
        for (c, r), mk in cohorts.items():
            if mk.sum() < 30:
                continue
            A = S[mk]
            cnt = np.sum(~np.isnan(A), axis=0)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mu = np.nanmean(A, axis=0)
                se = np.nanstd(A, axis=0, ddof=1) / np.sqrt(np.maximum(cnt, 1))
            rows.append(pd.DataFrame({"cohort": c, "renewal_type": r, "method": name, "days_before_expiry": L - np.arange(L), "day_offset": -(L - np.arange(L)),
                                      "mean": mu, "lo": mu - 1.96 * se, "hi": mu + 1.96 * se, "n": cnt}))
    cohort_curves = pd.concat(rows, ignore_index=True)

    lt = bench.copy()
    lt["is_leaky_reference"] = ~lt["causal"]
    lt["lead_days"] = pd.to_numeric(lt["metric"].str.extract(r"^tpr_L(\d+)$")[0], errors="coerce")
    cal_t = pd.concat([model_out["reliability_monitor"].assign(split="monitor"), model_out["reliability_train"].assign(split="train")])
    sc = model_out["scores"].merge(ev[["event_id", "outcome"]], on="event_id")
    sc = sc[sc["score"].notna() & sc["split"].isin(["train", "monitor"])]
    edges = np.linspace(0, min(1.0, float(sc["score"].quantile(0.995)) + 1e-9), 26)
    hist = []
    for (sp, oc), g in sc.groupby(["split", "outcome"]):
        h, _ = np.histogram(g["score"].clip(upper=edges[-1] - 1e-12), bins=edges)
        hist.append(pd.DataFrame({"split": sp, "outcome": oc, "bin_low": edges[:-1], "bin_high": edges[1:], "count": h}))
    method_dim = pd.DataFrame({"method": sorted(lt["method"].unique())})
    method_dim["family"] = method_dim["method"].map(dict(zip(lt["method"], lt["family"])))
    method_dim["is_leaky_reference"] = method_dim["method"].str.contains("LEAKY")

    tables = {"dim_event": dim_event, "fact_series": fs, "fact_cohort_curves": cohort_curves, "fact_leadtime": lt.drop(columns=["causal"]),
              "fact_calibration": cal_t, "fact_score_hist": pd.concat(hist, ignore_index=True), "fact_importance": model_out["importance"],
              "fact_drift": model_out["drift"], "fact_perf_week": model_out["perf_week"], "dim_method": method_dim}
    if "slices" in model_out:
        sl = model_out["slices"].copy()
        sl.insert(0, "slice_label", sl["slice_col"] + ": " + sl["level"])
        tables["fact_slices"] = sl
    wk = ((pd.to_datetime(ev["expiry_date"]) - pd.Timestamp("2017-02-01")).dt.days // 7).clip(lower=0).to_numpy()
    cw = []
    for w in sorted(set(wk[ok])):
        mk = ok & (wk == w)
        cw.append({"week": int(w), "week_start": (pd.Timestamp("2017-02-01") + pd.Timedelta(days=7 * int(w))).strftime("%Y-%m-%d"),
                   "n": int(mk.sum()), "share_with_cusum_alarm": float((cus_first[mk] > 0).mean())})
    tables["fact_cusum_week"] = pd.DataFrame(cw)
    if sampling is not None:
        tables["fact_sampling"] = sampling
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "theme.json").write_text(json.dumps(theme_json(), indent=2))
    (out / "dax_measures.txt").write_text(DAX)
    pd.DataFrame(DICTIONARY, columns=["table", "column", "meaning"]).to_csv(out / "data_dictionary.csv", index=False)
    return tables
