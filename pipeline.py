"""Pipeline driver. Each step is resumable: it skips if its output already exists (use --force to redo).

  python -m silentchurn.pipeline --sample-dir DIR --work DIR [--raw RAW_DIR] STEP [--force]
  STEP: validate | events | series | benchmark | model | export | figures | usability | usability-analyze | all

Outputs in --work:  validate.json, events.parquet, series.npz, benchmark.csv, benchmark_counts.csv, model_out.pkl,
                    powerbi/ (CSV tables, theme.json, dax_measures.txt), figures/ (PNG + fallback HTML)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import benchmark as bm, events as evm, exports, model as md, series, smoothers as sm

L_DEFAULT = 120


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _con(a):
    return evm.connect(a.memory, a.threads, str(Path(a.work) / "duckdb_tmp"))


def step_validate(a):
    out = Path(a.work) / "validate.json"
    if out.exists() and not a.force:
        _log("validate: skipped (exists)"); print(out.read_text()); return
    con = _con(a)
    res = [evm.validate_official(con, a.sample_dir, "2017-02", hs, "union") for hs in ("v1", "union")]
    out.write_text(json.dumps(res, indent=2, default=str))
    _log("OFFICIAL-LABEL CROSS-CHECK (Feb-2017 cohort re-derived from transactions vs train.csv)")
    print(pd.DataFrame(res).T.to_string())
    print("\nNote: the March cohort (train_v2) cannot be re-derived (needs April transactions); its labels come from train_v2.csv.")


def step_events(a):
    p = Path(a.work) / "events.parquet"
    if p.exists() and not a.force:
        _log("events: skipped (exists)"); return pd.read_parquet(p)
    ev = evm.build_events(_con(a), a.sample_dir, a.history_source)
    ev.to_parquet(p, index=False)
    _log(f"events: {len(ev):,} rows -> {p}")
    print(evm.events_report(ev).to_string())
    return ev


def step_series(a):
    p = Path(a.work) / "series.npz"
    ev = pd.read_parquet(Path(a.work) / "events.parquet")
    if p.exists() and not a.force:
        _log("series: skipped (exists)"); return
    X, has_log, info = series.build_series(_con(a), a.sample_dir, ev, a.L)
    outside = ~ev["in_target_month"].fillna(False).to_numpy().astype(bool)
    info["events_outside_target_month_excluded"] = int(outside.sum())
    has_log = has_log & ~outside          # 'analysable' requires listening data AND an expiry inside the target month
    series.save_series(p, X, has_log, ev["event_id"].to_numpy(), a.L)
    _log(f"series: matrix {X.shape}, {X.nbytes/1e6:.0f} MB in memory -> {p}")
    print(json.dumps(info, indent=2))


def step_benchmark(a):
    p = Path(a.work) / "benchmark.csv"
    if p.exists() and not a.force:
        _log("benchmark: skipped (exists)"); return
    ev = pd.read_parquet(Path(a.work) / "events.parquet")
    X, has_log, ids, L = series.load_series(Path(a.work) / "series.npz")
    sms = sm.make_smoothers(include_leaky=True)
    _log(f"benchmark: {len(sms)} smoothers, bootstrap B={a.B}; automated causality test runs first")
    out = bm.run_benchmark(X, ev, has_log, sms, L, B=a.B, seed=a.seed)
    out["results"].to_csv(p, index=False); out["counts"].to_csv(Path(a.work) / "benchmark_counts.csv", index=False)
    print(out["counts"].to_string(index=False))
    r = out["results"]
    for g in ("all", "auto_renew", "manual_renew", "silent_only"):
        d = r[(r.group == g) & (r.alpha == 0.10) & (r.metric == "tpr_L14")].sort_values("estimate", ascending=False)
        if len(d):
            print(f"\nTPR at lead >= 14 days, FPR 10%, group={g} (n_churn={d.n_churn.iloc[0]}):")
            print(d[["method", "causal", "estimate", "lo", "hi"]].head(8).round(3).to_string(index=False))


def step_model(a):
    p = Path(a.work) / "model_out.pkl"
    if p.exists() and not a.force:
        _log("model: skipped (exists)"); return
    ev = pd.read_parquet(Path(a.work) / "events.parquet")
    X, has_log, ids, L = series.load_series(Path(a.work) / "series.npz")
    mo = md.run_model(X, ev, has_log, L, h=a.h, exclude_shared=a.exclude_shared)
    members = _members(a)
    if members is not None:
        mo["slices"] = md.build_slices(mo["split_frame"], members)
    mo.pop("model", None)
    pickle.dump(mo, open(p, "wb"))
    _log("model: leakage test passed")
    print(json.dumps({**mo["metrics"], "n_train": mo["n_train"], "n_monitor": mo["n_monitor"],
                      "n_excluded_shared_customers": mo["n_excluded_shared_customers"],
                      "n_monitor_events_from_training_customers": mo["n_monitor_events_from_training_customers"]}, indent=2))
    print(mo["importance"].round(4).to_string(index=False))
    print(mo["perf_week"].round(3).to_string(index=False))


def _members(a):
    import duckdb
    p = Path(a.sample_dir) / "members.parquet"
    return duckdb.sql(f"SELECT msno, city, bd, gender, registered_via FROM read_parquet('{p}')").df() if p.exists() else None


def _sampling(a):
    m = Path(a.sample_dir) / "manifest.json"
    if not m.exists():
        return None
    rep = json.loads(m.read_text())["report"]["train"]
    rows = [{"file": k, "sample_n": v["sample_rows"], "sample_churn_rate": v["sample_churn_rate"], "sample_ci_lo": v["sample_ci95"][0],
             "sample_ci_hi": v["sample_ci95"][1], "full_n": v.get("full_rows"), "full_churn_rate": v.get("full_churn_rate")} for k, v in rep.items()]
    return pd.DataFrame(rows)


def step_export(a):
    ev = pd.read_parquet(Path(a.work) / "events.parquet")
    X, has_log, ids, L = series.load_series(Path(a.work) / "series.npz")
    bench = pd.read_csv(Path(a.work) / "benchmark.csv")
    bench["causal"] = bench["causal"].astype(bool)
    mo = pickle.load(open(Path(a.work) / "model_out.pkl", "rb"))
    t = exports.export_powerbi(Path(a.work) / "powerbi", ev, X, has_log, L, bench, mo, _members(a), _sampling(a))
    _log("export: Power BI tables written to " + str(Path(a.work) / "powerbi"))
    for k, v in t.items():
        print(f"  {k:22s} {len(v):>9,} rows x {v.shape[1]} cols")


def step_figures(a):
    from . import figures
    ev = pd.read_parquet(Path(a.work) / "events.parquet")
    X, has_log, ids, L = series.load_series(Path(a.work) / "series.npz")
    bench = pd.read_csv(Path(a.work) / "benchmark.csv"); bench["causal"] = bench["causal"].astype(bool)
    mo = pickle.load(open(Path(a.work) / "model_out.pkl", "rb"))
    files = figures.make_all(Path(a.work) / "figures", ev, X, has_log, L, bench, mo, _sampling(a))
    _log("figures: " + ", ".join(Path(f).name for f in files))


def step_usability(a):
    from . import usability as us
    out = Path(a.work) / "usability"
    if (out / "participant_sheets.csv").exists() and not a.force:
        _log("usability: skipped (exists)"); return
    out.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(Path(a.work) / "events.parquet")
    X, has_log, ids, L = series.load_series(Path(a.work) / "series.npz")
    sets = us.make_trials(ev, X, has_log, L, seed=a.seed)
    sheets = us.participant_sheets(sets, a.participants)
    h_c, _, _ = bm.cusum_threshold(X, ev, has_log, L)
    files = us.render_all(out / "images", sets, X, L, cusum_h=h_c)
    us.answer_key(sets).to_csv(out / "answer_key.csv", index=False)
    sheets.to_csv(out / "participant_sheets.csv", index=False)
    sheets.assign(n_correct="", time_s="", confidence="")[["participant", "condition", "trial_id", "n_correct", "time_s", "confidence"]].to_csv(out / "response_template.csv", index=False)
    _log(f"usability: {len(sets)} trial sets, {len(files)} stimulus images, {a.participants} participant sheets -> {out}")


def step_usability_analyze(a):
    from . import usability as us
    res = pd.read_csv(a.results)
    out = us.analyze(res)
    d = Path(a.work) / "usability"; d.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_csv(d / f"analysis_{k}.csv", index=False)
    print(out["summary"].round(3).to_string(index=False)); print(out["contrasts"].round(3).to_string(index=False)); print(out["omnibus"].round(4).to_string(index=False))


STEPS = {"validate": step_validate, "events": step_events, "series": step_series, "benchmark": step_benchmark,
         "model": step_model, "export": step_export, "figures": step_figures, "usability": step_usability}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS) + ["usability-analyze", "all"])
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--L", type=int, default=L_DEFAULT)
    ap.add_argument("--h", type=int, default=7, help="model cutoff: days before expiry")
    ap.add_argument("--B", type=int, default=200, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--history-source", default=None, choices=["v1", "v2", "union"], help="default: v1 for the Feb cohort, union for Mar")
    ap.add_argument("--exclude-shared", action="store_true", help="model step: drop monitoring events of customers seen in training")
    ap.add_argument("--participants", type=int, default=12)
    ap.add_argument("--results", default=None, help="CSV of usability responses for usability-analyze")
    ap.add_argument("--memory", default="8GB")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    Path(a.work).mkdir(parents=True, exist_ok=True)
    if a.step == "usability-analyze":
        step_usability_analyze(a); return
    for s in (list(STEPS) if a.step == "all" else [a.step]):
        _log(f"=== step: {s} ===")
        STEPS[s](a)


if __name__ == "__main__":
    main()
