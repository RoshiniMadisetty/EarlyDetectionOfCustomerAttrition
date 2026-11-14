"""End-to-end: synthetic raw files -> ingest -> every pipeline step -> exported tables, figures, usability package."""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silentchurn import ingest, pipeline
from silentchurn.config import Config
from silentchurn.synth import generate

L = 120


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    generate(raw, n_members=5000, seed=5)
    out = tmp_path_factory.mktemp("out")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ingest.run(Config(raw, out, 100, memory_limit="1GB", threads=1), verbose=False)
    work = tmp_path_factory.mktemp("work")
    sd = str(out / "sample100")
    pipeline.main(["all", "--sample-dir", sd, "--work", str(work), "--B", "10", "--participants", "8", "--memory", "1GB", "--threads", "1"])
    return work, sd


def test_every_step_wrote_its_outputs(run):
    work, _ = run
    for f in ["validate.json", "events.parquet", "series.npz", "benchmark.csv", "benchmark_counts.csv", "model_out.pkl",
              "powerbi/dim_event.csv", "powerbi/theme.json", "powerbi/dax_measures.txt", "figures/dashboard_fallback.html",
              "usability/participant_sheets.csv", "usability/answer_key.csv", "usability/response_template.csv"]:
        assert (work / f).exists(), f
    assert len(list((work / "figures").glob("page*.png"))) == 5
    assert len(list((work / "usability/images").glob("*.png"))) == 96


def test_steps_are_resumable(run, capsys):
    work, sd = run
    pipeline.main(["events", "--sample-dir", sd, "--work", str(work), "--memory", "1GB", "--threads", "1"])
    assert "skipped" in capsys.readouterr().out


def test_powerbi_tables_are_consistent_and_anonymous(run):
    work, _ = run
    p = work / "powerbi"
    dim, fs = pd.read_csv(p / "dim_event.csv"), pd.read_csv(p / "fact_series.csv")
    assert dim["event_id"].is_unique and set(fs["event_id"]) == set(dim["event_id"])
    assert len(fs) == len(dim) * L and fs.groupby("event_id")["t"].nunique().eq(L).all()
    assert set(dim["churn_type"]) <= {"Silent churn", "Explicit cancel", "Retained"} and set(dim["outcome"]) == {"Churned", "Retained"}
    for f in p.glob("*.csv"):
        assert "msno" not in pd.read_csv(f, nrows=2).columns, f
    for need in ["fact_cohort_curves", "fact_leadtime", "fact_calibration", "fact_importance", "fact_drift", "fact_perf_week",
                 "fact_slices", "fact_sampling", "fact_score_hist", "fact_cusum_week", "dim_method", "data_dictionary"]:
        assert len(pd.read_csv(p / f"{need}.csv")) > 0, need
    assert (fs["day_offset"] == -fs["days_before_expiry"]).all() and fs["day_offset"].min() == -L and fs["day_offset"].max() == -1
    sl = pd.read_csv(p / "fact_slices.csv"); assert (sl["slice_label"] == sl["slice_col"] + ": " + sl["level"].astype(str)).all()
    lt = pd.read_csv(p / "fact_leadtime.csv")
    assert set(lt.loc[lt["lead_days"].notna(), "lead_days"]) == {7, 14, 21, 28} and lt.loc[lt["lead_days"].notna(), "metric"].str.startswith("tpr_L").all()
    assert lt["is_leaky_reference"].sum() > 0 and lt.loc[lt["method"].str.contains("LEAKY"), "is_leaky_reference"].all()


def test_python_moving_average_equals_the_dax_definition(run):
    """DAX MA7 = mean of the 7 rows t-6..t (blank if fewer than 7). Re-create that definition in pandas and compare."""
    work, _ = run
    fs = pd.read_csv(work / "powerbi/fact_series.csv").sort_values(["event_id", "t"])
    dax_like = fs.groupby("event_id")["minutes"].transform(lambda s: s.rolling(7, min_periods=7).mean())
    assert np.allclose(dax_like.to_numpy(), fs["sma7_py"].to_numpy(), atol=1e-3, equal_nan=True)


def test_theme_is_valid_json_with_colour_blind_safe_palette(run):
    work, _ = run
    t = json.loads((work / "powerbi/theme.json").read_text())
    assert all(re.fullmatch(r"#[0-9A-F]{6}", c) for c in t["dataColors"]) and len(set(t["dataColors"])) == len(t["dataColors"])
    assert t["dataColors"][:3] == ["#0072B2", "#D55E00", "#009E73"]          # Okabe-Ito blue / vermillion / green
    assert t["maximum"] == "#BD0026" and t["minimum"] == "#FFFFB2"          # single sequential risk scale
    assert t["textClasses"]["label"]["fontSize"] >= 11


def test_dax_file_has_a_measure_for_every_page_and_balanced_brackets(run):
    work, _ = run
    d = (work / "powerbi/dax_measures.txt").read_text()
    for name in ["Churn Rate (dashboard sample)", "MA7 DAX", "MA7 Difference (DAX - Python)", "TPR Lower (bootstrap 2.5%)",
                 "Observed Rate", "PSI Alert Line", "Weekly AUC", "Sample Churn Rate", "Max Abs MA7 Difference", "CUSUM Alarm Share", "Latest Score PSI"]:
        assert name in d
    assert d.count("(") == d.count(")")


def test_usability_package_matches_design(run):
    work, _ = run
    sh = pd.read_csv(work / "usability/participant_sheets.csv")
    assert sh["participant"].nunique() == 8 and sh.groupby("participant").size().eq(24).all()
    assert (work / "usability/response_template.csv").read_text().startswith("participant,condition,trial_id,n_correct,time_s,confidence")


def test_usability_analyze_cli(run, tmp_path, capsys):
    from silentchurn import usability as us
    work, sd = run
    f = tmp_path / "resp.csv"; us.simulate_study(10, seed=2).to_csv(f, index=False)
    pipeline.main(["usability-analyze", "--sample-dir", sd, "--work", str(work), "--results", str(f)])
    assert (work / "usability/analysis_contrasts.csv").exists() and "mean_diff" in capsys.readouterr().out
