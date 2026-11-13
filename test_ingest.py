"""Tests for silentchurn.ingest on a synthetic dataset in the real KKBox file format.

These tests prove the plumbing (sampling rule, full-history retention, resumability, dtypes, report).
They say NOTHING about the real 29 GB data: run step 1 on Colab and read its printed report.
"""
import warnings
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from silentchurn import ingest
from silentchurn.config import Config
from silentchurn.synth import generate

PCT = 10
TABLES = ["train", "members", "transactions", "transactions_v2", "user_logs", "user_logs_v2"]


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    generate(d, n_members=1500, seed=11)
    return d


@pytest.fixture(scope="module")
def built(raw, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    cfg = Config(raw, out, PCT, memory_limit="1GB", threads=1)
    manifest = ingest.run(cfg, verbose=False)
    return cfg, manifest


def _raw_df(raw, name):
    fn = {"members": "members_v3.csv"}.get(name, f"{name}.csv")
    return pd.read_csv(Path(raw) / fn)


def _pq(cfg, name):
    return pd.read_parquet(cfg.sample_dir / f"{name}.parquet")


def test_all_tables_written_and_no_tmp_left(built):
    cfg, manifest = built
    for t in TABLES:
        assert (cfg.sample_dir / f"{t}.parquet").exists()
    assert not list(cfg.sample_dir.glob("*.tmp"))
    assert (cfg.sample_dir / "manifest.json").exists()


def test_sql_sampling_rule_matches_independent_python_rule(raw, built):
    cfg, _ = built
    members = _raw_df(raw, "members")
    expected = {m for m in members["msno"] if ingest.in_sample(m, PCT)}
    got = set(_pq(cfg, "members")["msno"])
    assert got == expected and len(expected) > 0


def test_sample_fraction_is_close_to_pct(raw, built):
    cfg, _ = built
    frac = len(_pq(cfg, "members")) / len(_raw_df(raw, "members"))
    assert abs(frac - PCT / 100) < 0.03  # 1500 customers -> binomial sd ~0.8 pp


@pytest.mark.parametrize("name", ["train", "transactions", "transactions_v2", "user_logs", "user_logs_v2"])
def test_only_sampled_customers_and_full_history_kept(raw, built, name):
    cfg, _ = built
    r = _raw_df(raw, name)
    keep = r[r["msno"].map(lambda m: ingest.in_sample(m, PCT))]
    p = _pq(cfg, name)
    assert len(p) == len(keep)
    assert set(p["msno"]) == set(keep["msno"])
    # per-customer row counts identical => full history kept
    a = keep.groupby("msno").size().sort_index()
    b = p.groupby("msno").size().sort_index()
    pd.testing.assert_series_equal(a, b, check_names=False, check_dtype=False)


def test_customer_consistent_across_tables(raw, built):
    cfg, _ = built
    sampled = set(_pq(cfg, "members")["msno"])
    for name in ["train", "transactions", "user_logs"]:
        assert set(_pq(cfg, name)["msno"]) <= sampled


def test_date_columns_are_dates_and_in_range(built):
    cfg, _ = built
    tx = _pq(cfg, "transactions")
    lg = _pq(cfg, "user_logs")
    for s in (tx["transaction_date"], tx["membership_expire_date"], lg["date"]):
        assert str(s.dtype).startswith("datetime64") or s.dtype == object  # date32 -> object or datetime64
    assert pd.to_datetime(tx["transaction_date"]).min() >= pd.Timestamp("2015-01-01")
    assert pd.to_datetime(lg["date"]).max() <= pd.Timestamp("2017-02-28")
    assert pd.to_datetime(tx["transaction_date"]).max() <= pd.Timestamp("2017-02-28")
    assert pd.to_datetime(_pq(cfg, "user_logs_v2")["date"]).min() >= pd.Timestamp("2017-03-01")
    assert tx["transaction_date"].notna().all()


def test_resume_skips_finished_and_rebuilds_only_missing(raw, built):
    cfg, _ = built
    mtimes = {t: (cfg.sample_dir / f"{t}.parquet").stat().st_mtime_ns for t in TABLES}
    con = ingest.connect(cfg)
    assert ingest.convert_table(con, cfg, "members")["status"] == "skipped"
    victim = cfg.sample_dir / "transactions_v2.parquet"
    victim.unlink()
    again = ingest.run(cfg, verbose=False)
    status = {r["table"]: r["status"] for r in again["tables"]}
    assert status["transactions_v2"] == "written"
    assert status["members"] == status["user_logs"] == "skipped"
    for t in TABLES:
        if t != "transactions_v2":
            assert (cfg.sample_dir / f"{t}.parquet").stat().st_mtime_ns == mtimes[t]
    assert victim.exists()


def test_report_churn_rate_matches_pandas_and_has_ci(raw, built):
    cfg, manifest = built
    rep = manifest["report"]["train"]["train"]
    full = _raw_df(raw, "train")
    assert rep["full_rows"] == len(full)
    assert rep["full_churn_rate"] == pytest.approx(full["is_churn"].mean())
    samp = full[full["msno"].map(lambda m: ingest.in_sample(m, PCT))]
    assert rep["sample_rows"] == len(samp)
    assert rep["sample_churn_rate"] == pytest.approx(samp["is_churn"].mean())
    lo, hi = rep["sample_ci95"]
    assert lo <= rep["sample_churn_rate"] <= hi
    assert rep["abs_diff_pp"] == pytest.approx(100 * abs(rep["sample_churn_rate"] - rep["full_churn_rate"]))
    cov = manifest["report"]["coverage"]
    assert "user_logs" in cov and "date" in cov["user_logs"]


def test_wilson_ci_sane():
    lo, hi = ingest.wilson_ci(10, 100)
    assert 0.05 < lo < 0.10 < hi < 0.18
    assert all(x != x for x in ingest.wilson_ci(0, 0))  # NaN


def test_pct_validation(tmp_path):
    with pytest.raises(ValueError):
        Config(tmp_path, tmp_path, 0).validate()
    with pytest.raises(ValueError):
        Config(tmp_path, tmp_path, 101).validate()
    with pytest.warns(UserWarning):
        Config(tmp_path, tmp_path, 3).validate()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Config(tmp_path, tmp_path, 10).validate()


def test_missing_required_file_gives_clear_error(raw, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="members_v3.csv"):
        ingest.run(Config(empty, tmp_path / "o", 10), verbose=False)


def test_gzip_csv_is_accepted(raw, tmp_path):
    import gzip
    import shutil
    d = tmp_path / "gz"
    d.mkdir()
    for f in Path(raw).glob("*.csv"):
        if f.name.startswith("_"):
            continue
        with open(f, "rb") as fi, gzip.open(d / (f.name + ".gz"), "wb") as fo:
            shutil.copyfileobj(fi, fo)
    cfg = Config(d, tmp_path / "o", PCT, memory_limit="1GB", threads=1)
    man = ingest.run(cfg, verbose=False)
    assert man["report"]["train"]["train"]["sample_rows"] > 0
