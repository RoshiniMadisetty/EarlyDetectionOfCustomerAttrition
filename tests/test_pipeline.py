"""Correctness tests for the pipeline, run on small synthetic data.

  pytest -q tests

The key test is test_features_ignore_data_after_cutoff: it rewrites every
transaction and log row dated after an event's cutoff and checks that every
feature of that event is unchanged. If any feature leaked future information,
it would change.
"""
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from src import models as M
from src import value_of_lead_time as V
from src.build_dataset import build
from src.config import FEATURES
from src.drift import psi
from src.prepare_data import prepare
from tests.make_synthetic import simulate

MONTHS = "2016-10:2017-02"


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("churn")
    raw, pq = root / "raw", root / "pq"
    raw.mkdir()
    mem, tx, logs = simulate(1500, seed=11)
    mem.to_csv(raw / "members.csv", index=False)
    tx.to_csv(raw / "transactions.csv", index=False)
    logs.to_csv(raw / "user_logs.csv", index=False)
    prepare(raw, pq)
    return {"root": root, "pq": pq}


@pytest.fixture(scope="session")
def dataset(env):
    out = env["root"] / "ds" / "dataset.parquet"
    build(env["pq"], out, [14, 7], MONTHS)
    return out


def ts(x):
    return pd.to_datetime(x)


def test_labels_and_eligibility_match_independent_definition(env, dataset):
    tx = pd.read_parquet(env["pq"] / "transactions.parquet")
    tx["transaction_date"] = ts(tx["transaction_date"])
    tx["membership_expire_date"] = ts(tx["membership_expire_date"])
    ds = pd.read_parquet(dataset).query("horizon == 14").sample(300, random_state=0)
    for r in ds.itertuples():
        a = tx[tx["tx_id"] == r.event_id].iloc[0]
        E = a["membership_expire_date"]
        C = E - pd.Timedelta(days=14)
        others = tx[(tx["msno"] == r.msno) & (tx["is_cancel"] == 0) & (tx["tx_id"] != r.event_id)]
        later = others[others["membership_expire_date"] > E]
        assert a["transaction_date"] <= C, "own purchase must precede the cutoff"
        assert not (later["transaction_date"] <= C).any(), "already extended before the cutoff"
        renewed = (later["transaction_date"] <= E + pd.Timedelta(days=30)).any()
        assert int(not renewed) == r.is_churn
        assert ts(r.cutoff_date) == C and ts(r.expire_date) == E


def test_features_match_independent_recomputation(env, dataset):
    logs = pd.read_parquet(env["pq"] / "user_logs.parquet")
    logs["date"] = ts(logs["date"])
    daily = logs.groupby(["msno", "date"], as_index=False)["total_secs"].sum()
    daily["total_secs"] = daily["total_secs"].clip(upper=86400)
    ds = pd.read_parquet(dataset).query("horizon == 14").sample(150, random_state=1)
    for r in ds.itertuples():
        C = ts(r.cutoff_date)
        u = daily[(daily["msno"] == r.msno) & (daily["date"] <= C) &
                  (daily["date"] > C - pd.Timedelta(days=30))]
        series = np.zeros(30)
        for d, s in zip(u["date"], u["total_secs"]):
            series[(d - (C - pd.Timedelta(days=29))).days] = s
        assert r.secs_30 == pytest.approx(series.sum())
        assert r.secs_7 == pytest.approx(series[-7:].sum())
        assert r.active_days_30 == (series > 0).sum() or r.active_days_30 == len(u)
        assert r.active_days_7 == len(u[u["date"] > C - pd.Timedelta(days=7)])
        slope = np.polyfit(np.arange(30), series, 1)[0]
        assert r.usage_slope_30 == pytest.approx(slope, rel=1e-6, abs=1e-6)
        active = np.flatnonzero(np.isin(np.arange(30), (u["date"] - (C - pd.Timedelta(days=29))).dt.days))
        if len(active) == 0:
            assert r.days_since_last_active == 30 and r.max_inactivity_gap_30 == 30
        else:
            gaps = [active[0], 29 - active[-1]] + list(np.diff(active) - 1)
            assert r.max_inactivity_gap_30 == max(gaps)
            assert r.days_since_last_active == 29 - active[-1]


@pytest.mark.parametrize("h", [14, 7])
def test_features_ignore_data_after_cutoff(env, h):
    """Rewrite all data dated after each event's cutoff; its features must not change."""
    root, pq = env["root"], env["pq"]
    base_out = root / f"leak_base_{h}" / "ds.parquet"
    build(pq, base_out, [h], MONTHS)
    base = pd.read_parquet(base_out)
    counts = base.groupby("msno")["event_id"].transform("count")
    single = base[counts == 1][["msno", "cutoff_date"]].rename(columns={"cutoff_date": "C"})
    assert len(single) > 200

    pert = root / f"pq_perturbed_{h}"
    if pert.exists():
        shutil.rmtree(pert)
    pert.mkdir()
    shutil.copy(pq / "members.parquet", pert / "members.parquet")
    con = duckdb.connect()
    con.register("single", single)
    P = lambda n: (pq / f"{n}.parquet").as_posix()
    con.execute(f"""
        COPY (
          SELECT t.tx_id, t.msno,
                 CASE WHEN s.C IS NOT NULL AND t.transaction_date > s.C THEN 99 ELSE t.payment_method_id END AS payment_method_id,
                 CASE WHEN s.C IS NOT NULL AND t.transaction_date > s.C THEN 7 ELSE t.payment_plan_days END AS payment_plan_days,
                 t.plan_list_price,
                 CASE WHEN s.C IS NOT NULL AND t.transaction_date > s.C THEN 999 ELSE t.actual_amount_paid END AS actual_amount_paid,
                 CASE WHEN s.C IS NOT NULL AND t.transaction_date > s.C THEN 1 - t.is_auto_renew ELSE t.is_auto_renew END AS is_auto_renew,
                 t.transaction_date, t.membership_expire_date,
                 CASE WHEN s.C IS NOT NULL AND t.transaction_date > s.C THEN 1 - t.is_cancel ELSE t.is_cancel END AS is_cancel
          FROM read_parquet('{P('transactions')}') t LEFT JOIN single s ON s.msno = t.msno
        ) TO '{(pert / 'transactions.parquet').as_posix()}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
          SELECT l.msno, l.date, l.num_25, l.num_50, l.num_75, l.num_985,
                 CASE WHEN s.C IS NOT NULL AND l.date > s.C THEN l.num_100 * 5 + 50 ELSE l.num_100 END AS num_100,
                 CASE WHEN s.C IS NOT NULL AND l.date > s.C THEN l.num_unq * 3 + 9 ELSE l.num_unq END AS num_unq,
                 CASE WHEN s.C IS NOT NULL AND l.date > s.C THEN LEAST(l.total_secs * 4 + 1000, 86400) ELSE l.total_secs END AS total_secs
          FROM read_parquet('{P('user_logs')}') l LEFT JOIN single s ON s.msno = l.msno
        ) TO '{(pert / 'user_logs.parquet').as_posix()}' (FORMAT PARQUET)
    """)
    changed = con.execute(f"""
        SELECT count(*) FROM read_parquet('{(pert / 'user_logs.parquet').as_posix()}') a
        JOIN read_parquet('{P('user_logs')}') b USING (msno, date)
        WHERE a.total_secs <> b.total_secs
    """).fetchone()[0]
    assert changed > 100, "perturbation did not alter enough future rows"

    pert_out = root / f"leak_pert_{h}" / "ds.parquet"
    build(pert, pert_out, [h], MONTHS)
    after = pd.read_parquet(pert_out)

    keys = ["event_id", "horizon"]
    ids = base.merge(single[["msno"]].drop_duplicates(), on="msno")[keys]
    a = base.merge(ids, on=keys).sort_values(keys).reset_index(drop=True)
    b = after.merge(ids, on=keys, how="inner").sort_values(keys).reset_index(drop=True)
    assert len(a) == len(b) == len(ids), "perturbation changed WHO is scored"
    cols = FEATURES + ["segment", "renewal_value", "cutoff_date", "expire_date"]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, rtol=1e-9, atol=1e-9)


def test_split_is_out_of_time(dataset):
    df = M.load_dataset(dataset, horizon=14)
    tr, va, te, info = M.split_by_month(df)
    assert tr["expire_month"].max() < va["expire_month"].min() < te["expire_month"].min()
    assert len(tr) and len(va) and len(te)


def test_common_cohort_flag(dataset):
    df = pd.read_parquet(dataset)
    per_h = df.groupby("event_id")["horizon"].nunique()
    flagged = df.groupby("event_id")["in_common_cohort"].first()
    assert (flagged == (per_h == 2)).all()


def test_calibrator_and_conformal_coverage(dataset):
    df = M.load_dataset(dataset, horizon=14)
    tr, va, te, _ = M.split_by_month(df)
    _, pv, pt, _ = M.fit_predict("xgb", tr, va, te, FEATURES)
    cal = M.RiskCalibrator(alpha=0.10).fit(pv, va["is_churn"].to_numpy())
    p = cal.transform(pt)
    assert p.min() >= M.EPS and p.max() <= 1 - M.EPS
    assert cal.coverage(p, te["is_churn"].to_numpy()) >= 0.85
    lo, hi = cal.interval(p)
    assert (lo >= 0).all() and (hi <= 1).all() and (lo <= hi).all()
    assert set(cal.risk_band(p)) <= {"Low", "Medium", "High", "Critical"}


def test_value_of_lead_time_matches_bruteforce(tmp_path):
    preds = pd.DataFrame({
        "horizon": [14] * 10,
        "p_cal": np.linspace(0.9, 0.05, 10),
        "is_churn": [1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
        "renewal_value": [100.0, 150.0, 100.0, 50.0, 100.0, 100.0, 100.0, 120.0, 100.0, 100.0]})
    p = tmp_path / "pred.parquet"
    preds.to_parquet(p)

    class A:
        predictions, out, h_ref = str(p), str(tmp_path), None

    V.main(A)
    res = pd.read_csv(tmp_path / "value_of_lead_time.csv")
    row = res[(res.lead_effect == "flat") & (res.p_save == 0.3) &
              (res.offer_cost_frac == 0.2) & (res.k_pct == 30)].iloc[0]
    top = preds.sort_values("p_cal", ascending=False).head(3)       # ceil(0.3 * 10) = 3
    expected = (0.3 * (top.renewal_value * top.is_churn).sum()
                - 0.2 * top.renewal_value.sum())
    assert row.n_contacted == 3
    assert row.profit == pytest.approx(expected)
    rand = 3 * (0.3 * (preds.renewal_value * preds.is_churn).mean()
                - 0.2 * preds.renewal_value.mean())
    assert row.profit_random == pytest.approx(rand)


def test_psi_behaviour():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, rng.normal(size=5000)) < 0.05
    assert psi(x, rng.normal(loc=1.0, size=5000)) > 0.25


# ----------------------------------------------------------------------------- real-file robustness
def _make_variant(raw_dir: Path, var_dir: Path):
    """Same data, but with different column names, ISO dates, timestamps, and the user logs
    split in two files whose date ranges OVERLAP (as a cumulative v2 file would)."""
    var_dir.mkdir(parents=True, exist_ok=True)
    iso = lambda s: pd.to_datetime(s.astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    mem = pd.read_csv(raw_dir / "members.csv")
    mem["registration_init_time"] = iso(mem["registration_init_time"])
    mem.rename(columns={"msno": "user_id"}).to_csv(var_dir / "members_v3.csv", index=False)

    tx = pd.read_csv(raw_dir / "transactions.csv")
    tx["transaction_date"] = iso(tx["transaction_date"])
    tx["membership_expire_date"] = iso(tx["membership_expire_date"]) + " 00:00:00"   # timestamp
    tx.rename(columns={"msno": "user_id", "transaction_date": "start_date",
                       "membership_expire_date": "expiry_date"}).to_csv(
        var_dir / "transactions.csv", index=False)

    lg = pd.read_csv(raw_dir / "user_logs.csv")
    lg["date"] = iso(lg["date"])
    lg = lg.rename(columns={"msno": "user_id", "date": "log_date", "total_secs": "listen_secs"})
    lg[lg["log_date"] <= "2017-02-28"].to_csv(var_dir / "user_logs.csv", index=False)
    lg[lg["log_date"] >= "2017-02-15"].to_csv(var_dir / "user_logs_v2.csv", index=False)  # overlap
    return {"members": {"user_id": "msno"},
            "transactions": {"user_id": "msno", "start_date": "transaction_date",
                             "expiry_date": "membership_expire_date"},
            "user_logs": {"user_id": "msno", "log_date": "date", "listen_secs": "total_secs"}}


@pytest.fixture(scope="session")
def variant(tmp_path_factory):
    root = tmp_path_factory.mktemp("variant")
    raw, var = root / "raw", root / "var"
    raw.mkdir()
    mem, tx, logs = simulate(500, seed=5)
    mem.to_csv(raw / "members.csv", index=False)
    tx.to_csv(raw / "transactions.csv", index=False)
    logs.to_csv(raw / "user_logs.csv", index=False)
    cmap = _make_variant(raw, var)
    prepare(raw, root / "pq_ref")
    prepare(var, root / "pq_var", column_map=cmap)
    return {"root": root, "raw": raw, "var": var, "cmap": cmap,
            "ref": root / "pq_ref", "got": root / "pq_var"}


def test_column_map_iso_dates_and_overlapping_logs_give_identical_tables(variant):
    con = duckdb.connect()
    q = lambda d, t: f"SELECT * FROM read_parquet('{(d / t).as_posix()}')"
    for t in ("members.parquet", "transactions.parquet"):
        a = con.execute(q(variant["ref"], t)).fetchdf().sort_values(
            list(con.execute(q(variant["ref"], t)).fetchdf().columns)).reset_index(drop=True)
        b = con.execute(q(variant["got"], t)).fetchdf()[a.columns].sort_values(
            list(a.columns)).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
    la = con.execute(f"SELECT count(*), sum(total_secs), min(date), max(date) FROM "
                     f"read_parquet('{(variant['ref'] / 'user_logs*.parquet').as_posix()}')").fetchone()
    lb = con.execute(f"SELECT count(*), sum(total_secs), min(date), max(date) FROM "
                     f"read_parquet('{(variant['got'] / 'user_logs*.parquet').as_posix()}')").fetchone()
    assert la[0] == lb[0] and la[1] == pytest.approx(lb[1]) and la[2:] == lb[2:]
    assert len(list(variant["got"].glob("user_logs*.parquet"))) == 2   # two parts written


def test_missing_required_column_gives_clear_error(variant, tmp_path):
    with pytest.raises(ValueError, match="required column"):
        prepare(variant["var"], tmp_path / "pq")          # no column map supplied


def test_auto_months_and_datasets_match(variant):
    from src.build_dataset import suggest_months
    m = suggest_months(variant["ref"])
    assert m.endswith("2017-02")            # last full month whose end + 30 days is observed
    out_a, out_b = variant["root"] / "a" / "d.parquet", variant["root"] / "b" / "d.parquet"
    build(variant["ref"], out_a, [14, 7], "auto")
    build(variant["got"], out_b, [14, 7], "auto")
    a = pd.read_parquet(out_a).sort_values(["horizon", "event_id"]).reset_index(drop=True)
    b = pd.read_parquet(out_b).sort_values(["horizon", "event_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[["event_id", "horizon", "is_churn"] + FEATURES],
                                  b[["event_id", "horizon", "is_churn"] + FEATURES],
                                  check_exact=False, rtol=1e-9)


def test_check_schema_suggests_the_renames(variant, capsys, tmp_path):
    from src import check_schema

    class A:
        raw, write_map, count = str(variant["var"]), str(tmp_path / "map.json"), False

    check_schema.main(A)
    import json
    draft = json.loads(Path(A.write_map).read_text())
    assert draft["transactions"]["start_date"] == "transaction_date"
    assert draft["transactions"]["expiry_date"] == "membership_expire_date"
    assert draft["user_logs"]["listen_secs"] == "total_secs"
    assert draft["members"]["user_id"] == "msno"
    assert "READY" not in capsys.readouterr().out.split("transactions.csv")[1].split("VERDICT")[1][:30]


def test_check_labels_reports_agreement(dataset, tmp_path):
    from src import check_labels
    df = pd.read_parquet(dataset).query("horizon == 7")
    df["m"] = pd.to_datetime(df["expire_month"]).dt.strftime("%Y-%m")
    feb = df[df["m"] == "2017-02"]
    one = feb[feb.groupby("msno")["event_id"].transform("count") == 1][["msno", "is_churn"]].copy()
    rng = np.random.default_rng(0)
    flip = rng.random(len(one)) < 0.03
    one.loc[flip, "is_churn"] = 1 - one.loc[flip, "is_churn"]
    lab = tmp_path / "train.csv"
    one.to_csv(lab, index=False)

    class A:
        pass
    A.dataset, A.labels, A.month, A.horizon, A.out = str(dataset), str(lab), "2017-02", 7, str(tmp_path)
    check_labels.main(A)
    res = pd.read_csv(tmp_path / "label_check.csv").iloc[0]
    assert 0.95 < res["agreement"] < 0.99


def test_stray_future_transaction_does_not_extend_the_observation_window(variant, tmp_path):
    """A transaction dated long after the logs end must not make later months look labelled."""
    from src.build_dataset import suggest_months
    base = suggest_months(variant["ref"])
    con = duckdb.connect()
    d = tmp_path / "pq"
    d.mkdir()
    for f in variant["ref"].glob("*.parquet"):
        shutil.copy(f, d / f.name)
    t = (d / "transactions.parquet").as_posix()
    tmp = (d / "t_new.parquet").as_posix()
    con.execute(f"""COPY (
        SELECT * FROM read_parquet('{t}')
        UNION ALL SELECT 99999999, 'u_stray', 21, 30, 149, 149, 1, DATE '2017-07-01', DATE '2017-07-31', 0
    ) TO '{tmp}' (FORMAT PARQUET)""")
    shutil.move(tmp, t)
    assert suggest_months(d) == base
