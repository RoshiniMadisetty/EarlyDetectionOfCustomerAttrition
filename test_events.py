"""Tests for the official-labeller re-implementation and the event table."""
import pandas as pd
import pytest

from silentchurn import events, ingest
from silentchurn.config import Config
from silentchurn.synth import generate

COLS = ["msno", "payment_method_id", "payment_plan_days", "plan_list_price", "actual_amount_paid",
        "is_auto_renew", "transaction_date", "membership_expire_date", "is_cancel"]


def tx(msno, td, ed, cancel=0, auto=1, days=30, price=149, method=41):
    return (msno, method, days, price, price, auto, pd.Timestamp(td).date(), pd.Timestamp(ed).date(), cancel)


def write_sample(tmp_path, rows_v1, rows_v2=(), train_msnos=("a",)):
    d = tmp_path / "s"
    d.mkdir(exist_ok=True)
    for name, rows in (("transactions", rows_v1), ("transactions_v2", rows_v2)):
        df = pd.DataFrame(list(rows), columns=COLS) if rows else pd.DataFrame(columns=COLS)
        for c in ("transaction_date", "membership_expire_date"):
            df[c] = pd.to_datetime(df[c]).dt.date
        df.to_parquet(d / f"{name}.parquet", index=False)
    pd.DataFrame({"msno": list(train_msnos), "is_churn": [0] * len(train_msnos)}).to_parquet(d / "train.parquet", index=False)
    return d


def label(tmp_path, rows_v1, rows_v2=()):
    d = write_sample(tmp_path, rows_v1, rows_v2)
    con = events.connect("1GB", 1)
    df = events.derive_official_labels(con, d, "2017-02")
    return df.set_index("msno")


HIST = lambda m, ed="2017-02-15": tx(m, "2017-01-16", ed)  # January purchase expiring mid-Feb


def test_renewal_on_expiry_is_not_churn(tmp_path):
    r = label(tmp_path, [HIST("a"), tx("a", "2017-02-15", "2017-03-17")])
    assert r.loc["a", "gap"] == 0 and r.loc["a", "derived_churn"] == 0


def test_gap_29_is_renewal_and_gap_30_is_churn(tmp_path):
    rows = [HIST("a"), tx("a", "2017-03-16", "2017-04-15"),     # 29 days after 2017-02-15
            HIST("b"), tx("b", "2017-03-17", "2017-04-16")]     # 30 days after
    r = label(tmp_path, rows)
    assert (r.loc["a", "gap"], r.loc["a", "derived_churn"]) == (29, 0)
    assert (r.loc["b", "gap"], r.loc["b", "derived_churn"]) == (30, 1)


def test_no_future_transaction_is_churn(tmp_path):
    r = label(tmp_path, [HIST("a")])
    assert r.loc["a", "gap"] == 9999 and r.loc["a", "derived_churn"] == 1


def test_early_renewal_has_negative_gap_and_is_not_churn(tmp_path):
    r = label(tmp_path, [HIST("a"), tx("a", "2017-02-01", "2017-03-03")])
    assert r.loc["a", "gap"] == -14 and r.loc["a", "derived_churn"] == 0


def test_cancel_moves_expiry_earlier(tmp_path):
    # last expiry Feb 20, cancel on Feb 5 with expiry Feb 5 -> effective expiry Feb 5
    base = [HIST("a", "2017-02-20"), tx("a", "2017-02-05", "2017-02-05", cancel=1)]
    r_ok = label(tmp_path, base + [tx("a", "2017-03-03", "2017-04-02")])           # 26 days after Feb 5
    assert (r_ok.loc["a", "gap"], r_ok.loc["a", "derived_churn"]) == (26, 0)
    base_b = [HIST("b", "2017-02-20"), tx("b", "2017-02-05", "2017-02-05", cancel=1)]
    r_late = label(tmp_path, base_b + [tx("b", "2017-03-10", "2017-04-09")])      # 33 days after Feb 5 (only 18 after Feb 20)
    assert (r_late.loc["b", "gap"], r_late.loc["b", "derived_churn"]) == (33, 1)


def test_cancel_with_later_expiry_does_not_move_expiry(tmp_path):
    rows = [HIST("a"), tx("a", "2017-02-10", "2017-02-25", cancel=1), tx("a", "2017-03-05", "2017-04-04")]
    r = label(tmp_path, rows)
    assert r.loc["a", "gap"] == 18  # measured from Feb 15, not Feb 25


def test_only_cancels_in_future_is_churn(tmp_path):
    r = label(tmp_path, [HIST("a"), tx("a", "2017-02-10", "2017-02-10", cancel=1)])
    assert r.loc["a", "derived_churn"] == 1


def test_candidates_must_expire_in_target_month(tmp_path):
    rows = [HIST("a", "2017-01-31"), HIST("b", "2017-03-01"), HIST("c", "2017-02-01")]
    r = label(tmp_path, rows)
    assert set(r.index) == {"c"}


def test_last_transaction_by_date_defines_expiry(tmp_path):
    rows = [tx("a", "2017-01-02", "2017-02-01"), tx("a", "2017-01-30", "2017-03-01")]  # renewed in Jan -> March expiry
    assert len(label(tmp_path, rows)) == 0


def test_same_day_cancel_follows_subscription(tmp_path):
    rows = [tx("a", "2017-01-20", "2017-02-19"), tx("a", "2017-01-20", "2017-02-10", cancel=1)]
    r = label(tmp_path, rows)
    assert str(r.loc["a", "last_expire"])[:10] == "2017-02-10"  # cancel is the last action that day


def test_history_only_uses_january(tmp_path):
    # a December purchase expiring in Feb is NOT a candidate (official labeller filters history to the cutoff month)
    r = label(tmp_path, [tx("a", "2016-12-20", "2017-02-18", days=60)])
    assert len(r) == 0


# ------------------------------------------------------------------ synthetic end-to-end
@pytest.fixture(scope="module")
def synth_sample(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    generate(raw, n_members=2500, seed=21)
    out = tmp_path_factory.mktemp("out")
    cfg = Config(raw, out, 100, memory_limit="1GB", threads=1)  # 100% so the sample has enough customers
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ingest.run(cfg, verbose=False)
    return cfg.sample_dir


def test_rederived_labels_agree_with_train_file_on_synthetic(synth_sample):
    con = events.connect("1GB", 1)
    v = events.validate_official(con, synth_sample, "2017-02", "v1", "union")
    assert v["in_both"] > 100
    assert v["label_agreement"] > 0.98, v
    assert v["in_both"] / v["train_rows"] > 0.8, v


def test_build_events_shape_and_rules(synth_sample):
    con = events.connect("1GB", 1)
    ev = events.build_events(con, synth_sample)
    assert set(ev["period"]) == {"2017-02", "2017-03"}
    assert ev["event_id"].is_unique and ev["event_id"].tolist() == list(range(len(ev)))
    assert ev["is_churn"].isin([0, 1]).all() and 0.03 < ev["is_churn"].mean() < 0.3
    assert ev["in_target_month"].mean() > 0.9
    assert (ev.loc[ev["silent"] == 1, "is_churn"] == 1).all()
    assert (ev.loc[ev["silent"] == 1, "explicit_cancel"] == 0).all()
    assert ev["expiry_date"].notna().all()
    rep = events.events_report(ev)
    assert {"events", "churn_pct", "silent_churn_pct_of_churn"} <= set(rep.columns)
    # silent churners exist and explicit cancel exists among auto-renew churners (planted)
    assert ev["silent"].sum() > 20 and ((ev["explicit_cancel"] == 1) & (ev["is_churn"] == 1)).sum() > 5
