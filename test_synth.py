"""Sanity tests for the synthetic generator: file schemas, churn rule, planted engagement decay."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silentchurn.synth import generate

SCHEMAS = {
    "members_v3.csv": ["msno", "city", "bd", "gender", "registered_via", "registration_init_time"],
    "transactions.csv": ["msno", "payment_method_id", "payment_plan_days", "plan_list_price", "actual_amount_paid",
                         "is_auto_renew", "transaction_date", "membership_expire_date", "is_cancel"],
    "user_logs.csv": ["msno", "date", "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"],
    "train.csv": ["msno", "is_churn"],
}


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    generate(d, n_members=1500, seed=3)
    return d


def test_schemas_match_real_format(raw):
    for fn, cols in SCHEMAS.items():
        assert list(pd.read_csv(raw / fn, nrows=2).columns) == cols
    assert list(pd.read_csv(raw / "transactions_v2.csv", nrows=2).columns) == SCHEMAS["transactions.csv"]
    assert list(pd.read_csv(raw / "user_logs_v2.csv", nrows=2).columns) == SCHEMAS["user_logs.csv"]


def test_deterministic_given_seed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate(a, n_members=150, seed=5)
    generate(b, n_members=150, seed=5)
    for fn in ["train.csv", "transactions.csv", "user_logs.csv"]:
        assert (a / fn).read_bytes() == (b / fn).read_bytes()


def test_labels_follow_the_30_day_noncancelled_renewal_rule(raw):
    """Independent re-derivation of the label from transactions (this is the logic labels.py must reproduce)."""
    tx = pd.concat([pd.read_csv(raw / "transactions.csv"), pd.read_csv(raw / "transactions_v2.csv")])
    tx["td"] = pd.to_datetime(tx["transaction_date"].astype(str), format="%Y%m%d")
    tx["ed"] = pd.to_datetime(tx["membership_expire_date"].astype(str), format="%Y%m%d")
    truth = pd.read_csv(raw / "_synth_truth.csv")
    truth["ed"] = pd.to_datetime(truth["expire_date"].astype(str), format="%Y%m%d")
    renew = tx[tx["is_cancel"] == 0][["msno", "td"]]
    tr = truth[truth["in_train"] == 1].drop_duplicates("msno", keep="last")
    merged = tr.merge(renew, on="msno", how="left")
    merged["gap"] = (merged["td"] - merged["ed"]).dt.days
    # a renewal "within 30 days after expiry" = a non-cancelled transaction with 0 <= gap <= 30
    # (gap < 0 are the earlier periods' own purchase rows; manual members may pre-renew a few days early,
    #  and those early rows belong to the NEXT period only if td > ed - 5).
    ok = merged[(merged["gap"] >= -3) & (merged["gap"] <= 30) & (merged["td"] > merged["ed"] - pd.Timedelta(days=4))]
    renewed = set(ok["msno"])
    derived = tr["msno"].map(lambda m: 0 if m in renewed else 1)
    # the row that created this very expiry has td = ed - plan (<= -7 days), so it is excluded above
    assert (derived.to_numpy() == tr["is_churn"].to_numpy()).mean() > 0.995


def test_churn_rate_plausible_and_both_classes_present(raw):
    t = pd.read_csv(raw / "train.csv")
    assert 0.03 < t["is_churn"].mean() < 0.25 and len(t) > 100


def test_planted_decay_is_visible_in_last_two_weeks(raw):
    """Mean listening in the last 14 days before expiry, relative to days 60-30 before, is lower for decayers."""
    truth = pd.read_csv(raw / "_synth_truth.csv")
    truth = truth[(truth["in_train"] == 1)].drop_duplicates("msno", keep="last")
    logs = pd.concat([pd.read_csv(raw / "user_logs.csv"), pd.read_csv(raw / "user_logs_v2.csv")])
    logs["d"] = pd.to_datetime(logs["date"].astype(str), format="%Y%m%d")
    logs = logs.merge(truth[["msno", "expire_date", "is_churn", "decay_days"]], on="msno")
    logs["ed"] = pd.to_datetime(logs["expire_date"].astype(str), format="%Y%m%d")
    logs["k"] = (logs["ed"] - logs["d"]).dt.days  # days before expiry
    def ratio(g):
        late = g.loc[(g["k"] >= 1) & (g["k"] <= 14), "total_secs"].sum() / 14
        base = g.loc[(g["k"] >= 30) & (g["k"] <= 60), "total_secs"].sum() / 31
        return late / base if base > 0 else np.nan
    r = logs.groupby("msno").apply(ratio, include_groups=False).rename("ratio").reset_index().merge(truth, on="msno")
    dec = r[(r["is_churn"] == 1) & (r["decay_days"] >= 14)]["ratio"].median()
    ret = r[r["is_churn"] == 0]["ratio"].median()
    assert dec < 0.6 * ret
    assert 0.7 < ret < 1.3
