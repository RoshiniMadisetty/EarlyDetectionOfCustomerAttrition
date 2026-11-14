"""Step 2: expiry events and labels.

Kaggle's official labeller (WSDMChurnLabeller.scala) works like this, and this module re-implements it in SQL:
  * history  = transactions with transaction_date in [history_start, cutoff]; per customer the transactions are
    ordered (date, then plan signature DESC, then non-cancel before cancel, then expiry) and the expiry date of the
    LAST one is ``last_expire``;
  * candidates = customers whose last_expire falls in the target month;
  * future   = transactions after the cutoff; walking them in the same order, a cancel row moves the expiry EARLIER
    if its expiry date is earlier; the first NON-cancel row gives gap = transaction_date - (adjusted expiry);
  * churn iff there is no such row, or gap >= 30 (so a gap of 30 days is churn, 29 is a renewal).
train.csv = Feb-2017 expiries (cutoff 2017-01-31); train_v2.csv = Mar-2017 expiries (cutoff 2017-02-28).
Labels for the March cohort need April transactions we do not have, so they are taken from train_v2.csv, and only
the February cohort can be re-derived and cross-checked (``validate_official``).

``build_events`` makes one row per (customer, expiry event) for the analysis, using the OFFICIAL labels from the
train files and an expiry date derived from transactions up to the cutoff only.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

PERIODS = {
    # period: (train table, history start, cutoff, expiry window start, expiry window end)
    "2017-02": ("train", "2017-01-01", "2017-01-31", "2017-02-01", "2017-02-28"),
    "2017-03": ("train_v2", "2017-02-01", "2017-02-28", "2017-03-01", "2017-03-31"),
}
def _order(asc: bool, p: str = "") -> str:
    """ORDER BY key reproducing the Scala comparator (p = table alias prefix such as 't.')."""
    sig = f"(CAST({p}plan_list_price AS VARCHAR) || CAST({p}payment_plan_days AS VARCHAR) || CAST({p}payment_method_id AS VARCHAR))"
    exp = f"({p}membership_expire_date - DATE '1970-01-01')"
    d1, d2 = ("ASC", "DESC") if asc else ("DESC", "ASC")
    return (f"{p}transaction_date {d1}, {sig} {d2}, {p}is_cancel {d1}, "
            f"CASE WHEN {p}is_cancel = 1 THEN -{exp} ELSE {exp} END {d1}")


ORDER_ASC, ORDER_DESC = _order(True), _order(False)
EXPLICIT_CANCEL_WINDOW_DAYS = 60


def _q(p) -> str:
    return "'" + str(p).replace("'", "''") + "'"


def connect(memory_limit: str = "8GB", threads: int = 2, temp_dir: Optional[str] = None):
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={int(threads)}")
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory={_q(temp_dir)}")
    return con


def tx_relation(sample_dir, source: str = "union") -> str:
    """SQL relation of transactions: 'v1', 'v2' or 'union' (exact duplicates removed)."""
    p1 = f"read_parquet({_q(Path(sample_dir) / 'transactions.parquet')})"
    p2 = f"read_parquet({_q(Path(sample_dir) / 'transactions_v2.parquet')})"
    if source == "v1":
        return f"(SELECT DISTINCT * FROM {p1})"
    if source == "v2":
        return f"(SELECT DISTINCT * FROM {p2})"
    if source == "union":
        return f"(SELECT * FROM {p1} UNION SELECT * FROM {p2})"  # UNION (not ALL) removes exact duplicates
    raise ValueError(source)


def derive_official_labels(con, sample_dir, period: str, history_source: str = "v1", future_source: str = "union",
                           restrict_to_train: bool = False) -> pd.DataFrame:
    """Re-derive churn labels exactly like the official labeller. Returns msno, last_expire, gap, derived_churn."""
    train_tbl, h0, cutoff, lo, hi = PERIODS[period]
    hist, fut = tx_relation(sample_dir, history_source), tx_relation(sample_dir, future_source)
    sql = f"""
    WITH last AS (
        SELECT msno, last_expire FROM (
            SELECT msno, membership_expire_date AS last_expire,
                   row_number() OVER (PARTITION BY msno ORDER BY {ORDER_DESC}) AS rn
            FROM {hist} WHERE transaction_date >= DATE '{h0}' AND transaction_date <= DATE '{cutoff}') WHERE rn = 1),
    cand AS (SELECT * FROM last WHERE last_expire >= DATE '{lo}' AND last_expire <= DATE '{hi}'),
    f AS (
        SELECT c.msno, c.last_expire, t.transaction_date, t.membership_expire_date, t.is_cancel,
               row_number() OVER (PARTITION BY c.msno ORDER BY {_order(True, 't.')}) AS rn
        FROM cand c JOIN {fut} t ON t.msno = c.msno WHERE t.transaction_date > DATE '{cutoff}'),
    first_nc AS (SELECT msno, min(rn) AS k FROM f WHERE is_cancel = 0 GROUP BY msno),
    adj AS (
        SELECT c.msno, c.last_expire, n.k,
               least(c.last_expire, coalesce((SELECT min(f.membership_expire_date) FROM f
                     WHERE f.msno = c.msno AND f.is_cancel = 1 AND f.rn < n.k), c.last_expire)) AS eff_expire
        FROM cand c JOIN first_nc n ON n.msno = c.msno)
    SELECT c.msno, c.last_expire,
           CASE WHEN a.k IS NULL THEN 9999
                ELSE date_diff('day', a.eff_expire, (SELECT transaction_date FROM f WHERE f.msno = c.msno AND f.rn = a.k)) END AS gap
    FROM cand c LEFT JOIN adj a ON a.msno = c.msno"""
    df = con.execute(sql).df()
    df["derived_churn"] = (df["gap"] >= 30).astype(int)
    if restrict_to_train:
        tr = con.execute(f"SELECT msno FROM read_parquet({_q(Path(sample_dir) / (train_tbl + '.parquet'))})").df()
        df = df[df["msno"].isin(set(tr["msno"]))]
    return df


def validate_official(con, sample_dir, period: str = "2017-02", history_source: str = "v1",
                      future_source: str = "union") -> dict:
    """Compare re-derived labels with train.csv (only the Feb cohort can be re-derived)."""
    train_tbl = PERIODS[period][0]
    d = derive_official_labels(con, sample_dir, period, history_source, future_source)
    tr = con.execute(f"SELECT msno, is_churn FROM read_parquet({_q(Path(sample_dir) / (train_tbl + '.parquet'))})").df()
    m = tr.merge(d, on="msno", how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    agree = float((both["is_churn"] == both["derived_churn"]).mean()) if len(both) else float("nan")
    return {
        "period": period, "history_source": history_source, "future_source": future_source,
        "train_rows": int(len(tr)), "derived_candidates": int(len(d)), "in_both": int(len(both)),
        "only_in_train_file": int((m["_merge"] == "left_only").sum()), "only_derived": int((m["_merge"] == "right_only").sum()),
        "label_agreement": agree,
        "mismatch_train1_derived0": int(((both["is_churn"] == 1) & (both["derived_churn"] == 0)).sum()),
        "mismatch_train0_derived1": int(((both["is_churn"] == 0) & (both["derived_churn"] == 1)).sum()),
        "train_churn_rate": float(tr["is_churn"].mean()), "derived_churn_rate_on_both": float(both["derived_churn"].mean()) if len(both) else float("nan"),
    }


DEFAULT_HISTORY_SOURCE = {"2017-02": "v1", "2017-03": "union"}  # Feb labels came from the original transactions.csv era


def build_events(con, sample_dir, history_source: Optional[str] = None) -> pd.DataFrame:
    """One row per (customer, period) from train/train_v2 with official labels and a cutoff-safe expiry date.
    history_source=None uses DEFAULT_HISTORY_SOURCE per period; 'v1' / 'v2' / 'union' forces one for both."""
    sd = Path(sample_dir)
    frames = []
    for period, (train_tbl, h0, cutoff, lo, hi) in PERIODS.items():
        if not (sd / f"{train_tbl}.parquet").exists():
            continue
        tx = tx_relation(sd, history_source or DEFAULT_HISTORY_SOURCE[period])
        allx = tx_relation(sd, "union")
        q = f"""
        WITH tr AS (SELECT msno, is_churn FROM read_parquet({_q(sd / (train_tbl + '.parquet'))})),
        win AS (SELECT msno, membership_expire_date AS e FROM (
                  SELECT msno, membership_expire_date,
                         row_number() OVER (PARTITION BY msno ORDER BY {ORDER_DESC}) AS rn
                  FROM {tx} WHERE transaction_date >= DATE '{h0}' AND transaction_date <= DATE '{cutoff}') WHERE rn = 1),
        allh AS (SELECT msno, membership_expire_date AS e FROM (
                  SELECT msno, membership_expire_date,
                         row_number() OVER (PARTITION BY msno ORDER BY {ORDER_DESC}) AS rn
                  FROM {tx} WHERE transaction_date <= DATE '{cutoff}') WHERE rn = 1),
        plan AS (SELECT msno, is_auto_renew, payment_plan_days FROM (
                  SELECT msno, is_auto_renew, payment_plan_days,
                         row_number() OVER (PARTITION BY msno ORDER BY {ORDER_DESC}) AS rn
                  FROM {tx} WHERE is_cancel = 0 AND transaction_date <= DATE '{cutoff}') WHERE rn = 1),
        base AS (SELECT tr.msno, tr.is_churn, coalesce(win.e, allh.e) AS expiry_date,
                        CASE WHEN win.e IS NOT NULL THEN 'official_window' ELSE 'fallback_all_history' END AS expiry_source
                 FROM tr LEFT JOIN win USING (msno) LEFT JOIN allh USING (msno))
        SELECT '{period}' AS period, DATE '{cutoff}' AS cutoff_date, b.msno, b.is_churn, b.expiry_date, b.expiry_source,
               p.is_auto_renew, p.payment_plan_days AS plan_days,
               coalesce((SELECT max((c.is_cancel = 1)::INT) FROM {allx} c
                         WHERE c.msno = b.msno AND c.is_cancel = 1
                           AND c.transaction_date >= b.expiry_date - {EXPLICIT_CANCEL_WINDOW_DAYS}
                           AND c.transaction_date <= b.expiry_date), 0) AS explicit_cancel,
               (b.expiry_date >= DATE '{lo}' AND b.expiry_date <= DATE '{hi}') AS in_target_month
        FROM base b LEFT JOIN plan p USING (msno)"""
        frames.append(con.execute(q).df())
    ev = pd.concat(frames, ignore_index=True)
    ev["explicit_cancel"] = ev["explicit_cancel"].fillna(0).astype(int)
    ev["silent"] = ((ev["is_churn"] == 1) & (ev["explicit_cancel"] == 0)).astype(int)
    ev["is_auto_renew"] = ev["is_auto_renew"].fillna(0).astype(int)
    ev["expiry_date"] = pd.to_datetime(ev["expiry_date"])
    ev["cutoff_date"] = pd.to_datetime(ev["cutoff_date"])
    ev.insert(0, "event_id", range(len(ev)))
    return ev


def events_report(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby("period")
    return pd.DataFrame({
        "events": g.size(), "churn_pct": 100 * g["is_churn"].mean(),
        "silent_churn_pct_of_churn": 100 * g.apply(lambda d: d.loc[d.is_churn == 1, "silent"].mean(), include_groups=False),
        "auto_renew_pct": 100 * g["is_auto_renew"].mean(),
        "expiry_in_target_month_pct": 100 * g["in_target_month"].mean(),
        "from_fallback_pct": 100 * g.apply(lambda d: (d.expiry_source != "official_window").mean(), include_groups=False),
    }).round(2)
