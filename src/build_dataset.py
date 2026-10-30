"""Build the multi-horizon, leakage-controlled modelling table.

Unit of analysis: one SUBSCRIPTION EXPIRY EVENT = one non-cancelled transaction
whose membership_expire_date E falls inside the study months.

For every horizon h (days before expiry):
    C = E - h                       cutoff day; data dated <= C is the ONLY input
    event is kept (at-risk) iff
        (a) its own transaction was made on or before C, and
        (b) the member had NOT already extended membership beyond E by C
            (no non-cancelled transaction dated <= C with a later expiry)
    label is_churn = 1 iff no non-cancelled transaction with a later expiry
        than E is dated within E + 30 days (the KKBox definition).
    Events whose E + 30 days lies beyond the last observed transaction date are
    dropped, because their label cannot be observed (right-censoring).

Selection rules (a) and (b) use only information dated <= C, so nothing about
the outcome leaks into who is scored. tests/test_pipeline.py verifies this by
rewriting all data after C and checking the features do not change.

`in_common_cohort` marks events that are at-risk at EVERY horizon. Compare
horizons on that cohort (default in run_experiments.py), otherwise accuracy
differences across horizons are confounded by changing populations.

Usage:
  python -m src.build_dataset --months 2016-09:2017-02 --horizons 30 21 14 7
"""
import argparse
import json
from pathlib import Path

import duckdb

from .config import (BASELINE_WINDOW, CHURN_GRACE_DAYS, DATA_PARQUET,
                     DEFAULT_HORIZONS, FEATURES, OUTPUTS, RECENT_WINDOW)

assert RECENT_WINDOW == 30, "slope constants below assume a 30-day window"


def suggest_months(data_dir, n_months: int = 6) -> str:
    """Pick the last `n_months` FULL expiry months that can be labelled and have history.

    A month is usable when its last day + 30 days is on or before the end of observation
    (the earlier of the last transaction and the last log row; otherwise labels are censored), and it starts at least 150 days after the
    first log date (30-day window + 90-day baseline + margin).
    """
    import datetime as dt
    d = Path(data_dir)
    con = duckdb.connect()
    data_end = con.execute(
        f"SELECT LEAST((SELECT max(transaction_date) FROM read_parquet('{(d / 'transactions.parquet').as_posix()}')), "
        f"(SELECT max(date) FROM read_parquet('{(d / 'user_logs*.parquet').as_posix()}')))"
    ).fetchone()[0]
    log_min = con.execute(
        f"SELECT min(date) FROM read_parquet('{(d / 'user_logs*.parquet').as_posix()}')"
    ).fetchone()[0]
    con.close()

    def first_of_next(y, m):
        return dt.date(y + (m // 12), (m % 12) + 1, 1)

    y, m = data_end.year, data_end.month
    while True:  # latest month whose last day + grace <= data_end
        last_day = first_of_next(y, m) - dt.timedelta(days=1)
        if last_day + dt.timedelta(days=CHURN_GRACE_DAYS) <= data_end:
            break
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    end = (y, m)
    earliest = log_min + dt.timedelta(days=150)
    months = []
    yy, mm = end
    for _ in range(n_months):
        if dt.date(yy, mm, 1) < earliest:
            break
        months.append((yy, mm))
        yy, mm = (yy, mm - 1) if mm > 1 else (yy - 1, 12)
    if not months:
        raise ValueError(f"data does not cover enough history: transactions end {data_end}, "
                         f"logs start {log_min}")
    a, b = months[-1], months[0]
    return f"{a[0]:04d}-{a[1]:02d}:{b[0]:04d}-{b[1]:02d}"


def month_range(spec: str):
    """'2016-09:2017-02' -> (DATE '2016-09-01', exclusive end DATE '2017-03-01')."""
    a, b = spec.split(":")
    ay, am = map(int, a.split("-"))
    by, bm = map(int, b.split("-"))
    ey, em = (by + (bm // 12), (bm % 12) + 1)
    return f"{ay:04d}-{am:02d}-01", f"{ey:04d}-{em:02d}-01"


def build(data_dir: Path, out_path: Path, horizons, months: str,
          sample_frac: float = 1.0, memory_limit: str = "6GB",
          report_path: Path = None) -> dict:
    data_dir, out_path = Path(data_dir), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / "_duckdb_tmp"
    tmp.mkdir(exist_ok=True)

    con = duckdb.connect(str(out_path.parent / "_build.duckdb"))
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    con.execute("PRAGMA enable_progress_bar=false")

    P = lambda n: (data_dir / (f"{n}*.parquet" if n == "user_logs" else f"{n}.parquet")).as_posix()
    con.execute(f"CREATE OR REPLACE VIEW tx AS SELECT * FROM read_parquet('{P('transactions')}')")
    con.execute(f"CREATE OR REPLACE VIEW logs AS SELECT * FROM read_parquet('{P('user_logs')}')")
    con.execute(f"CREATE OR REPLACE VIEW members AS SELECT * FROM read_parquet('{P('members')}')")

    if months == "auto":
        months = suggest_months(data_dir)
        print(f"months=auto -> {months}")
    start, end_excl = month_range(months)
    # End of observation = the EARLIER of the last transaction and the last log row, so a
    # stray future-dated row or a log file that ends early cannot make later months look observed.
    data_end = con.execute(
        "SELECT LEAST((SELECT max(transaction_date) FROM tx), (SELECT max(date) FROM logs))"
    ).fetchone()[0]
    print(f"end of observation: {data_end}")
    G = CHURN_GRACE_DAYS
    keep = int(round(sample_frac * 1000))
    sample_sql = f"AND hash(tx_id) % 1000 < {keep}" if sample_frac < 1.0 else ""

    report = []
    for h in horizons:
        # --- events and eligibility ------------------------------------------------
        con.execute(f"""
            CREATE OR REPLACE TABLE ev0 AS
            SELECT tx_id AS event_id, msno, transaction_date AS anchor_tx_date,
                   membership_expire_date AS E,
                   CAST(date_trunc('month', membership_expire_date) AS DATE) AS expire_month,
                   membership_expire_date - {h} AS C,
                   payment_method_id AS anchor_payment_method_id,
                   payment_plan_days AS anchor_plan_days,
                   plan_list_price AS anchor_list_price,
                   actual_amount_paid AS anchor_paid,
                   is_auto_renew AS anchor_auto_renew
            FROM tx
            WHERE is_cancel = 0
              AND membership_expire_date >= DATE '{start}'
              AND membership_expire_date <  DATE '{end_excl}'
              AND membership_expire_date + {G} <= DATE '{data_end}'
              {sample_sql}
        """)
        n0 = con.execute("SELECT count(*) FROM ev0").fetchone()[0]

        con.execute("""
            CREATE OR REPLACE TABLE ev1 AS
            SELECT * FROM ev0 WHERE anchor_tx_date <= C
        """)
        n1 = con.execute("SELECT count(*) FROM ev1").fetchone()[0]

        con.execute(f"""
            CREATE OR REPLACE TABLE ev AS
            SELECT c.*,
                   CASE WHEN EXISTS (
                        SELECT 1 FROM tx u
                        WHERE u.msno = c.msno AND u.is_cancel = 0
                          AND u.tx_id <> c.event_id
                          AND u.membership_expire_date > c.E
                          AND u.transaction_date <= c.E + {G}) THEN 0 ELSE 1 END AS is_churn
            FROM ev1 c
            WHERE NOT EXISTS (
                SELECT 1 FROM tx u
                WHERE u.msno = c.msno AND u.is_cancel = 0
                  AND u.tx_id <> c.event_id
                  AND u.membership_expire_date > c.E
                  AND u.transaction_date <= c.C)
        """)
        n2, churn_rate = con.execute("SELECT count(*), avg(is_churn) FROM ev").fetchone()
        report.append({"horizon": h, "expiry_events_observable": n0,
                       "own_tx_before_cutoff": n1, "at_risk_events": n2,
                       "churn_rate": churn_rate})
        print(f"h={h:>3}: events={n0:,}  own-tx-by-cutoff={n1:,}  at-risk={n2:,}  churn={churn_rate}")

        # --- features (every join is restricted to dates <= C) ------------------------
        lo = RECENT_WINDOW + BASELINE_WINDOW
        con.execute(f"""
            CREATE OR REPLACE TABLE feat_{h} AS
            WITH daily AS (
                SELECT e.event_id, l.date AS d, e.C,
                       LEAST(SUM(l.total_secs), 86400) AS secs,
                       SUM(l.num_unq) AS unq,
                       SUM(l.num_25) AS n25, SUM(l.num_50) AS n50, SUM(l.num_75) AS n75,
                       SUM(l.num_985) AS n985, SUM(l.num_100) AS n100
                FROM ev e
                JOIN logs l ON l.msno = e.msno AND l.date <= e.C AND l.date > e.C - {lo}
                GROUP BY e.event_id, l.date, e.C
            ),
            agg AS (
                SELECT event_id,
                  SUM(secs) FILTER (WHERE d > C - 7)  AS secs_7,
                  SUM(secs) FILTER (WHERE d > C - 14) AS secs_14,
                  SUM(secs) FILTER (WHERE d > C - 30) AS secs_30,
                  SUM(unq)  FILTER (WHERE d > C - 7)  AS unq_7,
                  SUM(unq)  FILTER (WHERE d > C - 14) AS unq_14,
                  SUM(unq)  FILTER (WHERE d > C - 30) AS unq_30,
                  SUM(n25 + n50 + n75 + n985 + n100) FILTER (WHERE d > C - 30) AS plays_30,
                  SUM(n25)  FILTER (WHERE d > C - 30) AS n25_30,
                  SUM(n100) FILTER (WHERE d > C - 30) AS n100_30,
                  COUNT(*) FILTER (WHERE d > C - 7)  AS active_days_7,
                  COUNT(*) FILTER (WHERE d > C - 14) AS active_days_14,
                  COUNT(*) FILTER (WHERE d > C - 30) AS active_days_30,
                  SUM(secs * (d - (C - 29))) FILTER (WHERE d > C - 30) AS sxy_30,
                  MIN(d) FILTER (WHERE d > C - 30) AS first_d,
                  MAX(d) FILTER (WHERE d > C - 30) AS last_d,
                  SUM(secs)  FILTER (WHERE d <= C - 30) AS base_secs,
                  COUNT(*)   FILTER (WHERE d <= C - 30) AS base_active_days,
                  ANY_VALUE(C) AS C
                FROM daily GROUP BY event_id
            ),
            gaps AS (
                SELECT event_id, MAX(g) AS max_inner_gap
                FROM (SELECT event_id,
                             d - LAG(d) OVER (PARTITION BY event_id ORDER BY d) - 1 AS g
                      FROM daily WHERE d > C - 30)
                GROUP BY event_id
            ),
            txseq AS (
                SELECT e.event_id, t.payment_plan_days, t.payment_method_id, t.is_auto_renew,
                       LAG(t.payment_plan_days) OVER w AS prev_plan,
                       LAG(t.payment_method_id) OVER w AS prev_pm
                FROM ev e JOIN tx t
                  ON t.msno = e.msno AND t.transaction_date <= e.C AND t.is_cancel = 0
                WINDOW w AS (PARTITION BY e.event_id
                             ORDER BY t.transaction_date, t.membership_expire_date, t.tx_id)
            ),
            txa AS (
                SELECT event_id,
                       COUNT(*) AS n_paid_tx,
                       SUM(CASE WHEN prev_plan IS NOT NULL AND prev_plan <> payment_plan_days
                                THEN 1 ELSE 0 END) AS n_plan_changes,
                       SUM(CASE WHEN prev_pm IS NOT NULL AND prev_pm <> payment_method_id
                                THEN 1 ELSE 0 END) AS n_pay_method_changes,
                       COUNT(DISTINCT payment_plan_days)  AS n_distinct_plan_days,
                       COUNT(DISTINCT payment_method_id)  AS n_distinct_pay_methods,
                       AVG(CASE WHEN is_auto_renew = 0 THEN 1.0 ELSE 0.0 END) AS manual_tx_share
                FROM txseq GROUP BY event_id
            ),
            canc AS (
                SELECT e.event_id,
                       COUNT(*) AS n_cancel_before,
                       COUNT(*) FILTER (WHERE t.transaction_date > e.C - 30) AS n_cancel_30d,
                       COUNT(*) FILTER (WHERE t.transaction_date >= e.anchor_tx_date) AS n_cancel_since_anchor
                FROM ev e JOIN tx t
                  ON t.msno = e.msno AND t.is_cancel = 1 AND t.transaction_date <= e.C
                GROUP BY e.event_id
            ),
            base AS (
                SELECT e.*,
                       COALESCE(a.secs_7, 0) AS secs_7, COALESCE(a.secs_14, 0) AS secs_14,
                       COALESCE(a.secs_30, 0) AS secs_30,
                       COALESCE(a.unq_7, 0) AS unq_7, COALESCE(a.unq_14, 0) AS unq_14,
                       COALESCE(a.unq_30, 0) AS unq_30,
                       COALESCE(a.plays_30, 0) AS plays_30,
                       COALESCE(a.n25_30, 0) AS n25_30, COALESCE(a.n100_30, 0) AS n100_30,
                       COALESCE(a.active_days_7, 0) AS active_days_7,
                       COALESCE(a.active_days_14, 0) AS active_days_14,
                       COALESCE(a.active_days_30, 0) AS active_days_30,
                       COALESCE(a.sxy_30, 0) AS sxy_30,
                       COALESCE(a.base_secs, 0) AS base_secs,
                       COALESCE(a.base_active_days, 0) AS base_active_days,
                       CASE WHEN COALESCE(a.active_days_30, 0) = 0 THEN 30
                            ELSE e.C - a.last_d END AS days_since_last_active,
                       CASE WHEN COALESCE(a.active_days_30, 0) = 0 THEN 30
                            ELSE GREATEST(COALESCE(g.max_inner_gap, 0),
                                          a.first_d - (e.C - 29), e.C - a.last_d) END
                            AS max_inactivity_gap_30,
                       COALESCE(x.n_paid_tx, 1) AS n_paid_tx,
                       COALESCE(x.n_plan_changes, 0) AS n_plan_changes,
                       COALESCE(x.n_pay_method_changes, 0) AS n_pay_method_changes,
                       COALESCE(x.n_distinct_plan_days, 1) AS n_distinct_plan_days,
                       COALESCE(x.n_distinct_pay_methods, 1) AS n_distinct_pay_methods,
                       x.manual_tx_share,
                       COALESCE(c.n_cancel_before, 0) AS n_cancel_before,
                       COALESCE(c.n_cancel_30d, 0) AS n_cancel_30d,
                       COALESCE(c.n_cancel_since_anchor, 0) AS n_cancel_since_anchor,
                       m.city, m.bd, m.gender_code, m.registered_via,
                       CASE WHEN m.registration_date <= e.C
                            THEN e.C - m.registration_date END AS tenure_days
                FROM ev e
                LEFT JOIN agg a ON a.event_id = e.event_id
                LEFT JOIN gaps g ON g.event_id = e.event_id
                LEFT JOIN txa x ON x.event_id = e.event_id
                LEFT JOIN canc c ON c.event_id = e.event_id
                LEFT JOIN members m ON m.msno = e.msno
            )
            SELECT
                event_id, msno, {h} AS horizon, E AS expire_date, C AS cutoff_date,
                expire_month, is_churn,
                CASE WHEN n_cancel_since_anchor > 0 THEN 'explicit_cancel'
                     WHEN anchor_auto_renew = 1 THEN 'auto_renew_on'
                     ELSE 'manual_renew' END AS segment,
                CASE WHEN anchor_paid > 0 THEN anchor_paid
                     WHEN anchor_list_price > 0 THEN anchor_list_price ELSE 0 END AS renewal_value,
                -- contract
                anchor_payment_method_id, anchor_plan_days, anchor_list_price, anchor_paid,
                anchor_auto_renew,
                anchor_list_price - anchor_paid AS anchor_discount,
                CASE WHEN anchor_list_price > 0
                     THEN (anchor_list_price - anchor_paid) / anchor_list_price END AS anchor_discount_ratio,
                n_cancel_since_anchor, n_cancel_30d, n_cancel_before,
                n_plan_changes, n_pay_method_changes, n_distinct_plan_days,
                n_distinct_pay_methods, manual_tx_share,
                C - anchor_tx_date AS days_since_anchor_tx,
                CASE WHEN anchor_plan_days > 0
                     THEN (C - anchor_tx_date) * 1.0 / anchor_plan_days END AS plan_progress,
                -- tenure / profile
                tenure_days, GREATEST(n_paid_tx - 1, 0) AS n_renewals,
                city, registered_via, gender_code, bd,
                -- usage level
                secs_7, secs_14, secs_30, unq_7, unq_14, unq_30, plays_30,
                CASE WHEN plays_30 > 0 THEN n100_30 * 1.0 / plays_30 END AS completion_ratio_30,
                CASE WHEN plays_30 > 0 THEN n25_30 * 1.0 / plays_30 END AS skip_ratio_30,
                CASE WHEN active_days_30 > 0 THEN secs_30 * 1.0 / active_days_30 END
                     AS avg_secs_per_active_day_30,
                -- engagement
                active_days_7, active_days_14, active_days_30,
                days_since_last_active, max_inactivity_gap_30,
                -- trajectory
                (secs_7 / 7.0) / ((secs_30 - secs_7) / 23.0 + 1) AS usage_ratio_7_vs_prev,
                (secs_14 / 14.0) / ((secs_30 - secs_14) / 16.0 + 1) AS usage_ratio_14_vs_prev,
                (30.0 * sxy_30 - 435.0 * secs_30) / 67425.0 AS usage_slope_30,
                CASE WHEN base_active_days > 0
                     THEN (secs_30 / 30.0) / (base_secs / {BASELINE_WINDOW}.0 + 1) END AS usage_vs_baseline,
                CASE WHEN base_active_days > 0
                     THEN (active_days_30 / 30.0) / (base_active_days / {BASELINE_WINDOW}.0) END
                     AS active_days_ratio_vs_baseline
            FROM base
        """)

    # --- stack horizons, flag the common cohort, write parquet ------------------------
    union = " UNION ALL BY NAME ".join(f"SELECT * FROM feat_{h}" for h in horizons)
    nh = len(horizons)
    con.execute(f"""
        COPY (
            SELECT *, (COUNT(*) OVER (PARTITION BY event_id) = {nh}) AS in_common_cohort
            FROM ({union})
            ORDER BY horizon, event_id
        ) TO '{out_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n_common = con.execute(f"""
        SELECT count(*) FROM (SELECT event_id FROM ({union}) GROUP BY event_id HAVING count(*) = {nh})
    """).fetchone()[0]
    con.close()

    # sanity: every configured feature must exist in the output
    cols = duckdb.connect().execute(
        f"SELECT * FROM read_parquet('{out_path.as_posix()}') LIMIT 0").fetchdf().columns
    missing = [f for f in FEATURES if f not in cols]
    assert not missing, f"features missing from output: {missing}"

    result = {"horizons": list(horizons), "months": months, "data_end": str(data_end),
              "common_cohort_events": n_common, "per_horizon": report}
    rp = report_path or (out_path.parent / "cohort_report.json")
    Path(rp).write_text(json.dumps(result, indent=2, default=str))
    print(f"common cohort (at risk at every horizon): {n_common:,} events")
    print(f"wrote {out_path}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_PARQUET)
    ap.add_argument("--out", type=Path, default=OUTPUTS / "dataset.parquet")
    ap.add_argument("--months", default="auto",
                    help="expiry months START:END inclusive, e.g. 2016-09:2017-02, "
                         "or 'auto' to pick the last six labelled months")
    ap.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="deterministic event sample for quick dev runs (e.g. 0.1)")
    ap.add_argument("--memory-limit", default="6GB")
    a = ap.parse_args()
    build(a.data_dir, a.out, sorted(set(a.horizons), reverse=True), a.months,
          a.sample_frac, a.memory_limit)
