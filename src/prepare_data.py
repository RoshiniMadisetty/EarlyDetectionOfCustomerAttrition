"""Convert the raw subscription CSV files into typed, cleaned Parquet files.

Run once. Reads from data/raw by default and writes to data/parquet.

  python -m src.prepare_data --raw data/raw --out data/parquet
  python -m src.prepare_data --raw data/raw --map column_map.json     # renamed columns
  python -m src.check_schema --raw data/raw                           # inspect files first

Standard (KKBox) columns the pipeline understands:
  members:       msno, city, bd, gender, registered_via, registration_init_time
  transactions:  msno, payment_method_id, payment_plan_days, plan_list_price,
                 actual_amount_paid, is_auto_renew, transaction_date,
                 membership_expire_date, is_cancel
  user_logs:     msno, date, num_25, num_50, num_75, num_985, num_100, num_unq, total_secs

REQUIRED: members.msno | transactions.msno, transaction_date, membership_expire_date,
is_cancel | user_logs.msno, date, total_secs. Every other column is optional; a missing
one becomes NULL (counts become 0) and the features built from it are blank, with a warning.

Files differ? Use a column map (JSON) to rename YOUR columns to the standard ones:
  {"members": {"user_id": "msno"}, "transactions": {"uid": "msno", "start_date": "transaction_date"},
   "user_logs": {"user_id": "msno", "listen_secs": "total_secs"}}

Dates may be YYYYMMDD integers, ISO dates (2017-03-01) or timestamps; the format is
detected per value. Rows whose dates cannot be parsed are dropped and counted.

Cleaning rules (all explicit, nothing silent):
  * exact duplicate transaction rows are removed; each gets a stable integer tx_id
  * bd (age) outside 10..80 becomes NULL
  * gender -> gender_code (male=1, female=0, otherwise NULL)
  * per-row listening seconds are clipped to [0, 86400]; play counts to >= 0
"""
import argparse
import json
from pathlib import Path

import duckdb

from .config import DATA_PARQUET, DATA_RAW

CANDIDATES = {
    "members": ["members_v3.csv", "members.csv", "members_v2.csv"],      # first match is used
    "transactions": ["transactions.csv", "transactions_v2.csv"],         # every match is used
    "user_logs": ["user_logs.csv", "user_logs_v2.csv"],                  # every match is used, in order
}
MULTI = {"transactions", "user_logs"}
REQUIRED = {
    "members": ["msno"],
    "transactions": ["msno", "transaction_date", "membership_expire_date", "is_cancel"],
    "user_logs": ["msno", "date", "total_secs"],
}


def find_files(raw: Path, table: str):
    """members: first match. transactions / user_logs: every match (v1 then v2)."""
    found = [raw / n for n in CANDIDATES[table] if (raw / n).exists()]
    if not found:
        raise FileNotFoundError(
            f"none of {CANDIDATES[table]} found in {raw}. Pass the file explicitly "
            f"(--members-file, --transactions-file, --logs-file) or rename it.")
    return found if table in MULTI else found[:1]


def date_expr(col: str) -> str:
    """YYYYMMDD ints, ISO dates and timestamps all become DATE; anything else is NULL."""
    return (f"COALESCE(TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%Y%m%d') AS DATE), "
            f"TRY_CAST(TRY_CAST({col} AS TIMESTAMP) AS DATE))")


DATE_COL = date_expr('"date"')


class Source:
    """A CSV read as text, with the user's column renames applied."""

    def __init__(self, con, paths, cmap: dict, table: str):
        self.table = table
        path = paths[0]
        plist = "[" + ", ".join(f"'{Path(p).as_posix()}'" for p in paths) + "]"
        reader = f"read_csv({plist}, header=true, all_varchar=true, union_by_name=true)"
        raw_cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {reader}").fetchall()]
        ren = [(old, new) for old, new in (cmap or {}).items()
               if old in raw_cols and new not in raw_cols]
        rename_sql = (" RENAME (" + ", ".join(f'"{o}" AS "{n}"' for o, n in ren) + ")") if ren else ""
        self.sql = f"(SELECT *{rename_sql} FROM {reader})"
        renamed_old = {o for o, _ in ren}
        self.cols = [c for c in raw_cols if c not in renamed_old] + [n for _, n in ren]
        missing = [c for c in REQUIRED[table] if c not in self.cols]
        if missing:
            raise ValueError(
                f"{table}: required column(s) {missing} not found in {Path(path).name}.\n"
                f"  columns in the file: {raw_cols}\n"
                f"  run `python -m src.check_schema` and add a column map (see the "
                f"docstring of src/prepare_data.py)")

    def col(self, name, expr, default="NULL"):
        """expr uses {c} for the column; falls back to `default` when the column is absent."""
        if name in self.cols:
            return expr.format(c=f'"{name}"')
        print(f"  warning: {self.table}.{name} not found; filled with {default}")
        return default


def prepare(raw: Path, out: Path, memory_limit: str = "6GB", column_map: dict = None,
            files: dict = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    column_map, files = column_map or {}, files or {}

    paths = {t: [Path(files[t])] if files.get(t) else find_files(raw, t) for t in CANDIDATES}
    for t, ps in paths.items():
        print(f"{t}: reading {[p.name for p in ps]}")
    src = {t: Source(con, paths[t], column_map.get(t, {}), t) for t in ("members", "transactions")}

    # ------------------------------------------------------------------ members
    m = src["members"]
    order = '"registration_init_time" DESC' if "registration_init_time" in m.cols else "msno"
    reg = (date_expr('"registration_init_time"') if "registration_init_time" in m.cols else "NULL")
    con.execute(f"""
        COPY (
          SELECT msno,
                 {m.col('city', 'TRY_CAST({c} AS INTEGER)')}                       AS city,
                 {m.col('bd', "CASE WHEN TRY_CAST({c} AS DOUBLE) BETWEEN 10 AND 80 THEN TRY_CAST({c} AS DOUBLE) END")} AS bd,
                 {m.col('gender', "CASE lower(trim({c})) WHEN 'male' THEN 1.0 WHEN 'female' THEN 0.0 END")} AS gender_code,
                 {m.col('registered_via', 'TRY_CAST({c} AS INTEGER)')}             AS registered_via,
                 {reg}                                                             AS registration_date
          FROM {m.sql}
          QUALIFY row_number() OVER (PARTITION BY msno ORDER BY {order}) = 1
        ) TO '{(out / 'members.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # ------------------------------------------------------------ transactions
    t = src["transactions"]
    con.execute(f"""
        COPY (
          SELECT row_number() OVER (ORDER BY msno, transaction_date, membership_expire_date,
                                    payment_plan_days, actual_amount_paid, plan_list_price,
                                    payment_method_id, is_auto_renew, is_cancel)::BIGINT AS tx_id,
                 *
          FROM (
            SELECT DISTINCT
                   msno,
                   {t.col('payment_method_id', 'TRY_CAST({c} AS INTEGER)')}  AS payment_method_id,
                   {t.col('payment_plan_days', 'TRY_CAST({c} AS INTEGER)')}  AS payment_plan_days,
                   {t.col('plan_list_price', 'TRY_CAST({c} AS DOUBLE)')}     AS plan_list_price,
                   {t.col('actual_amount_paid', 'TRY_CAST({c} AS DOUBLE)')}  AS actual_amount_paid,
                   {t.col('is_auto_renew', 'TRY_CAST({c} AS INTEGER)')}      AS is_auto_renew,
                   {date_expr('"transaction_date"')}                         AS transaction_date,
                   {date_expr('"membership_expire_date"')}                   AS membership_expire_date,
                   TRY_CAST("is_cancel" AS INTEGER)                          AS is_cancel
            FROM {t.sql}
          )
          WHERE transaction_date IS NOT NULL AND membership_expire_date IS NOT NULL
                AND is_cancel IN (0, 1)
        ) TO '{(out / 'transactions.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # --------------------------------------------------------------- user logs
    # Several files (v1, v2) are written as separate parquet parts. A later file only
    # contributes rows dated AFTER everything already written, so an overlapping
    # (cumulative) v2 file cannot double-count a day.
    for stale in out.glob("user_logs*.parquet"):
        stale.unlink()
    prev_max = None
    for i, lp in enumerate(paths["user_logs"]):
        l = Source(con, [lp], column_map.get("user_logs", {}), "user_logs")

        def count(name):
            return l.col(name, "GREATEST(COALESCE(TRY_CAST({c} AS DOUBLE), 0), 0)", "0")

        target = out / ("user_logs.parquet" if i == 0 else f"user_logs_part{i}.parquet")
        later_only = f"AND {DATE_COL} > DATE '{prev_max}'" if prev_max else ""
        con.execute(f"""
            COPY (
              SELECT msno,
                     {date_expr('"date"')}                                     AS date,
                     {count('num_25')} AS num_25, {count('num_50')} AS num_50,
                     {count('num_75')} AS num_75, {count('num_985')} AS num_985,
                     {count('num_100')} AS num_100, {count('num_unq')} AS num_unq,
                     LEAST(GREATEST(COALESCE(TRY_CAST("total_secs" AS DOUBLE), 0), 0), 86400) AS total_secs
              FROM {l.sql}
              WHERE {date_expr('"date"')} IS NOT NULL {later_only}
            ) TO '{target.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        prev_max = con.execute(
            f"SELECT max(date) FROM read_parquet('{(out / 'user_logs*.parquet').as_posix()}')"
        ).fetchone()[0]

    for name in ["members", "transactions", "user_logs"]:
        pq = (out / f"{name}*.parquet").as_posix()
        n = con.execute(f"SELECT count(*) FROM read_parquet('{pq}')").fetchone()[0]
        print(f"{name}: {n:,} rows -> {out}  (sources: {[p.name for p in paths[name]]})")
    r = con.execute(f"""
        SELECT min(transaction_date), max(transaction_date), min(membership_expire_date),
               max(membership_expire_date), avg(is_cancel), avg(is_auto_renew)
        FROM read_parquet('{(out / 'transactions.parquet').as_posix()}')""").fetchone()
    print(f"transactions: dates {r[0]} to {r[1]}; expiry dates {r[2]} to {r[3]}; "
          f"cancel share {r[4]:.3f}; auto-renew share {r[5]}")
    r = con.execute(f"SELECT min(date), max(date) FROM read_parquet('{(out / 'user_logs*.parquet').as_posix()}')").fetchone()
    print(f"user_logs: dates {r[0]} to {r[1]}")
    con.close()
    from .build_dataset import suggest_months
    print(f"suggested --months for the next step: {suggest_months(out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=DATA_RAW)
    ap.add_argument("--out", type=Path, default=DATA_PARQUET)
    ap.add_argument("--memory-limit", default="6GB")
    ap.add_argument("--map", type=Path, default=None, help="JSON column map (see docstring)")
    ap.add_argument("--members-file", type=Path)
    ap.add_argument("--transactions-file", type=Path)
    ap.add_argument("--logs-file", type=Path)
    a = ap.parse_args()
    cmap = json.loads(a.map.read_text()) if a.map else {}
    prepare(a.raw, a.out, a.memory_limit, cmap,
            {"members": a.members_file, "transactions": a.transactions_file,
             "user_logs": a.logs_file})
