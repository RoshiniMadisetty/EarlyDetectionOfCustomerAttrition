"""Step 1: stream the raw KKBox CSVs into Parquet, keeping ONLY sampled customers.

* Sampling is by CUSTOMER: a customer is kept iff  int(md5(msno)[:8], 16) % 100 < sample_pct.
  Every table uses the same rule, so a kept customer keeps their full history in every table.
  (md5 instead of DuckDB's internal hash() because DuckDB does not promise hash() is stable across
  versions; a Colab pip upgrade must never silently change the sample.)
* The filter is applied inside the CSV scan, so non-sampled rows are never written. DuckDB streams the
  CSV; peak memory is bounded by ``memory_limit`` and never holds a whole table in pandas.
* Resumable: each table is written to ``<name>.parquet.tmp`` and atomically renamed; existing finished
  tables are skipped. A crash only costs the table that was in flight.
* YYYYMMDD integer date columns are stored as DATE.
* Afterwards a sampling report compares the sample's churn rate with the FULL train.csv churn rate
  (train.csv is small, so the full rate is computed exactly) and prints date coverage per table.

Run:  python -m silentchurn.ingest --raw RAW_DIR --out OUT_DIR [--pct 10]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import duckdb

from .config import OPTIONAL_TABLES, RAW_FILES, REQUIRED_TABLES, Config

# --------------------------------------------------------------------------------------------
# table specs: logical name -> (raw column types, SELECT list). {src} is the CSV relation.
# --------------------------------------------------------------------------------------------
_DATE = "try_strptime(CAST({c} AS VARCHAR), '%Y%m%d')::DATE"

_TX_TYPES = {
    "msno": "VARCHAR", "payment_method_id": "INTEGER", "payment_plan_days": "INTEGER",
    "plan_list_price": "INTEGER", "actual_amount_paid": "INTEGER", "is_auto_renew": "INTEGER",
    "transaction_date": "BIGINT", "membership_expire_date": "BIGINT", "is_cancel": "INTEGER",
}
_TX_SELECT = (
    "msno, payment_method_id, payment_plan_days, plan_list_price, actual_amount_paid, is_auto_renew, "
    + _DATE.format(c="transaction_date") + " AS transaction_date, "
    + _DATE.format(c="membership_expire_date") + " AS membership_expire_date, is_cancel"
)
_LOG_TYPES = {
    "msno": "VARCHAR", "date": "BIGINT", "num_25": "INTEGER", "num_50": "INTEGER", "num_75": "INTEGER",
    "num_985": "INTEGER", "num_100": "INTEGER", "num_unq": "INTEGER", "total_secs": "DOUBLE",
}
_LOG_SELECT = (
    "msno, " + _DATE.format(c="date") + " AS date, num_25, num_50, num_75, num_985, num_100, num_unq, total_secs"
)
_TRAIN_TYPES = {"msno": "VARCHAR", "is_churn": "INTEGER"}

SPECS = {
    "train": (_TRAIN_TYPES, "msno, is_churn"),
    "train_v2": (_TRAIN_TYPES, "msno, is_churn"),
    "members": (
        {"msno": "VARCHAR", "city": "INTEGER", "bd": "INTEGER", "gender": "VARCHAR",
         "registered_via": "INTEGER", "registration_init_time": "BIGINT"},
        "msno, city, bd, gender, registered_via, " + _DATE.format(c="registration_init_time") + " AS registration_init_time",
    ),
    "transactions": (_TX_TYPES, _TX_SELECT),
    "transactions_v2": (_TX_TYPES, _TX_SELECT),
    "user_logs": (_LOG_TYPES, _LOG_SELECT),
    "user_logs_v2": (_LOG_TYPES, _LOG_SELECT),
}
# tables small enough that exact FULL distinct-customer counts are cheap (user_logs is skipped: 29 GB)
FULL_COUNT_TABLES = ("train", "train_v2", "members", "transactions", "transactions_v2")


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def in_sample(msno: str, pct: int) -> bool:
    """Pure-Python version of the sampling rule (used by tests as an independent check)."""
    return int(hashlib.md5(msno.encode("utf-8")).hexdigest()[:8], 16) % 100 < pct


def sample_predicate(pct: int, col: str = "msno") -> str:
    """SQL version of the same rule."""
    return f"(('0x' || substr(md5({col}), 1, 8))::BIGINT % 100) < {int(pct)}"


def _q(path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _cols_sql(types: Dict[str, str]) -> str:
    return "{" + ", ".join(f"'{k}': '{v}'" for k, v in types.items()) + "}"


def _csv_rel(path, types) -> str:
    return f"read_csv({_q(path)}, header=true, columns={_cols_sql(types)})"


def resolve_raw(cfg: Config, name: str) -> Optional[Path]:
    for cand in RAW_FILES[name]:
        p = Path(cfg.raw_dir) / cand
        if p.exists():
            return p
    return None


def connect(cfg: Config) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{cfg.memory_limit}'")
    con.execute(f"SET threads={int(cfg.threads)}")
    con.execute("SET preserve_insertion_order=false")  # lets COPY stream with lower memory
    if cfg.temp_dir:
        Path(cfg.temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory={_q(cfg.temp_dir)}")
    return con


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def _log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------------------------
def convert_table(con, cfg: Config, name: str) -> dict:
    """CSV -> sampled Parquet for one table. Returns a status dict."""
    src = resolve_raw(cfg, name)
    if src is None:
        status = "missing_required" if name in REQUIRED_TABLES else "missing_optional"
        return {"table": name, "status": status}
    cfg.sample_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.sample_dir / f"{name}.parquet"
    if out.exists():
        rows = con.execute(f"SELECT count(*) FROM read_parquet({_q(out)})").fetchone()[0]
        return {"table": name, "status": "skipped", "rows": rows, "path": str(out)}
    types, select = SPECS[name]
    tmp = out.with_suffix(".parquet.tmp")
    if tmp.exists():
        tmp.unlink()
    t0 = time.time()
    sql = (
        f"COPY (SELECT {select} FROM {_csv_rel(src, types)} WHERE {sample_predicate(cfg.sample_pct)}) "
        f"TO {_q(tmp)} (FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 500000)"
    )
    con.execute(sql)
    os.replace(tmp, out)
    rows = con.execute(f"SELECT count(*) FROM read_parquet({_q(out)})").fetchone()[0]
    return {"table": name, "status": "written", "rows": rows, "seconds": round(time.time() - t0, 1),
            "size_mb": round(out.stat().st_size / 1e6, 2), "path": str(out)}


# --------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------
def sampling_report(con, cfg: Config, full_counts: bool = True) -> dict:
    sd = cfg.sample_dir
    rep: dict = {"sample_pct": cfg.sample_pct, "rule": "int(md5(msno)[:8],16) % 100 < sample_pct", "train": {}, "customers": {},
                 "coverage": {}, "linkage": {}}

    def pq(name):
        p = sd / f"{name}.parquet"
        return _q(p) if p.exists() else None

    # --- churn rate: sample vs full ---
    for name in ("train", "train_v2"):
        p = pq(name)
        if not p:
            continue
        s_n, s_k = con.execute(f"SELECT count(*), coalesce(sum(is_churn),0) FROM read_parquet({p})").fetchone()
        lo, hi = wilson_ci(int(s_k), int(s_n))
        entry = {"sample_rows": int(s_n), "sample_churn_rate": (s_k / s_n) if s_n else float("nan"),
                 "sample_ci95": (lo, hi)}
        src = resolve_raw(cfg, name)
        if src is not None:
            f_n, f_k = con.execute(
                f"SELECT count(*), coalesce(sum(is_churn),0) FROM {_csv_rel(src, _TRAIN_TYPES)}").fetchone()
            f_rate = f_k / f_n if f_n else float("nan")
            entry.update({"full_rows": int(f_n), "full_churn_rate": f_rate,
                          "abs_diff_pp": 100 * abs(entry["sample_churn_rate"] - f_rate),
                          "full_rate_inside_sample_ci": bool(lo <= f_rate <= hi)})
        rep["train"][name] = entry

    # --- customer fractions ---
    for name in ("train", "train_v2", "members", "transactions", "transactions_v2", "user_logs", "user_logs_v2"):
        p = pq(name)
        if not p:
            continue
        s_d = con.execute(f"SELECT count(DISTINCT msno) FROM read_parquet({p})").fetchone()[0]
        entry = {"sample_distinct_customers": int(s_d)}
        src = resolve_raw(cfg, name)
        if full_counts and name in FULL_COUNT_TABLES and src is not None:
            f_d = con.execute(f"SELECT count(DISTINCT msno) FROM {_csv_rel(src, SPECS[name][0])}").fetchone()[0]
            entry.update({"full_distinct_customers": int(f_d),
                          "sampled_fraction_pct": 100 * s_d / f_d if f_d else float("nan")})
        rep["customers"][name] = entry

    # --- date coverage (tells us what the data really spans: do not assume) ---
    for name, cols in (("transactions", ("transaction_date", "membership_expire_date")),
                       ("transactions_v2", ("transaction_date", "membership_expire_date")),
                       ("user_logs", ("date",)), ("user_logs_v2", ("date",)), ("members", ("registration_init_time",))):
        p = pq(name)
        if not p:
            continue
        rep["coverage"][name] = {}
        for c in cols:
            lo, hi = con.execute(f"SELECT min({c}), max({c}) FROM read_parquet({p})").fetchone()
            rep["coverage"][name][c] = (str(lo), str(hi))

    # --- linkage of sampled train customers to the other tables ---
    if pq("train"):
        tr = pq("train")
        for name in ("members", "transactions", "transactions_v2", "user_logs", "user_logs_v2"):
            p = pq(name)
            if p:
                n = con.execute(f"SELECT count(*) FROM (SELECT DISTINCT msno FROM read_parquet({tr})) t "
                                f"WHERE msno IN (SELECT DISTINCT msno FROM read_parquet({p}))").fetchone()[0]
                rep["linkage"][f"train_customers_with_{name}"] = int(n)
    return rep


def print_report(rep: dict) -> None:
    print("\n=== SAMPLING REPORT ===")
    print(f"rule: {rep['rule']}   sample_pct={rep['sample_pct']}")
    for name, e in rep["train"].items():
        lo, hi = e["sample_ci95"]
        line = (f"{name}: sample n={e['sample_rows']:,} churn={100*e['sample_churn_rate']:.2f}% "
                f"(95% CI {100*lo:.2f}-{100*hi:.2f})")
        if "full_churn_rate" in e:
            line += (f" | FULL n={e['full_rows']:,} churn={100*e['full_churn_rate']:.2f}% | "
                     f"diff={e['abs_diff_pp']:.2f} pp | full rate inside sample CI: {e['full_rate_inside_sample_ci']}")
        print(line)
    print("customers (sample vs full):")
    for name, e in rep["customers"].items():
        s = f"  {name:16s} sample={e['sample_distinct_customers']:,}"
        if "full_distinct_customers" in e:
            s += f"  full={e['full_distinct_customers']:,}  fraction={e['sampled_fraction_pct']:.2f}%"
        print(s)
    print("date coverage in the SAMPLE:")
    for name, d in rep["coverage"].items():
        print(f"  {name:16s} " + "; ".join(f"{c}: {a} .. {b}" for c, (a, b) in d.items()))
    print("sampled train customers present in other tables:")
    for k, v in rep["linkage"].items():
        print(f"  {k}: {v:,}")


# --------------------------------------------------------------------------------------------
def run(cfg: Config, full_counts: bool = True, verbose: bool = True) -> dict:
    cfg.validate()
    con = connect(cfg)
    missing = [t for t in REQUIRED_TABLES if resolve_raw(cfg, t) is None]
    if missing:
        raise FileNotFoundError(f"Missing required raw files for {missing} in {cfg.raw_dir}. "
                                f"Expected names: { {t: RAW_FILES[t][0] for t in missing} }")
    results = []
    order = ["train", "train_v2", "members", "transactions", "transactions_v2", "user_logs_v2", "user_logs"]
    for name in order:
        if verbose:
            _log(f"{name}: starting")
        r = convert_table(con, cfg, name)
        results.append(r)
        if verbose:
            _log(f"{name}: {r['status']}" + (f", rows={r['rows']:,}" if "rows" in r else "")
                 + (f", {r.get('seconds')}s, {r.get('size_mb')} MB" if r["status"] == "written" else ""))
    if verbose:
        _log("building sampling report")
    rep = sampling_report(con, cfg, full_counts=full_counts)
    manifest = {"created": dt.datetime.now().isoformat(timespec="seconds"), "duckdb_version": duckdb.__version__,
                "sample_pct": cfg.sample_pct, "tables": results, "report": rep}
    (cfg.sample_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if verbose:
        print_report(rep)
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="CSV -> sampled Parquet (customer-level hash sample).")
    ap.add_argument("--raw", required=True, help="directory with the extracted Kaggle CSVs")
    ap.add_argument("--out", required=True, help="output base directory (e.g. on Google Drive)")
    ap.add_argument("--pct", type=int, default=10, help="percentage of customers to keep (5-15 recommended)")
    ap.add_argument("--memory", default="8GB")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tmp", default=None, help="DuckDB spill dir (local disk)")
    ap.add_argument("--no-full-counts", action="store_true", help="skip exact full-data distinct-customer counts")
    a = ap.parse_args(argv)
    cfg = Config(Path(a.raw), Path(a.out), a.pct, a.memory, a.threads, Path(a.tmp) if a.tmp else None)
    run(cfg, full_counts=not a.no_full_counts)


if __name__ == "__main__":
    main()
