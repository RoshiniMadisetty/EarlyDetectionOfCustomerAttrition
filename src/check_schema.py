"""Inspect the raw CSV files BEFORE running the pipeline.

  python -m src.check_schema --raw data/raw
  python -m src.check_schema --raw data/raw --write-map column_map.json

For each file it reports: the columns found, which standard columns are missing,
a suggested rename for each missing one, how well the date columns parse, and a
verdict (READY, or what must be mapped). With --write-map it saves the suggested
renames as a draft column map, which you MUST review before using (`--map`).
"""
import argparse
import difflib
import json
from pathlib import Path

import duckdb

from .prepare_data import REQUIRED, date_expr

EXPECTED = {
    "members": ["msno", "city", "bd", "gender", "registered_via", "registration_init_time"],
    "transactions": ["msno", "payment_method_id", "payment_plan_days", "plan_list_price",
                     "actual_amount_paid", "is_auto_renew", "transaction_date",
                     "membership_expire_date", "is_cancel"],
    "user_logs": ["msno", "date", "num_25", "num_50", "num_75", "num_985", "num_100",
                  "num_unq", "total_secs"],
}
DATE_COLS = {"members": ["registration_init_time"],
             "transactions": ["transaction_date", "membership_expire_date"],
             "user_logs": ["date"]}
SYNONYMS = {
    "msno": ["user_id", "userid", "customer_id", "customerid", "member_id", "memberid", "uid", "id", "cust_id"],
    "registration_init_time": ["registration_date", "registered_on", "signup_date", "join_date", "created_at"],
    "gender": ["sex"], "bd": ["age", "birth_year"],
    "registered_via": ["registration_channel", "channel", "signup_channel"],
    "payment_method_id": ["payment_method", "pay_method", "payment_type"],
    "payment_plan_days": ["plan_days", "plan_length", "subscription_days", "duration_days"],
    "plan_list_price": ["list_price", "price", "plan_price"],
    "actual_amount_paid": ["amount_paid", "paid", "amount", "payment_amount"],
    "is_auto_renew": ["auto_renew", "autorenew", "auto_renewal", "is_autorenew"],
    "transaction_date": ["start_date", "purchase_date", "payment_date", "txn_date", "order_date", "subscription_start"],
    "membership_expire_date": ["expiry_date", "expire_date", "expiration_date", "end_date",
                               "subscription_end", "valid_until", "membership_expiry_date"],
    "is_cancel": ["cancelled", "canceled", "is_cancelled", "is_canceled", "cancel_flag", "cancel"],
    "date": ["log_date", "activity_date", "day", "dt", "event_date"],
    "total_secs": ["listen_secs", "total_seconds", "seconds", "secs", "listening_time", "duration", "total_listen_time"],
    "num_unq": ["unique_songs", "n_unique", "unique_tracks"],
}


def classify(name: str):
    n = name.lower()
    for key, table in (("member", "members"), ("transac", "transactions"),
                       ("log", "user_logs"), ("train", "train")):
        if key in n:
            return table
    return None


def suggest(missing, cols):
    out = {}
    lower = {c.lower(): c for c in cols}
    for m in missing:
        hit = next((lower[s] for s in SYNONYMS.get(m, []) if s in lower), None)
        if not hit:
            close = difflib.get_close_matches(m.lower(), list(lower), n=1, cutoff=0.7)
            hit = lower[close[0]] if close else None
        out[m] = hit
    return out


def main(a):
    raw = Path(a.raw)
    con = duckdb.connect()
    archives = [p.name for p in raw.glob("*") if p.suffix in (".7z", ".zip", ".gz", ".rar")]
    if archives:
        print(f"NOTE: compressed files found ({archives}). Extract them to .csv first.\n")
    files = sorted(p for p in raw.glob("*.csv"))
    if not files:
        print(f"No .csv files in {raw}. Put the dataset files there and run again.")
        return
    draft, verdicts = {}, {}
    for f in files:
        table = classify(f.name)
        print("=" * 78)
        print(f"{f.name}   ->   treated as: {table or 'UNKNOWN (ignored)'}")
        reader = f"read_csv('{f.as_posix()}', header=true, all_varchar=true)"
        cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {reader}").fetchall()]
        print(f"columns ({len(cols)}): {cols}")
        sample = con.execute(f"SELECT * FROM {reader} LIMIT 3").fetchdf()
        print(sample.to_string(index=False))
        if a.count:
            n = con.execute(f"SELECT count(*) FROM {reader}").fetchone()[0]
            print(f"rows: {n:,}")
        if table in (None, "train"):
            if table == "train":
                print("official labels file: not used by the pipeline; use "
                      "`python -m src.check_labels` to validate the rebuilt labels against it.")
            continue
        missing = [c for c in EXPECTED[table] if c not in cols]
        sug = suggest(missing, cols)
        req_missing = [c for c in REQUIRED[table] if c in missing]
        if missing:
            print("missing standard columns -> suggested rename from this file:")
            for m in missing:
                tag = "REQUIRED" if m in REQUIRED[table] else "optional"
                print(f"   {m:24s} ({tag}) <- {sug[m] or 'no match found'}")
        mapped = {v: k for k, v in sug.items() if v}
        if mapped:
            draft.setdefault(table, {}).update(mapped)
        # date parse check, using the suggested/own column names
        for dc in DATE_COLS[table]:
            src_col = dc if dc in cols else sug.get(dc)
            if not src_col:
                continue
            q = (f"SELECT avg(CASE WHEN {date_expr('d')} IS NOT NULL THEN 1.0 ELSE 0.0 END), "
                 f"min({date_expr('d')}), max({date_expr('d')}) "
                 f"FROM (SELECT \"{src_col}\" AS d FROM {reader} LIMIT 200000)")
            share, lo, hi = con.execute(q).fetchone()
            print(f"date column {dc} (file column '{src_col}'): {share:.1%} parse, "
                  f"range in first 200k rows {lo} to {hi}")
            if share is not None and share < 0.95:
                print("   WARNING: many values do not parse; check the date format")
        unresolved = [c for c in req_missing if not sug[c]]
        verdicts[f.name] = ("READY" if not req_missing else
                            ("NEEDS COLUMN MAP (review the suggestions)" if not unresolved
                             else f"CANNOT PROCEED: no match for required {unresolved}"))
        print(f"VERDICT: {verdicts[f.name]}")
    print("=" * 78)
    if draft:
        print("draft column map (review every line before using it):")
        print(json.dumps(draft, indent=2))
        if a.write_map:
            Path(a.write_map).write_text(json.dumps(draft, indent=2))
            print(f"saved to {a.write_map}; use it with: python -m src.prepare_data --map {a.write_map}")
    found = {classify(f.name) for f in files}
    for t in ("members", "transactions", "user_logs"):
        if t not in found:
            print(f"MISSING FILE: no file that looks like '{t}' (name should contain "
                  f"'{ {'members': 'member', 'transactions': 'transac', 'user_logs': 'log'}[t] }')")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--write-map", default=None)
    ap.add_argument("--count", action="store_true", help="also count rows (slow on huge files)")
    main(ap.parse_args())
