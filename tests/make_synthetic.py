"""Generate a small synthetic dataset in the KKBox file format.

Purpose: test the pipeline mechanics (leakage, splits, metrics) when the real
files are not at hand. The numbers it produces mean nothing scientifically.

Files written (YYYYMMDD integer dates, same columns as the KKBox release):
  members.csv, transactions.csv, user_logs.csv

Usage:
  python -m tests.make_synthetic --out data/raw_synth --users 4000 --seed 7
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DAY0 = pd.Timestamp("2016-05-01")
END_IDX = (pd.Timestamp("2017-03-31") - DAY0).days  # last observed day

PRICE = {7: 35, 30: 149, 90: 399, 180: 699}


def to_yyyymmdd(idx):
    idx = np.asarray(idx, dtype="int64")
    return (DAY0 + pd.to_timedelta(idx, unit="D")).strftime("%Y%m%d").astype("int64")


def simulate(n_users: int, seed: int):
    rng = np.random.default_rng(seed)
    tx_rows, log_chunks, mem_rows = [], [], []

    for u in range(n_users):
        msno = f"u{u:06d}"
        engagement = rng.beta(2.0, 3.0)
        plan = int(rng.choice([30, 90, 7, 180], p=[0.78, 0.10, 0.04, 0.08]))
        auto = rng.random() < 0.65
        pay_method = int(rng.choice([21, 29, 36, 38, 40, 41]))
        t = int(rng.integers(0, 60))
        first_t = t

        while t <= END_IDX:
            exp = t + plan
            list_price = PRICE[plan]
            paid = list_price if rng.random() > 0.15 else int(list_price * 0.8)
            tx_rows.append((msno, pay_method, plan, list_price, paid, int(auto), t, exp, 0))

            p_churn = 0.04 if auto else 0.30
            p_churn *= float(np.clip(1.6 - 2.0 * engagement, 0.3, 1.6))
            churn = rng.random() < p_churn

            # daily activity within this cycle
            last = min(exp, END_IDX)
            days = np.arange(t, last + 1)
            p_act = 0.12 + 0.80 * engagement
            if churn:
                decay = 1.0 - 0.9 * np.clip((days - (exp - 25)) / 25.0, 0, 1)
                p_act = p_act * decay
            active = rng.random(len(days)) < p_act
            d_act = days[active]
            n = len(d_act)
            if n:
                scale = 400 + 2600 * engagement
                secs = rng.gamma(2.0, scale, n)
                plays = rng.poisson(8 + 30 * engagement, n) + 1
                n25 = rng.binomial(plays, 0.18)
                n50 = rng.binomial(plays - n25, 0.10)
                n75 = rng.binomial(plays - n25 - n50, 0.08)
                n985 = rng.binomial(plays - n25 - n50 - n75, 0.06)
                n100 = plays - n25 - n50 - n75 - n985
                unq = np.maximum(1, (plays * rng.uniform(0.5, 1.0, n)).astype(int))
                log_chunks.append(pd.DataFrame({
                    "msno": msno, "date": d_act, "num_25": n25, "num_50": n50,
                    "num_75": n75, "num_985": n985, "num_100": n100,
                    "num_unq": unq, "total_secs": secs,
                }))

            if churn and auto and rng.random() < 0.45:
                cdate = max(t + 1, exp - int(rng.integers(1, 20)))
                if cdate <= END_IDX:             # never create future-dated transactions
                    tx_rows.append((msno, pay_method, plan, list_price, paid, 0, cdate, exp, 1))

            if churn:
                if rng.random() < 0.15:           # comes back too late to count
                    t = exp + int(rng.integers(35, 90))
                    continue
                break
            nxt = exp + int(rng.integers(-3, 4))
            t = max(nxt, t + 1)

        reg = first_t - int(rng.integers(0, 400))
        gender = rng.choice(["male", "female", ""], p=[0.3, 0.2, 0.5])
        bd = int(rng.choice([0, 2016, int(np.clip(rng.normal(28, 8), 13, 70))], p=[0.15, 0.01, 0.84]))
        mem_rows.append((msno, int(rng.choice([1, 5, 13, 22, 4, 15])), bd, gender,
                         int(rng.choice([3, 4, 7, 9, 13])), reg))

    tx = pd.DataFrame(tx_rows, columns=[
        "msno", "payment_method_id", "payment_plan_days", "plan_list_price",
        "actual_amount_paid", "is_auto_renew", "transaction_date",
        "membership_expire_date", "is_cancel"])
    tx["transaction_date"] = to_yyyymmdd(tx["transaction_date"])
    tx["membership_expire_date"] = to_yyyymmdd(tx["membership_expire_date"])

    logs = pd.concat(log_chunks, ignore_index=True)
    logs["date"] = to_yyyymmdd(logs["date"])
    # a few corrupted values, as in the real file
    bad = rng.random(len(logs)) < 0.001
    logs.loc[bad, "total_secs"] = 9.2e12
    bad2 = rng.random(len(logs)) < 0.0005
    logs.loc[bad2, "total_secs"] = -123.0

    mem = pd.DataFrame(mem_rows, columns=[
        "msno", "city", "bd", "gender", "registered_via", "registration_init_time"])
    mem["registration_init_time"] = to_yyyymmdd(mem["registration_init_time"])
    return mem, tx, logs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw_synth")
    ap.add_argument("--users", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mem, tx, logs = simulate(a.users, a.seed)
    mem.to_csv(out / "members.csv", index=False)
    tx.to_csv(out / "transactions.csv", index=False)
    logs.to_csv(out / "user_logs.csv", index=False)
    print(f"wrote {len(mem)} members, {len(tx)} transactions, {len(logs)} log rows to {out}")
