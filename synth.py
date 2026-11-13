"""Synthetic dataset in the SAME FILE FORMAT as the KKBox churn data (for tests and dry runs).

Writes into ``raw_dir``: members_v3.csv, transactions.csv, transactions_v2.csv, user_logs.csv,
user_logs_v2.csv, train.csv  (+ ``_synth_truth.csv``, which does NOT exist in the real data).

What is planted (so later pipeline stages can be tested against known ground truth):
  * monthly/quarterly plans; auto-renew and manual-renew members (manual ones sometimes renew late);
  * churn = no non-cancelled renewal more than 30 days after expiry (the Kaggle definition);
  * 75% of churners show a gradual engagement decay over D in [10, 35] days before expiry,
    25% churn "abruptly" with no decay (hard cases);
  * ~half of auto-renew churners post an explicit cancel row (is_cancel=1); the rest are SILENT;
  * 12% of retained events have a temporary engagement dip (false-alarm cases);
  * ~3% of retained events contain a cancel row followed by a real renewal (so is_cancel alone is
    not the label);
  * user_logs only contain days with activity (missing day = no listening), weekend uplift.

This is a toy world: it validates plumbing and leakage logic, NOT any scientific claim.
Real files differ in size, noise and quirks; nothing here is evidence about the real data.
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

import numpy as np
import pandas as pd

EPOCH = np.datetime64("1970-01-01")


def _d(s: str) -> int:
    return int((np.datetime64(s) - EPOCH) / np.timedelta64(1, "D"))


def _ymd(days) -> np.ndarray:
    """Epoch-day integers -> YYYYMMDD integers."""
    days = np.asarray(days, dtype="int64")
    if days.size == 0:
        return days
    return pd.to_datetime(days, unit="D").strftime("%Y%m%d").astype("int64").to_numpy()


D_START = _d("2015-10-01")
D_JOIN_MAX = _d("2016-08-01")
D_V1_END = _d("2017-02-28")  # last day in transactions.csv / user_logs.csv
D_END = _d("2017-03-31")  # last day in *_v2 files
FEB_START = _d("2017-02-01")
CHURN_GAP_DAYS = 30


def _msnos(rng: np.random.Generator, n: int) -> list:
    out = set()
    while len(out) < n:
        out.add(base64.b64encode(rng.bytes(33)).decode())
    out = sorted(out)
    rng.shuffle(out)
    return out


def generate(
    raw_dir,
    n_members: int = 2000,
    seed: int = 7,
    p_auto: float = 0.6,
    p_churn_auto: float = 0.07,
    p_churn_manual: float = 0.22,
) -> dict:
    """Generate the synthetic raw files. Returns a small summary dict."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    msnos = _msnos(rng, n_members)

    members, tx_rows, log_frames, truth = [], [], [], []

    for msno in msnos:
        auto = rng.random() < p_auto
        plan = 30 if rng.random() < 0.9 else 90
        price = 149 if plan == 30 else 447
        start = int(rng.integers(D_START, D_JOIN_MAX))
        p_churn = p_churn_auto if auto else p_churn_manual

        members.append(
            (
                msno,
                int(rng.choice([1, 4, 5, 13, 15, 22])),
                int(0 if rng.random() < 0.5 else rng.integers(15, 60)),
                str(rng.choice(["male", "female", ""])),
                int(rng.choice([3, 4, 7, 9])),
                int(_ymd([start])[0]),
            )
        )

        events = []  # per expiry: dict(expire, churn, decay_d, dip, next_start)
        txn_day = start
        active_until = D_END  # logs end here unless the member churns for good
        gaps = []  # (from_day, to_day) windows with no listening (churn gaps)
        while txn_day <= D_END:
            expire = txn_day + plan
            tx_rows.append((msno, int(rng.choice([38, 40, 41])), plan, price, price, int(auto), txn_day, expire, 0))
            churn = rng.random() < p_churn
            ev = dict(expire=expire, churn=churn, decay_d=0, dip=None, cancel=False, auto=auto)
            if churn:
                if rng.random() < 0.75:
                    ev["decay_d"] = int(rng.integers(10, 36))
                if auto and rng.random() < 0.5:
                    ev["cancel"] = True
                    cday = expire - int(rng.integers(1, 11))
                    tx_rows.append((msno, int(rng.choice([38, 40, 41])), plan, price, price, 1, cday, expire, 1))
                if rng.random() < 0.15:  # returns after > 30 days
                    nxt = expire + int(rng.integers(CHURN_GAP_DAYS + 1, 120))
                    gaps.append((expire + int(rng.integers(0, 4)), nxt - int(rng.integers(0, 3))))
                    events.append(ev)
                    txn_day = nxt
                    continue
                gaps.append((expire + int(rng.integers(0, 4)), D_END + 1))
                events.append(ev)
                break
            # retained
            if rng.random() < 0.12:
                ev["dip"] = (expire - int(rng.integers(20, 40)), int(rng.integers(7, 15)))
            if rng.random() < 0.03:
                tx_rows.append((msno, int(rng.choice([38, 40, 41])), plan, price, price, int(auto), expire - 5, expire, 1))
            events.append(ev)
            if auto:
                txn_day = expire
            else:
                late = int(rng.integers(1, 26)) if rng.random() < 0.10 else int(rng.integers(-3, 3))
                txn_day = expire + late
        # ---- daily engagement ----
        days = np.arange(start, D_END + 1)
        factor = np.ones(days.size)
        alive = np.ones(days.size, dtype=bool)
        for ev in events:
            e = ev["expire"]
            if ev["decay_d"]:
                d = ev["decay_d"]
                lo = e - d
                m = (days >= lo) & (days < e)
                factor[m] *= 1.0 - 0.9 * ((days[m] - lo + 1) / d)
            if ev["dip"] is not None:
                s, ln = ev["dip"]
                m = (days >= s) & (days < s + ln)
                factor[m] *= 0.35
        for g0, g1 in gaps:
            alive[(days >= g0) & (days < g1)] = False
        weekend = ((days + 3) % 7) >= 5
        p_i = rng.beta(5, 2)
        mu_i = rng.lognormal(np.log(5000), 0.5)
        p_act = np.clip(p_i * np.sqrt(np.clip(factor, 0, 1)), 0, 1)
        active = alive & (rng.random(days.size) < p_act)
        di = days[active]
        if di.size:
            f = factor[active] * np.where(weekend[active], 1.15, 1.0)
            secs = mu_i * rng.lognormal(0, 0.6, di.size) * f
            n100 = rng.poisson(secs / 220.0)
            log_frames.append(
                pd.DataFrame(
                    {
                        "msno": msno,
                        "date": _ymd(di),
                        "num_25": rng.poisson(0.15 * n100 + 0.5),
                        "num_50": rng.poisson(0.05 * n100 + 0.2),
                        "num_75": rng.poisson(0.04 * n100 + 0.1),
                        "num_985": rng.poisson(0.05 * n100 + 0.1),
                        "num_100": n100,
                        "num_unq": np.maximum(1, np.round(n100 * 0.8 + rng.normal(0, 1, di.size))).astype(int),
                        "total_secs": np.round(secs, 3),
                    }
                )
            )
        for ev in events:
            truth.append((msno, ev["expire"], int(ev["churn"]), int(ev["churn"] and not ev["cancel"]),
                          ev["decay_d"], int(ev["auto"])))

    # ---- write CSVs in the real column order ----
    pd.DataFrame(members, columns=["msno", "city", "bd", "gender", "registered_via", "registration_init_time"]).to_csv(
        raw_dir / "members_v3.csv", index=False)

    tx = pd.DataFrame(tx_rows, columns=["msno", "payment_method_id", "payment_plan_days", "plan_list_price",
                                        "actual_amount_paid", "is_auto_renew", "transaction_date",
                                        "membership_expire_date", "is_cancel"])
    tx["_td"] = tx["transaction_date"]
    for c in ("transaction_date", "membership_expire_date"):
        tx[c] = _ymd(tx[c].to_numpy())
    cols = [c for c in tx.columns if c != "_td"]
    tx[tx["_td"] <= D_V1_END][cols].to_csv(raw_dir / "transactions.csv", index=False)
    tx[(tx["_td"] > D_V1_END) & (tx["_td"] <= D_END)][cols].to_csv(raw_dir / "transactions_v2.csv", index=False)

    logs = pd.concat(log_frames, ignore_index=True)
    dates = pd.to_datetime(logs["date"].astype(str), format="%Y%m%d")
    v1 = dates <= pd.Timestamp("2017-02-28")
    logs[v1].to_csv(raw_dir / "user_logs.csv", index=False)
    logs[~v1].to_csv(raw_dir / "user_logs_v2.csv", index=False)

    # train.csv: customers with an expiry in Feb 2017, label by the 30-day rule (truth by construction).
    truth_df = pd.DataFrame(truth, columns=["msno", "expire_day", "is_churn", "is_silent", "decay_days", "is_auto_renew"])
    truth_df["expire_date"] = _ymd(truth_df["expire_day"].to_numpy())
    truth_df["in_train"] = ((truth_df["expire_day"] >= FEB_START) & (truth_df["expire_day"] <= D_V1_END)).astype(int)
    train = truth_df[truth_df["in_train"] == 1].sort_values("expire_day").drop_duplicates("msno", keep="last")
    train[["msno", "is_churn"]].to_csv(raw_dir / "train.csv", index=False)
    truth_df.drop(columns=["expire_day"]).to_csv(raw_dir / "_synth_truth.csv", index=False)

    return {
        "n_members": n_members,
        "n_transactions": int(len(tx)),
        "n_log_rows": int(len(logs)),
        "n_train": int(len(train)),
        "train_churn_rate": float(train["is_churn"].mean()) if len(train) else float("nan"),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic KKBox-format dataset.")
    ap.add_argument("--out", required=True, help="directory for the synthetic raw CSVs")
    ap.add_argument("--n-members", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    print(generate(a.out, n_members=a.n_members, seed=a.seed))


if __name__ == "__main__":
    main()
