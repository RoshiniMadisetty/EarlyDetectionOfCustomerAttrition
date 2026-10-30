"""Compare the churn labels rebuilt from transactions with the OFFICIAL label file.

  python -m src.check_labels --dataset outputs/dataset.parquet \
         --labels data/raw/train.csv --month 2017-02

The official file has columns msno, is_churn for members whose membership expired in
one month (KKBox: train.csv = February 2017, train_v2.csv = March 2017). Only members
with exactly ONE expiry event in that month in our dataset are compared, because the
official file has one label per member.

Report the agreement in the paper's data section. Public projects that rebuilt the
KKBox labels from the transaction files report roughly 97% to 99% agreement; anything
far below that means a different file version, a different month, or a column problem.
Note: the rebuilt labels can only cover months whose +30 day window is observed in the
files (the pipeline drops later months), so March 2017 labels cannot be rebuilt from
files that end on 2017-03-31.
"""
import argparse
from pathlib import Path

import pandas as pd

from .config import OUTPUTS


def main(a):
    ds = pd.read_parquet(a.dataset, columns=["event_id", "msno", "horizon", "expire_month", "is_churn"])
    h = a.horizon or int(ds["horizon"].min())
    ds = ds[ds["horizon"] == h].copy()
    ds["expire_month"] = pd.to_datetime(ds["expire_month"]).dt.strftime("%Y-%m")
    in_month = ds[ds["expire_month"] == a.month]
    if in_month.empty:
        raise SystemExit(f"no events with expiry month {a.month} at horizon {h}; "
                         f"available: {sorted(ds['expire_month'].unique())}")
    n_events = in_month.groupby("msno")["event_id"].transform("count")
    one = in_month[n_events == 1][["msno", "is_churn"]].rename(columns={"is_churn": "ours"})

    off = pd.read_csv(a.labels, usecols=["msno", "is_churn"]).rename(columns={"is_churn": "official"})
    off = off.drop_duplicates("msno")
    m = one.merge(off, on="msno", how="inner")
    if m.empty:
        raise SystemExit("no members in common; check the month and the label file")

    agree = float((m["ours"] == m["official"]).mean())
    conf = pd.crosstab(m["official"], m["ours"], rownames=["official"], colnames=["ours"])
    res = {"horizon": h, "month": a.month, "official_members": len(off),
           "our_single_event_members": len(one), "compared": len(m),
           "share_of_official_covered": len(m) / len(off),
           "agreement": agree, "official_churn_rate": float(m["official"].mean()),
           "our_churn_rate": float(m["ours"].mean())}
    print(pd.Series(res).to_string())
    print("\nconfusion matrix (rows official, columns ours):")
    print(conf.to_string())
    if agree < 0.97:
        print("\nWARNING: agreement below 97%. Check the label file/month, the file versions "
              "(transactions vs transactions_v2) and the date columns.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res]).to_csv(out / "label_check.csv", index=False)
    print(f"\nwrote {out / 'label_check.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(OUTPUTS / "dataset.parquet"))
    ap.add_argument("--labels", required=True, help="official CSV with msno, is_churn")
    ap.add_argument("--month", required=True, help="expiry month of the official labels, YYYY-MM")
    ap.add_argument("--horizon", type=int, default=None, help="default: smallest horizon")
    ap.add_argument("--out", default=str(OUTPUTS))
    main(ap.parse_args())
