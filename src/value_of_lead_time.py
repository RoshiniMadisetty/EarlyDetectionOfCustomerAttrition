"""Experiment E4: Value of Lead Time (a profit-based view of early warning).

Policy: rank the test-month customers by calibrated churn probability and contact
the top k%. For every contacted customer i:
    cost_i = offer_cost_frac * renewal_value_i                 (paid whether or not they churn)
    gain_i = p_save_eff(h) * renewal_value_i * [customer i is a true churner]
    profit(h, k) = sum(gain_i) - sum(cost_i) over the contacted set

p_save_eff(h) is the assumed probability that an offer retains a churner:
    lead_effect = "flat"          : p_save_eff = p_save at every horizon
    lead_effect = "proportional"  : p_save_eff = p_save * min(1, h / h_ref)
                                    (earlier warnings leave more time to act)
Both are ASSUMPTIONS, varied in a sensitivity grid. This scores a targeting policy
under stated costs; it does not estimate the causal effect of an offer.
"profit_random" is the expected profit of contacting the same number of customers
at random, so profit_gain_vs_random isolates what the model adds.

Outputs (outputs/): value_of_lead_time.csv, value_of_lead_time_best.csv
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUTS

K_GRID = list(range(1, 31))
P_SAVE_GRID = [0.1, 0.3, 0.5]
COST_GRID = [0.02, 0.05, 0.1, 0.2]
EFFECTS = ["flat", "proportional"]


def main(a):
    out = Path(a.out)
    preds = pd.read_parquet(Path(a.predictions))
    h_ref = a.h_ref or int(preds["horizon"].max())
    rows = []
    for h, d in preds.groupby("horizon"):
        d = d.sort_values("p_cal", ascending=False, kind="stable")
        y = d["is_churn"].to_numpy(float)
        v = d["renewal_value"].to_numpy(float)
        N, pos = len(d), y.sum()
        cum_tp = np.cumsum(y)
        cum_val_churn = np.cumsum(v * y)
        cum_val = np.cumsum(v)
        mean_gain_unit = float((v * y).mean())   # E[value * churn] for random contact
        mean_val = float(v.mean())
        for effect in EFFECTS:
            for ps in P_SAVE_GRID:
                ps_eff = ps if effect == "flat" else ps * min(1.0, h / h_ref)
                for cf in COST_GRID:
                    for k in K_GRID:
                        m = int(math.ceil(k / 100 * N))
                        gain = ps_eff * cum_val_churn[m - 1]
                        cost = cf * cum_val[m - 1]
                        profit = gain - cost
                        rand = m * (ps_eff * mean_gain_unit - cf * mean_val)
                        rows.append({
                            "horizon": int(h), "lead_effect": effect, "p_save": ps,
                            "p_save_effective": ps_eff, "offer_cost_frac": cf, "k_pct": k,
                            "n_contacted": m, "true_positives": float(cum_tp[m - 1]),
                            "precision": float(cum_tp[m - 1] / m),
                            "recall": float(cum_tp[m - 1] / pos) if pos else np.nan,
                            "profit": float(profit), "profit_random": float(rand),
                            "profit_gain_vs_random": float(profit - rand),
                            "profit_per_contact": float(profit / m)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "value_of_lead_time.csv", index=False)

    keys = ["lead_effect", "p_save", "offer_cost_frac", "horizon"]
    best = res.loc[res.groupby(keys)["profit"].idxmax()].copy()
    # contacting nobody (profit 0) beats any policy with negative profit
    best["contact_nobody"] = best["profit"] < 0
    zero_cols = ["k_pct", "n_contacted", "true_positives", "profit", "profit_random",
                 "profit_gain_vs_random", "profit_per_contact"]
    best.loc[best["contact_nobody"], zero_cols] = 0
    best.loc[best["contact_nobody"], ["precision", "recall"]] = np.nan
    # marginal profit change per extra day of lead time between adjacent horizons
    best = best.sort_values(["lead_effect", "p_save", "offer_cost_frac", "horizon"])
    g = best.groupby(["lead_effect", "p_save", "offer_cost_frac"])
    best["profit_change_per_extra_day"] = g["profit"].diff() / g["horizon"].diff()
    best.to_csv(out / "value_of_lead_time_best.csv", index=False)
    print(best[best.p_save.eq(0.3) & best.offer_cost_frac.eq(0.2)]
          [["lead_effect", "horizon", "k_pct", "profit", "profit_gain_vs_random"]]
          .to_string(index=False))
    print(f"wrote value_of_lead_time*.csv to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(OUTPUTS / "predictions.parquet"))
    ap.add_argument("--out", default=str(OUTPUTS))
    ap.add_argument("--h-ref", type=int, default=None,
                    help="horizon at which an offer is fully effective (default: largest)")
    main(ap.parse_args())
