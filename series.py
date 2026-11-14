"""Step 3: per-event daily engagement matrix.

X[i, j] = log(1 + minutes listened) on the calendar day  expiry_i - (L - j)   (j = 0..L-1, so the LAST column is the
day before expiry, k = days before expiry = L - j). Days with no log row are 0 (no listening). total_secs is clipped to
[0, 86400] (the real data has negative and >24h values; counts are reported).
Only days strictly before the expiry date are ever read, and the matrix is the only thing later stages see, so the
smoothers/features are leakage-safe by construction plus the explicit tests in smoothers.py / model.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .events import _q


def build_series(con, sample_dir, events: pd.DataFrame, L: int = 120):
    sd = Path(sample_dir)
    ev = events[["event_id", "msno", "expiry_date"]].copy()
    ev["expiry_date"] = pd.to_datetime(ev["expiry_date"]).dt.date
    ev["row"] = np.arange(len(ev))
    con.register("ev_tmp", ev)
    logs = f"(SELECT msno, date, total_secs FROM read_parquet({_q(sd / 'user_logs.parquet')}) " \
           f"UNION ALL SELECT msno, date, total_secs FROM read_parquet({_q(sd / 'user_logs_v2.parquet')}))"
    q = f"""SELECT e.row AS row, date_diff('day', l.date, e.expiry_date) AS k,
                   least(greatest(l.total_secs, 0), 86400) AS secs, (l.total_secs < 0 OR l.total_secs > 86400) AS bad
            FROM ev_tmp e JOIN {logs} l ON l.msno = e.msno
            WHERE l.date >= e.expiry_date - {int(L)} AND l.date <= e.expiry_date - 1"""
    r = con.execute(q).fetchnumpy()
    con.unregister("ev_tmp")
    X = np.zeros((len(ev), L), dtype=np.float32)
    j = L - r["k"].astype(np.int64)
    X[r["row"].astype(np.int64), j] = np.log1p(r["secs"].astype(np.float64) / 60.0).astype(np.float32)
    has_log = np.zeros(len(ev), dtype=bool)
    has_log[np.unique(r["row"].astype(np.int64))] = True
    info = {"log_rows_used": int(len(r["row"])), "out_of_range_secs_rows": int(np.asarray(r["bad"]).sum()),
            "events_with_any_log_in_window": int(has_log.sum()), "events": int(len(ev))}
    return X, has_log, info


def save_series(path, X, has_log, event_ids, L):
    np.savez_compressed(path, X=X, has_log=has_log, event_ids=np.asarray(event_ids), L=np.array(L))


def load_series(path):
    z = np.load(path)
    return z["X"], z["has_log"], z["event_ids"], int(z["L"])
