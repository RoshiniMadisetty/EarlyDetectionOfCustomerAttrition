import numpy as np
import pandas as pd
import pytest

from silentchurn import events, series, ingest
from silentchurn.config import Config
from silentchurn.synth import generate

LOG_COLS = ["msno", "date", "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"]


def logs_df(rows):
    df = pd.DataFrame([(m, pd.Timestamp(d).date(), 0, 0, 0, 0, 0, 1, s) for m, d, s in rows], columns=LOG_COLS)
    return df


def make_sample(tmp_path, rows_v1, rows_v2=()):
    d = tmp_path / "s"
    d.mkdir(exist_ok=True)
    logs_df(rows_v1).to_parquet(d / "user_logs.parquet", index=False)
    logs_df(rows_v2).to_parquet(d / "user_logs_v2.parquet", index=False)
    return d


def test_matrix_placement_zero_fill_and_exclusion_of_expiry_day_and_later(tmp_path):
    d = make_sample(tmp_path,
                    [("a", "2017-02-14", 600), ("a", "2017-02-10", 60), ("a", "2017-02-15", 9999),   # expiry day: excluded
                     ("a", "2017-02-16", 9999), ("b", "2017-02-14", 60)],
                    [("a", "2017-03-01", 5)])                                                          # later than expiry: excluded
    ev = pd.DataFrame({"event_id": [0, 1, 2], "msno": ["a", "b", "zzz"], "expiry_date": pd.to_datetime(["2017-02-15"] * 3)})
    X, has, info = series.build_series(events.connect("1GB", 1), d, ev, L=10)
    assert X.shape == (3, 10)
    assert X[0, -1] == pytest.approx(np.log1p(10.0))           # day before expiry (k=1) is the last column; 600s = 10 min
    assert X[0, -5] == pytest.approx(np.log1p(1.0))            # Feb 10 -> k=5 -> column L-5
    assert (X[0, :-5] == 0).all() and (X[0, -4:-1] == 0).all()
    assert has.tolist() == [True, True, False] and X[2].sum() == 0
    assert info["log_rows_used"] == 3


def test_negative_and_over_24h_seconds_are_clipped_and_counted(tmp_path):
    d = make_sample(tmp_path, [("a", "2017-02-14", -50), ("a", "2017-02-13", 10**6)])
    ev = pd.DataFrame({"event_id": [0], "msno": ["a"], "expiry_date": pd.to_datetime(["2017-02-15"])})
    X, has, info = series.build_series(events.connect("1GB", 1), d, ev, L=5)
    assert X[0, -1] == 0 and X[0, -2] == pytest.approx(np.log1p(1440.0))
    assert info["out_of_range_secs_rows"] == 2


def test_save_and_load_roundtrip(tmp_path):
    X = np.random.rand(4, 6).astype(np.float32)
    series.save_series(tmp_path / "s.npz", X, np.array([1, 0, 1, 1], bool), [5, 6, 7, 8], 6)
    X2, h, ids, L = series.load_series(tmp_path / "s.npz")
    assert np.array_equal(X, X2) and ids.tolist() == [5, 6, 7, 8] and L == 6 and h.dtype == bool


def test_synthetic_planted_decay_is_visible_in_the_matrix(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    generate(raw, n_members=2500, seed=21)
    out = tmp_path_factory.mktemp("out")
    cfg = Config(raw, out, 100, memory_limit="1GB", threads=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ingest.run(cfg, verbose=False)
    con = events.connect("1GB", 1)
    ev = events.build_events(con, cfg.sample_dir)
    X, has, info = series.build_series(con, cfg.sample_dir, ev, L=120)
    base = X[:, 30:60].mean(axis=1)        # k = 90..61
    late = X[:, -14:].mean(axis=1)         # k = 14..1
    ok = has & (base > 0.5)
    drop = base - late
    ch = ok & (ev["is_churn"].to_numpy() == 1)
    rt = ok & (ev["is_churn"].to_numpy() == 0)
    assert ch.sum() > 30 and rt.sum() > 300
    assert np.median(drop[ch]) > np.median(drop[rt]) + 0.3
