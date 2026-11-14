import numpy as np
import pandas as pd
import pytest

from silentchurn import usability as us

L = 120


def pool(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    level = rng.uniform(1.0, 4.0, size=(n, 1))
    X = np.maximum(level + rng.normal(0, 1.0, size=(n, L)), 0).astype(np.float32)
    y = (rng.random(n) < 0.3).astype(int)
    ev = pd.DataFrame({"event_id": range(n), "is_churn": y, "silent": y, "msno": [f"c{i}" for i in range(n)]})
    return X, ev, np.ones(n, bool)


def test_latin_square_is_balanced_for_multiples_of_four():
    o = us.latin_square_orders(8)
    for pos in range(1, 5):
        assert o[o.position == pos]["condition"].value_counts().nunique() == 1       # each condition equally often per position
    # first-order carry-over balance: each ordered pair (a -> b) occurs equally often
    pairs = []
    for p, g in o.groupby("participant"):
        c = g.sort_values("position")["condition"].tolist()
        pairs += list(zip(c[:-1], c[1:]))
    counts = pd.Series(pairs).value_counts()
    assert counts.nunique() == 1 and len(counts) == 12


def test_trial_sets_have_two_churners_and_matched_retained_and_no_reuse():
    X, ev, ok = pool()
    sets = us.make_trials(ev, X, ok, L, seed=1)
    assert len(sets) == 24 and sets["trial_id"].is_unique
    seen = set()
    base = X[:, 30:60].mean(axis=1)
    for r in sets.itertuples():
        rows = r.panel_rows
        assert len(rows) == 6 and len(set(rows)) == 6 and not (set(rows) & seen)
        seen |= set(rows)
        y = ev["is_churn"].to_numpy()[rows]
        assert y.sum() == 2
        assert sorted(np.array(list("ABCDEF"))[y == 1]) == sorted(r.at_risk_positions)


def test_participant_sheets_balance_sets_over_conditions():
    X, ev, ok = pool()
    sets = us.make_trials(ev, X, ok, L, seed=2)
    sh = us.participant_sheets(sets, 8)
    assert len(sh) == 8 * 24 and sh.groupby("participant").size().eq(24).all()
    assert sh.groupby(["participant", "condition"]).size().eq(6).all()
    per_set = sh.groupby(["trial_id", "condition"]).size().unstack()
    assert (per_set == 2).all().all()                      # 8 participants / 4 conditions = each set in each condition twice
    first = sh[sh.position == 1]
    assert first["condition"].nunique() == 4


def test_not_enough_data_raises():
    X, ev, ok = pool(60)
    with pytest.raises(ValueError):
        us.make_trials(ev, X, ok, L)


def test_render_creates_images_for_all_conditions(tmp_path):
    X, ev, ok = pool(600)
    sets = us.make_trials(ev, X, ok, L, trials_per_condition=1, seed=3)
    files = us.render_all(tmp_path, sets, X, L)
    assert len(files) == 4 * 4 and all((tmp_path / f.split("/")[-1]).stat().st_size > 5000 for f in files)
    key = us.answer_key(sets)
    assert key["at_risk_positions"].str.count(",").eq(1).all()


def test_analysis_recovers_planted_effects():
    df = us.simulate_study(n_participants=14, seed=3)
    res = us.analyze(df, B=2000)
    c = res["contrasts"]
    acc = c[c.metric == "accuracy"].set_index("condition")
    assert acc.loc["smoothed_band_cusum", "mean_diff"] > acc.loc["smoothed", "mean_diff"] > 0
    assert acc.loc["smoothed_band_cusum", "ci_lo"] > 0                      # planted +0.20 is detectable with 14 participants
    assert (acc["ci_lo"] <= acc["mean_diff"]).all() and (acc["mean_diff"] <= acc["ci_hi"]).all()
    assert set(res["summary"]["condition"]) == set(us.CONDITIONS)
    assert res["omnibus"].set_index("metric").loc["accuracy", "p"] < 0.05


def test_analysis_with_no_effect_has_ci_covering_zero():
    df = us.simulate_study(n_participants=12, seed=9, effects={c: 0.0 for c in us.CONDITIONS})
    acc = us.analyze(df, B=2000)["contrasts"].query("metric == 'accuracy'")
    assert ((acc["ci_lo"] < 0) & (acc["ci_hi"] > 0)).all()
