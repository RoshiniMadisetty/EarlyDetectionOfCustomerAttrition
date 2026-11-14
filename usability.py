"""RQ2 usability study: do uncertainty bands and annotated change-points make at-risk customers easier to spot?

Design (within-subject, 8-15 participants, pilot scale):
  * 4 display conditions: raw | smoothed | smoothed_band | smoothed_band_cusum (band + change-point annotation)
  * each trial: 6 customers (panels A-F), 2 of whom churned (silent churners), shown ONLY up to cutoff = expiry - h days
  * task: tick the 2 customers most likely to leave; measures: n_correct (0-2), time_s, confidence (1-5)
  * 24 trials per participant = 6 per condition; condition order counterbalanced with a Latin square
Analysis: participant-level means, paired differences vs the raw condition with bootstrap CIs over participants,
Cohen's dz, Wilcoxon signed-rank and Friedman omnibus (exploratory: with n <= 15 report effect sizes and CIs, not 'significance').
The 'band' is a DESCRIPTIVE variability band (EMA +/- 1.96 * trailing 14-day sd / sqrt(14)), not a confidence interval of the
customer's true level; say so in the paper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import smoothers as sm
from .benchmark import baseline, MIN_BASELINE

CONDITIONS = ["raw", "smoothed", "smoothed_band", "smoothed_band_cusum"]
LABELS = list("ABCDEF")


def latin_square_orders(n_participants: int, conditions=CONDITIONS) -> pd.DataFrame:
    """Balanced Latin square (Williams design for an even number of conditions) cycled over participants."""
    n = len(conditions)
    seq, lo, hi = [0], 1, n - 1
    for i in range(1, n):
        seq.append(lo if i % 2 == 1 else hi)
        lo, hi = (lo + 1, hi) if i % 2 == 1 else (lo, hi - 1)
    rows = []
    for p in range(n_participants):
        shift = p % n
        order = [conditions[(s + shift) % n] for s in seq]
        rows += [{"participant": f"P{p + 1:02d}", "position": i + 1, "condition": c} for i, c in enumerate(order)]
    return pd.DataFrame(rows)


def make_trials(ev: pd.DataFrame, X, has_log, L: int, trials_per_condition: int = 6, seed: int = 0, h: int = 14) -> pd.DataFrame:
    """Trial sheet. Each trial: 2 silent churners + 4 retained customers from the same baseline-activity tercile.
    Condition assignment is done per participant in ``participant_sheets``."""
    rng = np.random.default_rng(seed)
    base = baseline(X, L)
    ok = np.asarray(has_log, bool) & (base >= MIN_BASELINE)
    y, silent = ev["is_churn"].to_numpy(), ev["silent"].to_numpy()
    ch_pool = np.where(ok & (y == 1) & (silent == 1))[0]
    rt_pool = np.where(ok & (y == 0))[0]
    n_sets = trials_per_condition * len(CONDITIONS)
    if len(ch_pool) < 2 * n_sets or len(rt_pool) < 4 * n_sets:
        raise ValueError("not enough analysable churners/retained customers for the requested number of trials")
    terc = np.quantile(base[ok], [1 / 3, 2 / 3])
    tb = lambda idx: np.digitize(base[idx], terc)
    ch_perm = rng.permutation(ch_pool)
    used = set()
    rows = []
    ci = 0
    for s in range(n_sets):
        pair = ch_perm[ci: ci + 2]; ci += 2
        pool_t = tb(rt_pool)
        chosen = []
        for c in pair:
            cand = [r for r in rt_pool[pool_t == tb(np.array([c]))[0]] if r not in used]
            take = rng.choice(cand, size=2, replace=False)
            chosen += list(take); used.update(take)
        panel = list(pair) + chosen
        order = rng.permutation(6)
        rows.append({"set_id": s + 1, "panel_rows": [int(panel[o]) for o in order],
                     "event_ids": [int(ev["event_id"].iloc[panel[o]]) for o in order],
                     "at_risk_positions": [LABELS[i] for i, o in enumerate(order) if o < 2], "cutoff_days_before_expiry": h})
    sets = pd.DataFrame(rows)
    sets["trial_id"] = [f"T{i + 1:02d}" for i in range(n_sets)]
    return sets


def participant_sheets(sets: pd.DataFrame, n_participants: int, trials_per_condition: int = 6) -> pd.DataFrame:
    """Which set each participant sees, in which condition and order. Set s is shown in condition (s + p) mod 4, so every
    set appears in every condition equally often across participants (set difficulty is not confounded with condition);
    conditions are presented in blocks whose order follows the Latin square; trials inside a block are shuffled."""
    orders = latin_square_orders(n_participants)
    rng = np.random.default_rng(12345)
    rows = []
    for p in range(n_participants):
        cond_of = {s: CONDITIONS[(s + p) % len(CONDITIONS)] for s in range(len(sets))}
        pos = 1
        for c in orders[orders.participant == f"P{p + 1:02d}"].sort_values("position")["condition"]:
            members = [s for s, cc in cond_of.items() if cc == c]
            for s in rng.permutation(members):
                rows.append({"participant": f"P{p + 1:02d}", "position": pos, "condition": c, "trial_id": sets["trial_id"].iloc[s]})
                pos += 1
    return pd.DataFrame(rows)


def render_trial(path, X, panel_rows: List[int], condition: str, L: int, h: int = 14, show_days: int = 60):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = np.expm1(X[panel_rows].astype(np.float64)).astype(np.float32)
    ema = sm.ema(M, 0.15)
    _, first = sm.cusum_down(X[panel_rows], L)
    k = np.arange(L, 0, -1)
    vis = (k >= h) & (k <= show_days + h)
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for i, ax in enumerate(axs.ravel()):
        kk, raw, e = k[vis], M[i][vis], ema[i][vis]
        ax.plot(kk, raw, color="#9CA3AF" if condition != "raw" else "#0072B2", lw=1, label="daily minutes")
        if condition != "raw":
            ax.plot(kk, e, color="#0072B2", lw=2, label="smoothed (EMA)")
        if condition in ("smoothed_band", "smoothed_band_cusum"):
            sd = pd.Series(M[i]).rolling(14, min_periods=7).std().to_numpy()[vis]
            ax.fill_between(kk, np.maximum(e - 1.96 * sd / np.sqrt(14), 0), e + 1.96 * sd / np.sqrt(14), color="#0072B2", alpha=0.18, lw=0)
        if condition == "smoothed_band_cusum" and first[i] >= h and first[i] > 0:
            ax.axvline(first[i], color="#D55E00", lw=1.5, ls="--"); ax.text(first[i], ax.get_ylim()[1] * 0.92, " change", color="#D55E00", fontsize=8)
        ax.set_title(LABELS[i], loc="left", fontweight="bold"); ax.invert_xaxis()
        ax.set_ylim(bottom=0)
    for ax in axs[1]:
        ax.set_xlabel("days before expiry")
    fig.suptitle("Which TWO customers are most likely to leave? (data shown only up to the cutoff)", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def render_all(out_dir, trials: pd.DataFrame, X, L: int) -> List[str]:
    """One PNG per (set, condition): 24 sets x 4 conditions = 96 images."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    files = []
    for r in trials.itertuples():
        for cond in CONDITIONS:
            f = out / f"{r.trial_id}_{cond}.png"
            render_trial(f, X, list(r.panel_rows), cond, L, r.cutoff_days_before_expiry)
            files.append(str(f))
    return files


def answer_key(trials: pd.DataFrame) -> pd.DataFrame:
    return trials[["trial_id", "at_risk_positions", "event_ids"]].assign(
        at_risk_positions=lambda d: d["at_risk_positions"].map(lambda v: ",".join(sorted(v))),
        event_ids=lambda d: d["event_ids"].map(lambda v: ",".join(map(str, v))))


# ------------------------------------------------------------------------------------------ analysis
def _boot_ci(d: np.ndarray, B: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(B, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def analyze(df: pd.DataFrame, reference: str = "raw", B: int = 5000) -> Dict[str, pd.DataFrame]:
    """df columns: participant, condition, trial_id, n_correct (0-2), time_s, confidence (1-5)."""
    from scipy import stats
    d = df.copy()
    d["accuracy"] = d["n_correct"] / 2.0
    metrics = ["accuracy", "time_s", "confidence"]
    pp = d.groupby(["participant", "condition"])[metrics].mean().reset_index()
    summ = []
    for c, g in pp.groupby("condition"):
        row = {"condition": c, "n_participants": len(g)}
        for m in metrics:
            lo, hi = _boot_ci(g[m].to_numpy(), B)
            row.update({f"{m}_mean": g[m].mean(), f"{m}_lo": lo, f"{m}_hi": hi})
        summ.append(row)
    wide = {m: pp.pivot(index="participant", columns="condition", values=m).dropna() for m in metrics}
    contrasts, omni = [], []
    for m in metrics:
        w = wide[m]
        if len(w) >= 3 and w.shape[1] >= 3:
            st = stats.friedmanchisquare(*[w[c] for c in w.columns])
            omni.append({"metric": m, "friedman_chi2": float(st.statistic), "p": float(st.pvalue), "n": len(w)})
        for c in w.columns:
            if c == reference:
                continue
            diff = (w[c] - w[reference]).to_numpy()
            lo, hi = _boot_ci(diff, B)
            sd = diff.std(ddof=1) if len(diff) > 1 else np.nan
            try:
                p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            except ValueError:
                p = np.nan
            contrasts.append({"metric": m, "condition": c, "vs": reference, "mean_diff": float(diff.mean()), "ci_lo": lo, "ci_hi": hi,
                              "cohens_dz": float(diff.mean() / sd) if sd and sd > 0 else np.nan, "wilcoxon_p": p, "n": len(diff)})
    return {"participant_means": pp, "summary": pd.DataFrame(summ), "contrasts": pd.DataFrame(contrasts), "omnibus": pd.DataFrame(omni)}


def simulate_study(n_participants=12, trials_per_condition=6, seed=0, effects=None) -> pd.DataFrame:
    """Synthetic responses with planted effects, ONLY for testing the analysis code (not data)."""
    rng = np.random.default_rng(seed)
    effects = effects or {"raw": 0.0, "smoothed": 0.10, "smoothed_band": 0.14, "smoothed_band_cusum": 0.20}
    rows = []
    for p in range(n_participants):
        skill = rng.normal(0, 0.08)
        for c in CONDITIONS:
            for t in range(trials_per_condition):
                pr = np.clip(0.45 + effects[c] + skill, 0.05, 0.95)
                n_correct = int(rng.binomial(2, pr))
                rows.append({"participant": f"P{p + 1:02d}", "condition": c, "trial_id": f"{c}-{t}", "n_correct": n_correct,
                             "time_s": float(rng.gamma(9, 3.0) * (1.0 - 0.1 * effects[c])), "confidence": int(np.clip(round(2.5 + 2 * effects[c] + rng.normal(0, 1)), 1, 5))})
    return pd.DataFrame(rows)
