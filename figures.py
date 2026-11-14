"""Static fallbacks for the five Power BI pages (PNG) + one self-contained HTML (images embedded as base64).

Page 1 Foundations & honest charts   (Unit 1)    Page 2 Smoothing lab            (Unit 1 / temporal visualisation)
Page 3 Detection lead time           (RQ1, uncertainty)   Page 4 Model output & uncertainty (Unit 4)
Page 5 Drift & monitoring            (Unit 4)
Same colour-blind-safe palette as theme.json. Every page states what it does NOT show.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import smoothers as sm
from .benchmark import baseline, best_per_family, MIN_BASELINE
from .exports import CATEGORICAL, REPRESENTATIVE

FAM_COL = {"raw": "#999999", "SMA": "#0072B2", "EMA": "#D55E00", "LOWESS": "#009E73", "Spline": "#CC79A7", "SMA-centered": "#000000"}
CH_COL, RT_COL = "#D55E00", "#0072B2"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "figure.dpi": 110})


def _foot(fig, text):
    fig.text(0.01, 0.005, "NOT shown: " + text, fontsize=8, color="#4B5563", ha="left", va="bottom", wrap=True)


def _examples(ev, X, has_log, L, n=3, seed=3):
    rng = np.random.default_rng(seed)
    ok = np.asarray(has_log, bool) & (baseline(X, L) >= MIN_BASELINE)
    pick = lambda m: rng.choice(np.where(m & ok)[0], size=n, replace=False)
    return pick((ev["silent"] == 1).to_numpy()), pick((ev["is_churn"] == 0).to_numpy())


def page1(path, ev, X, has_log, L, sampling):
    ok = np.asarray(has_log, bool) & (baseline(X, L) >= MIN_BASELINE)
    ch, rt = ok & (ev["silent"] == 1).to_numpy(), ok & (ev["is_churn"] == 0).to_numpy()
    k = np.arange(L, 0, -1)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.38, wspace=0.3)
    ax = fig.add_subplot(gs[0, :2])
    for m, col, lab in ((rt, RT_COL, f"Retained (n={rt.sum():,})"), (ch, CH_COL, f"Silent churn (n={ch.sum():,})")):
        mu = X[m].mean(0); se = X[m].std(0, ddof=1) / np.sqrt(m.sum())
        ax.plot(k, mu, color=col, lw=1.8, label=lab)
        ax.fill_between(k, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=0.2, lw=0)
    ax.invert_xaxis(); ax.set_xlabel("days before expiry"); ax.set_ylabel("mean log(1+minutes) per day")
    ax.set_ylim(0, None); ax.set_title("Cohort means with 95% CI, zero-based axis (honest)"); ax.legend(frameon=False, loc="lower left")
    ax = fig.add_subplot(gs[0, 2:])
    m_r, m_c = float(X[rt][:, -7:].mean()), float(X[ch][:, -7:].mean())
    gap = abs(m_r - m_c)
    lo = min(m_r, m_c) - 0.25 * gap
    ax.bar(["Retained", "Silent churn"], [m_r, m_c], color=[RT_COL, CH_COL])
    ax.set_ylim(lo, max(m_r, m_c) + 0.1 * gap)
    ax.set_title(f"Last-7-day mean, y-axis starts at {lo:.2f}: bars look {((max(m_r, m_c) - lo) / (min(m_r, m_c) - lo)):.0f}x apart\n(true ratio {max(m_r, m_c) / max(min(m_r, m_c), 1e-9):.2f}x)  - MISLEADING", color="#9B2C2C")
    ax.set_ylabel("mean log(1+minutes), truncated axis")
    chi, rti = _examples(ev, X, has_log, L, n=2)
    M = np.expm1(X.astype(np.float64))
    for i, (idx, col, lab) in enumerate([(chi[0], CH_COL, "Silent churner"), (chi[1], CH_COL, "Silent churner"),
                                         (rti[0], RT_COL, "Retained"), (rti[1], RT_COL, "Retained")]):
        ax = fig.add_subplot(gs[1, i])
        ax.bar(k, M[idx], color="#BDBDBD", width=0.9, label="daily minutes")
        ax.plot(k, sm.fir_apply(M[idx][None, :], sm.sma_taps(7))[0], color=col, lw=2, label="7-day moving avg")
        ax.invert_xaxis(); ax.set_title(lab); ax.set_xlabel("days before expiry")
        if i == 0:
            ax.legend(frameon=False, fontsize=8); ax.set_ylabel("minutes")
    s = ""
    if sampling is not None and len(sampling):
        r = sampling.iloc[0]
        s = f"  Sample churn {100*r.sample_churn_rate:.2f}% vs full {100*r.full_churn_rate:.2f}% ({r.file})." if pd.notna(r.get("full_churn_rate")) else ""
    fig.suptitle("Page 1 - Foundations: raw signal, cohort curves, and an honest vs misleading axis (Unit 1)." + s, x=0.01, ha="left", fontweight="bold")
    _foot(fig, "causality or lead time; individual panels are randomly chosen examples, not typical or best cases; CI assumes independent customers.")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def page2(path, ev, X, has_log, L):
    chi, _ = _examples(ev, X, has_log, L, n=1, seed=5)
    M = np.expm1(X.astype(np.float64)).astype(np.float32)
    i = chi[0]
    k = np.arange(L, 0, -1)
    fig, axs = plt.subplots(2, 2, figsize=(15, 8.5), gridspec_kw={"hspace": 0.4, "wspace": 0.22})
    ax = axs[0, 0]
    ax.bar(k, M[i], color="#D0D0D0", width=0.9, label="raw minutes")
    for name, fam in (("SMA w=7", "SMA"), ("EMA a=0.15", "EMA"), ("LOWESS w=14", "LOWESS"), ("Spline w=28 lam=10", "Spline")):
        s = [x for x in sm.make_smoothers(False) if x.name == name][0]
        ax.plot(k, s.fn(M[i][None, :])[0], color=FAM_COL[fam], lw=2, label=name)
    ax.set_xlim(90, 0); ax.set_title("One silent churner: four causal smoothers (end-point estimates)"); ax.set_xlabel("days before expiry"); ax.legend(frameon=False, fontsize=8)
    ax = axs[0, 1]
    lag = np.arange(28)
    ax.plot(lag, sm.sma_taps(7).tolist() + [0] * 21, color=FAM_COL["SMA"], lw=2, label="SMA w=7")
    a = 0.15; ax.plot(lag, a * (1 - a) ** lag, color=FAM_COL["EMA"], lw=2, label="EMA a=0.15")
    ax.plot(lag, np.pad(sm.lowess_taps(14), (0, 14)), color=FAM_COL["LOWESS"], lw=2, label="LOWESS w=14 (one-sided)")
    ax.plot(lag, sm.spline_taps(28, 10), color=FAM_COL["Spline"], lw=2, label="Spline w=28 lam=10")
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("how many days ago"); ax.set_ylabel("weight on that day")
    ax.set_title("How each method weighs the past (spline/LOWESS weights can be negative)"); ax.legend(frameon=False, fontsize=8)
    ax = axs[1, 0]
    step = np.r_[np.full(60, 10.0), np.full(60, 2.0)][None, :].astype(np.float32)
    x = np.arange(120)
    ax.plot(x, step[0], color="#999999", lw=2, label="engagement falls from 10 to 2 at day 60")
    for name, fam in (("SMA w=7", "SMA"), ("EMA a=0.15", "EMA"), ("LOWESS w=14", "LOWESS"), ("Spline w=28 lam=10", "Spline")):
        s = [q for q in sm.make_smoothers(False) if q.name == name][0]
        ax.plot(x, s.fn(step)[0], color=FAM_COL[fam], lw=2, label=name)
    ax.set_xlim(45, 90); ax.set_title("Step response: speed vs smoothness (and overshoot)"); ax.set_xlabel("day"); ax.legend(frameon=False, fontsize=8)
    ax = axs[1, 1]
    ax.plot(x, step[0], color="#999999", lw=2, label="engagement")
    ax.plot(x, sm.fir_apply(step, sm.sma_taps(7))[0], color=FAM_COL["SMA"], lw=2, label="SMA w=7 (causal, valid)")
    ax.plot(x, sm.sma_centered(step, 7)[0], color="black", lw=2, ls="--", label="centred SMA (LEAKY: uses future days)")
    ax.set_xlim(45, 90); ax.set_title("Why a centred smoother is invalid for early warning: it reacts BEFORE the drop"); ax.legend(frameon=False, fontsize=8); ax.set_xlabel("day")
    fig.suptitle("Page 2 - Smoothing lab: moving average vs EMA vs LOWESS vs spline (Unit 1 calculation, temporal visualisation)", x=0.01, ha="left", fontweight="bold")
    _foot(fig, "which method is best at detecting churn (page 3); the weights shown are for fixed illustrative settings.")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def page3(path, bench):
    r = bench[(bench.alpha == 0.10) & (bench.group == "all") & (bench.metric == "tpr_L14")].sort_values("estimate")
    fig, axs = plt.subplots(1, 3, figsize=(16, 7), gridspec_kw={"wspace": 0.38, "width_ratios": [1.3, 1, 1]})
    ax = axs[0]
    y = np.arange(len(r))
    cols = [FAM_COL.get(f, "#999") for f in r["family"]]
    ax.barh(y, r["estimate"], color=cols, hatch=None)
    for yi, (m, e, lo, hi, causal) in enumerate(zip(r["method"], r["estimate"], r["lo"], r["hi"], r["causal"])):
        ax.errorbar(e, yi, xerr=[[max(e - lo, 0)], [max(hi - e, 0)]], color="k", lw=1, capsize=2)
        if not causal:
            ax.barh(yi, e, color="white", edgecolor="black", hatch="///", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(r["method"], fontsize=8); ax.set_xlabel("share of churners alarmed >= 14 days before expiry")
    ax.set_title("Lead-time TPR at FPR 10% (95% bootstrap CI)\nhatched = LEAKY reference, not valid")
    best = best_per_family(bench, "all", 0.10, "tpr_L14")
    ax = axs[1]
    for fam, name in zip(best["family"], best["method"]):
        d = bench[(bench.method == name) & (bench.alpha == 0.10) & (bench.group == "all") & bench.metric.str.match(r"tpr_L\d+")].copy()
        d["L"] = d["metric"].str.extract(r"(\d+)").astype(int); d = d.sort_values("L")
        ax.plot(d["L"], d["estimate"], marker="o", color=FAM_COL[fam], label=name); ax.fill_between(d["L"], d["lo"], d["hi"], color=FAM_COL[fam], alpha=0.15, lw=0)
    ax.set_xlabel("required lead time (days before expiry)"); ax.set_ylabel("share of churners detected"); ax.set_title("Best setting per family:\nearlier warning costs detection"); ax.legend(frameon=False, fontsize=8)
    ax = axs[2]
    for gi, (g, off) in enumerate((("auto_renew", -0.12), ("manual_renew", 0.12))):
        for fi, (fam, name) in enumerate(zip(best["family"], best["method"])):
            d = bench[(bench.method == name) & (bench.alpha == 0.10) & (bench.group == g) & (bench.metric == "tpr_L14")]
            if len(d):
                d = d.iloc[0]
                ax.errorbar(d["estimate"], fi + off, xerr=[[max(d["estimate"] - d["lo"], 0)], [max(d["hi"] - d["estimate"], 0)]], fmt="o" if g == "auto_renew" else "s",
                            color=FAM_COL[fam], mfc="white" if g == "manual_renew" else FAM_COL[fam], capsize=2)
    ax.set_yticks(range(len(best))); ax.set_yticklabels(best["family"]); ax.set_xlabel("TPR at >=14 days, FPR 10%")
    ax.set_title("Auto-renew (filled) vs manual (open)")
    fig.suptitle("Page 3 - RQ1: which smoother separates churners earliest at a matched false-positive rate?", x=0.01, ha="left", fontweight="bold")
    _foot(fig, "causal effects of any intervention; thresholds are calibrated on retained customers of each group, so FPR is matched but the groups differ in size.")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def page4(path, mo):
    fig, axs = plt.subplots(2, 2, figsize=(15, 9), gridspec_kw={"hspace": 0.4, "wspace": 0.25})
    rel = mo["reliability_monitor"]
    ax = axs[0, 0]
    ax.plot([0, rel["mean_pred"].max() * 1.1] * 2, [0, rel["mean_pred"].max() * 1.1] * 2, alpha=0)
    lim = max(rel["mean_pred"].max(), rel["observed"].max(), rel["obs_hi"].max()) * 1.1
    ax.plot([0, lim], [0, lim], color="#999999", ls="--", label="perfect calibration")
    ax.errorbar(rel["mean_pred"], rel["observed"], yerr=[rel["observed"] - rel["obs_lo"], rel["obs_hi"] - rel["observed"]], fmt="o-", color=RT_COL, capsize=3, label="monitoring set (out of time)")
    ax.set_xlabel("mean predicted churn probability (decile)"); ax.set_ylabel("observed churn rate (95% Wilson CI)")
    m = mo["metrics"]; ax.set_title(f"Reliability: AUC {m['auc_monitor']:.3f}, ECE {m['ece_monitor']:.3f}, Brier {m['brier_monitor']:.3f}"); ax.legend(frameon=False)
    ax = axs[0, 1]
    d = mo["split_frame"]; d = d[d["split"] == "monitor"]
    bins = np.linspace(0, d["score"].quantile(0.995), 31)
    ax.hist(d.loc[d.y == 0, "score"], bins=bins, color=RT_COL, alpha=0.6, density=True, label="retained")
    ax.hist(d.loc[d.y == 1, "score"], bins=bins, color=CH_COL, alpha=0.6, density=True, label="churned")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("density"); ax.set_title("Score distribution by actual outcome"); ax.legend(frameon=False)
    ax = axs[1, 0]
    imp = mo["importance"].iloc[::-1]
    ax.barh(imp["feature"], imp["auc_drop"], xerr=imp["auc_drop_sd"], color="#56B4E9")
    ax.set_xlabel("drop in AUC when the feature is shuffled"); ax.set_title("What drives the score? (permutation importance, XAI)")
    ax = axs[1, 1]
    if "slices" in mo:
        s = mo["slices"]; s = s[s["reliable"]].sort_values("calibration_gap")
        lab = s["slice_col"] + ": " + s["level"]
        ax.barh(lab, s["calibration_gap"], color=np.where(s["calibration_gap"] > 0, CH_COL, RT_COL))
        ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("mean score - observed churn rate (>0 = over-predicts)"); ax.tick_params(axis="y", labelsize=7)
        ax.set_title("Bias check: calibration gap by subgroup (reliable slices only)")
    else:
        ax.axis("off")
    fig.suptitle("Page 4 - Model output and uncertainty (Unit 4): probabilities, calibration, importance, subgroup bias", x=0.01, ha="left", fontweight="bold")
    _foot(fig, "causes of churn (importance is predictive, not causal); small subgroups are dropped; the model is deliberately simple and is not the contribution.")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def page5(path, ev, X, L, mo):
    dr, pw = mo["drift"], mo["perf_week"]
    fig, axs = plt.subplots(2, 2, figsize=(15, 9), gridspec_kw={"hspace": 0.4, "wspace": 0.25})
    ax = axs[0, 0]
    for f, col in zip(sorted(dr["feature"].unique()), CATEGORICAL * 2):
        d = dr[dr.feature == f]
        ax.plot(d["week"], d["psi"], marker="o", ms=3, lw=2 if f == "score" else 1, color="k" if f == "score" else col, label=f)
    ax.axhline(0.10, color="#E69F00", ls="--"); ax.axhline(0.25, color="#D55E00", ls="--")
    ax.text(ax.get_xlim()[0], 0.105, "0.10 watch", color="#B7791F", fontsize=8); ax.text(ax.get_xlim()[0], 0.255, "0.25 alert", color="#9B2C2C", fontsize=8)
    ax.set_xlabel("weeks since 2017-02-01"); ax.set_ylabel("PSI vs training window"); ax.set_title("Input and score drift (PSI; thresholds are rules of thumb)"); ax.legend(frameon=False, fontsize=7, ncol=2)
    ax = axs[0, 1]
    ax.plot(pw["week"], pw["auc"], marker="o", color=RT_COL); ax.set_ylabel("AUC (weeks with too few churners omitted)", color=RT_COL)
    ax2 = ax.twinx(); ax2.bar(pw["week"], pw["n"], alpha=0.15, color="#999999"); ax2.set_ylabel("events in week", color="#666"); ax2.spines["right"].set_visible(True)
    ax.set_title("Discrimination over time"); ax.set_xlabel("weeks since 2017-02-01")
    ax = axs[1, 0]
    ax.plot(pw["week"], pw["base_rate"], marker="o", color=CH_COL, label="observed churn rate"); ax.plot(pw["week"], pw["mean_score"], marker="s", color=RT_COL, label="mean predicted")
    ax.set_xlabel("weeks since 2017-02-01"); ax.set_ylabel("rate"); ax.set_title("Calibration-in-the-large over time (gap = model under/over-predicts)"); ax.legend(frameon=False)
    ax = axs[1, 1]
    ok = (baseline(X, L) >= MIN_BASELINE)
    _, first = sm.cusum_down(X, L)
    wk = ((pd.to_datetime(ev["expiry_date"]) - pd.Timestamp("2017-02-01")).dt.days // 7).clip(lower=0).to_numpy()
    rows = [(w, (first[(wk == w) & ok] > 0).mean()) for w in sorted(set(wk[ok]))]
    ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", color="#009E73")
    ax.set_xlabel("weeks since 2017-02-01 (expiry week)"); ax.set_ylabel("share of customers with a CUSUM alarm"); ax.set_title("Anomaly monitoring: CUSUM alarm rate by expiry week")
    fig.suptitle("Page 5 - Drift and monitoring (Unit 4): is the model and its input still behaving as when it was trained?", x=0.01, ha="left", fontweight="bold")
    _foot(fig, "why drift happens; only two months of data, so weekly points are few and noisy; PSI thresholds are conventions, not significance tests.")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def make_all(out_dir, ev, X, has_log, L, bench, mo, sampling=None) -> List[str]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ev = ev.reset_index(drop=True)
    files = [out / f"page{i}.png" for i in range(1, 6)]
    page1(files[0], ev, X, has_log, L, sampling); page2(files[1], ev, X, has_log, L); page3(files[2], bench)
    page4(files[3], mo); page5(files[4], ev, X, L, mo)
    titles = ["1. Foundations (Unit 1)", "2. Smoothing lab (Unit 1 / temporal)", "3. Detection lead time (RQ1)", "4. Model output and uncertainty (Unit 4)", "5. Drift and monitoring (Unit 4)"]
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>Silent churn - static fallback</title>"
            "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#fff;color:#1f2937}img{max-width:100%;border:1px solid #e5e7eb}"
            "h2{margin-top:36px}@media(prefers-color-scheme:dark){body{background:#111827;color:#e5e7eb}img{background:#fff}}</style></head><body>"
            "<h1>Silent churn visual analytics - static fallback of the Power BI pages</h1>"]
    for t, f in zip(titles, files):
        html.append(f"<h2>{t}</h2><img alt='{t}' src='data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}'>")
    html.append("</body></html>")
    (out / "dashboard_fallback.html").write_text("\n".join(html), encoding="utf-8")
    return [str(f) for f in files] + [str(out / "dashboard_fallback.html")]
