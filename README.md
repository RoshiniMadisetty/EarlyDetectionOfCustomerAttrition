# silentchurn - visual analytics of silent churn (KKBox) for VDUI

Self-contained repo (no code reused from earlier projects).

## Modules (silentchurn/)
| file | what it does |
|---|---|
| config.py, ingest.py | step 1: CSV -> customer-sampled Parquet (resumable) + sampling report |
| synth.py | synthetic data in the real KKBox CSV format (tests only) |
| events.py | re-implementation of Kaggle's official labeller; event table (customer x expiry) |
| series.py | per-event daily engagement matrix (days before expiry) |
| smoothers.py | causal SMA / EMA / end-point LOWESS / end-point spline, CUSUM, leaky reference, causality test |
| benchmark.py | RQ1: detection lead time at matched false-positive rate, customer-level bootstrap |
| model.py | one simple logistic model, calibration, importance, PSI drift, subgroup slices, feature-leakage test |
| exports.py | Power BI tables, theme.json, DAX measures |
| figures.py | static PNG/HTML fallback of the five dashboard pages |
| usability.py | RQ2 stimuli, counterbalancing, analysis |
| pipeline.py | command-line driver: validate, events, series, benchmark, model, export, figures, usability |

## Run
    python -m pytest
    python -m silentchurn.pipeline all --sample-dir SAMPLE_DIR --work WORK_DIR

## Verified vs not
Tested: 86 tests on SYNTHETIC data (labelling rules on hand-made cases, causality/leakage tests, bootstrap, model,
exports, usability analysis, end-to-end pipeline). NOT tested: real-data runtime/memory of the analysis steps, whether our
re-derived labels reproduce train.csv on the real files (Cell 9 prints this), Power BI (DAX, theme, visuals).
