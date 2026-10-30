# Silent churn early warning

Multi-horizon, explainable, profit-aware churn early warning for subscription data
(KKBox file format), with a Power BI decision dashboard.

## What this code does

For every subscription expiry event it asks: **how well can we predict, `h` days before
expiry, that the member will not renew, and what is that warning worth?**

| Experiment | Script | Question |
| --- | --- | --- |
| E1 horizon sweep | `run_experiments.py` | How does accuracy decay as the warning moves earlier? |
| E2 ablation | `run_experiments.py` | Which feature families add signal, overall and per segment? |
| E3 explanation timeline | `explain.py` | Do the drivers change with the horizon, and are explanations stable? |
| E4 value of lead time | `value_of_lead_time.py` | What profit does a targeting policy earn at each horizon? |
| E5 calibration | `run_experiments.py` | Are the probabilities trustworthy? Conformal sets and intervals. |
| E6 drift | `drift.py` | Does a static model decay over months, and does rolling retraining help? |

## Install and run

```bash
pip install -r requirements.txt
pytest -q tests                      # 15 correctness, leakage and file-format tests (about 20 s)
# put the three raw files in data/raw, then:
bash run_all.sh                      # Windows PowerShell: .\run_all.ps1
```

### Loading your dataset (KKBox files)

1. Put the CSV files in `data/raw` (extract any .7z first). Names the code looks for:
   `members_v3.csv` (or `members.csv`), `transactions.csv` and/or `transactions_v2.csv`,
   `user_logs.csv` and/or `user_logs_v2.csv`. If both v1 and v2 exist, **both are used**:
   transactions are de-duplicated, and a later log file only adds days after the last day
   already loaded, so an overlapping file cannot double-count.
2. Check the files: `python -m src.check_schema --raw data/raw`. It lists columns, finds
   renamed ones, tests date parsing and says READY or what to map.
3. If columns are named differently, write a JSON map (or let step 2 draft one with
   `--write-map column_map.json`, then review it) and pass it with `--map column_map.json`
   (or `COLUMN_MAP=column_map.json` to `run_all`).
4. `--months auto` (default) picks the last six FULL expiry months whose +30 day window is
   observed. End of observation is the EARLIER of the last transaction and the last log row.
5. Optional label check against the official file:
   `python -m src.check_labels --labels data/raw/train.csv --month 2017-02`.
   `train.csv`/`train_data.csv` is otherwise unused: labels are re-derived from transactions.

Files that differ in more than names (extra or missing columns): missing optional columns
become blank features with a warning; missing required columns stop with a clear message.
Only `src/prepare_data.py` ever needs editing for a new source.

Quick trial without the real data: `python -m tests.make_synthetic --out data/raw_synth`
then `RAW=data/raw_synth PQ=data/parquet_synth OUT=outputs_synth ART=artifacts_synth PBI=powerbi_synth bash run_all.sh`.
Synthetic numbers mean nothing scientifically; they only prove the mechanics.

Large data: `user_logs` is tens of GB. All heavy work runs in DuckDB and streams to
Parquet, but modelling loads one horizon into pandas. For development use
`SAMPLE=0.1 bash run_all.sh`.

## Design decisions you should be able to defend

1. **Unit = subscription expiry event.** One row is one non-cancelled transaction whose
   expiry date `E` falls in the study months. A member can contribute several events.
2. **Cutoff `C = E - h`.** Features use only rows dated on or before `C` (inclusive).
3. **At-risk rule.** An event is scored at horizon `h` only if its own purchase was made
   by `C` and the member had not already extended membership beyond `E` by `C`. Both rules
   use only information available at `C`.
4. **Label.** `is_churn = 1` if no non-cancelled transaction with a later expiry is dated
   within `E + 30` days. Events whose `E + 30` is after the last observed transaction are
   dropped (censoring).
5. **Common cohort.** `in_common_cohort` marks events at risk at every horizon. Horizons
   are compared on that cohort by default, so differences are not caused by a changing
   population. Use `--cohort per_horizon` to see the alternative.
6. **Time-based split on expiry month.** Last month = test, the one before = validation,
   earlier months = train. The same member can appear in different months; a
   member-grouped split is a sensible extra robustness check.
7. **Leakage test.** `tests/test_pipeline.py` rewrites all data after each event's cutoff
   and asserts the features do not change. Injecting a 3-day leak into the code makes it fail.
8. **Segments** (from information at `C`): `explicit_cancel` (a cancel transaction since the
   plan was bought), else `auto_renew_on`, else `manual_renew`. "Silent" churners are those
   in the last two segments, who give no explicit cancel signal.
9. **Metrics.** PR-AUC and ROC-AUC use raw scores; log loss, Brier and ECE use calibrated
   probabilities (isotonic fitted on half of the validation month). Conformal quantile and
   interval bins use the other half.
10. **Value of Lead Time** is a policy score under assumed costs and an assumed save rate
    (`p_save`), including a "proportional" scenario where earlier warnings are more
    effective. These are assumptions varied in a grid, not estimates. It does not estimate
    the causal effect of an offer.

## Outputs

`outputs/` holds `dataset.parquet`, all result CSVs, `predictions.parquet` and
`figures/` (SHAP summaries and `signal_timeline.png`). `artifacts/h<h>/` holds the
XGBoost model, calibrator and metadata per horizon. `powerbi/data/` holds the CSV tables for
Power BI; see `powerbi/POWERBI_GUIDE.md` and `powerbi/measures.dax`.

## Known limitations (state these in the paper)

- One dataset and one domain (music streaming). No support-ticket data exists in this schema.
- SHAP values are on the log-odds scale of the class-weighted raw score, not calibrated
  probabilities. Rankings and relative sizes are meaningful; absolute sizes are not.
- XGBoost early stopping uses the validation month, which is then also used for
  calibration. The test month is untouched, but the validation month is used twice.
- Profit results depend on the assumed `p_save` and cost fractions. Always show the
  sensitivity grid.
- Interval bounds are Wilson intervals of the observed churn rate in the score's
  validation bin. They describe calibration uncertainty, not per-customer prediction intervals.
- Labels are rebuilt from the transaction files; public projects report 97% to 99% agreement
  with the official KKBox labels, so run `check_labels` and report your own agreement.
- Tested on Python 3.10 and 3.13 with synthetic data only.
- The DAX measures are untested. The pipeline and tests were run only on synthetic data
  in this workspace, not on the real files.

## Not included (optional extensions)

Change-point features (CUSUM), a sequence model (GRU/Transformer), Optuna tuning,
survival models (lifelines), and a member-grouped split.
