# silentchurn — visual analytics of silent churn (KKBox) for VDUI

New, self-contained repo (no code reused from earlier projects). Status: **step 1 (ingest) only**.

## Layout
```
silentchurn/config.py   paths, sample %, DuckDB limits, raw file names
silentchurn/synth.py    synthetic data in the real KKBox CSV format (tests / dry runs)
silentchurn/ingest.py   CSV -> customer-sampled Parquet (resumable) + sampling report
tests/                  pytest (synthetic data only)
```
More modules (labels, features with leakage test, smoothers, lead-time benchmark, exports, Power BI
package, fallbacks) are added step by step.

## Quick start (local)
```
pip install -r requirements.txt
python -m pytest
python -m silentchurn.synth --out /tmp/raw --n-members 3000
python -m silentchurn.ingest --raw /tmp/raw --out /tmp/out --pct 10
```

## Sampling rule
Customer kept iff `int(md5(msno)[:8], 16) % 100 < pct` (default 10, recommended 5-15). Same rule on every
table, applied during the CSV scan, so kept customers keep their full history and nothing else is written.
It uses md5 rather than DuckDB's `hash()` because DuckDB does not guarantee `hash()` is stable across
versions. State this (and the sample's churn rate vs the full data) in the paper's limitations.

## What is and is not verified
Tested: everything under `tests/` on synthetic data (21 tests). NOT tested: the real KKBox files, real
runtime/memory on Colab, Kaggle download/extraction steps, real date coverage.
