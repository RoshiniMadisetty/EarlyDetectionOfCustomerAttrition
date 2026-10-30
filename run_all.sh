#!/usr/bin/env bash
# Run the whole pipeline. Override any setting from the environment, e.g.
#   SAMPLE=0.1 bash run_all.sh                     # quick run on a 10% event sample
#   RAW=data/raw_synth PQ=data/parquet_synth OUT=outputs_synth bash run_all.sh
set -euo pipefail

RAW=${RAW:-data/raw}
PQ=${PQ:-data/parquet}
OUT=${OUT:-outputs}
ART=${ART:-artifacts}
PBI=${PBI:-powerbi}
MONTHS=${MONTHS:-auto}                 # "auto" or e.g. 2016-09:2017-02; the last month is the test month
HORIZONS=${HORIZONS:-"30 21 14 7"}     # days before expiry
SAMPLE=${SAMPLE:-1.0}                  # fraction of expiry events kept (dev runs)

MAP_ARGS=""
[ -n "${COLUMN_MAP:-}" ] && MAP_ARGS="--map $COLUMN_MAP"   # JSON column map for renamed columns
python -m src.prepare_data   --raw "$RAW" --out "$PQ" $MAP_ARGS
python -m src.build_dataset  --data-dir "$PQ" --out "$OUT/dataset.parquet" \
                             --months "$MONTHS" --horizons $HORIZONS --sample-frac "$SAMPLE"
python -m src.run_experiments --dataset "$OUT/dataset.parquet" --out "$OUT" --artifacts "$ART"
python -m src.explain         --dataset "$OUT/dataset.parquet" --out "$OUT" --artifacts "$ART"
python -m src.value_of_lead_time --predictions "$OUT/predictions.parquet" --out "$OUT"
python -m src.drift           --dataset "$OUT/dataset.parquet" --out "$OUT"
python -m src.export_powerbi  --dataset "$OUT/dataset.parquet" --outputs "$OUT" \
                              --data-dir "$PQ" --powerbi "$PBI"
echo "Done. Open Power BI Desktop and load the CSV files in $PBI/data (see powerbi/POWERBI_GUIDE.md)."
