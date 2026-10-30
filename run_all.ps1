# Run the whole pipeline in Windows PowerShell (same steps as run_all.sh).
# NOT TESTED: written without access to PowerShell. It only calls the Python
# modules, which are tested, so a failure here is a scripting slip, not a result problem.
#
#   .\run_all.ps1                         full run
#   $env:SAMPLE = "0.1"; .\run_all.ps1    quick run on 10% of the expiry events
#   Remove-Item Env:SAMPLE                go back to the full run afterwards
#
# If PowerShell refuses to run scripts:  Set-ExecutionPolicy -Scope Process Bypass

$ErrorActionPreference = "Stop"

function Get-Setting([string]$Name, [string]$Default) {
    $v = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrEmpty($v)) { return $Default } else { return $v }
}

$RAW      = Get-Setting "RAW" "data/raw"
$PQ       = Get-Setting "PQ" "data/parquet"
$OUT      = Get-Setting "OUT" "outputs"
$ART      = Get-Setting "ART" "artifacts"
$PBI      = Get-Setting "PBI" "powerbi"
$MONTHS   = Get-Setting "MONTHS" "auto"                # "auto" or e.g. 2016-09:2017-02; the last month is the test month
$COLUMN_MAP = Get-Setting "COLUMN_MAP" ""             # path to a JSON column map (renamed columns)
$HORIZONS = (Get-Setting "HORIZONS" "30 21 14 7") -split " "
$SAMPLE   = Get-Setting "SAMPLE" "1.0"                 # fraction of events kept (dev runs)

function Run-Step([string[]]$StepArgs) {
    Write-Host ""
    Write-Host ">> python $($StepArgs -join ' ')" -ForegroundColor Cyan
    & python @StepArgs
    if ($LASTEXITCODE -ne 0) { throw "Step failed: python $($StepArgs -join ' ')" }
}

$prepArgs = @("-m", "src.prepare_data", "--raw", $RAW, "--out", $PQ)
if ($COLUMN_MAP -ne "") { $prepArgs += @("--map", $COLUMN_MAP) }
Run-Step $prepArgs
Run-Step (@("-m", "src.build_dataset", "--data-dir", $PQ, "--out", "$OUT/dataset.parquet",
            "--months", $MONTHS, "--horizons") + $HORIZONS + @("--sample-frac", $SAMPLE))
Run-Step @("-m", "src.run_experiments", "--dataset", "$OUT/dataset.parquet", "--out", $OUT, "--artifacts", $ART)
Run-Step @("-m", "src.explain", "--dataset", "$OUT/dataset.parquet", "--out", $OUT, "--artifacts", $ART)
Run-Step @("-m", "src.value_of_lead_time", "--predictions", "$OUT/predictions.parquet", "--out", $OUT)
Run-Step @("-m", "src.drift", "--dataset", "$OUT/dataset.parquet", "--out", $OUT)
Run-Step @("-m", "src.export_powerbi", "--dataset", "$OUT/dataset.parquet", "--outputs", $OUT,
           "--data-dir", $PQ, "--powerbi", $PBI)

Write-Host ""
Write-Host "Done. Open Power BI Desktop and load the CSV files in $PBI/data." -ForegroundColor Green
