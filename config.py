"""Central configuration: paths, sampling percentage, DuckDB resource limits, raw file names.

Everything that other modules need to agree on (which customers are sampled, where files live)
is defined here, so a re-run in a fresh Colab session reproduces exactly the same sample.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Logical table name -> candidate raw file names (plain or gzip-compressed CSV are both accepted).
RAW_FILES = {
    "train": ["train.csv", "train.csv.gz"],
    "train_v2": ["train_v2.csv", "train_v2.csv.gz"],  # optional
    "members": ["members_v3.csv", "members_v3.csv.gz"],
    "transactions": ["transactions.csv", "transactions.csv.gz"],
    "transactions_v2": ["transactions_v2.csv", "transactions_v2.csv.gz"],
    "user_logs": ["user_logs.csv", "user_logs.csv.gz"],
    "user_logs_v2": ["user_logs_v2.csv", "user_logs_v2.csv.gz"],
}
REQUIRED_TABLES = ("train", "members", "transactions", "user_logs")
OPTIONAL_TABLES = ("train_v2", "transactions_v2", "user_logs_v2")

# Recommended range from the project brief; values outside it only trigger a warning.
RECOMMENDED_PCT_RANGE = (5, 15)


@dataclass(frozen=True)
class Config:
    """Run configuration.

    raw_dir     directory holding the extracted Kaggle CSV files
    out_dir     base output directory (e.g. a Google Drive folder); the sampled Parquet goes in
                ``out_dir/sample{sample_pct}/``
    sample_pct  integer percentage of CUSTOMERS to keep (md5(msno) first 8 hex digits % 100 < pct)
    memory_limit / threads   DuckDB limits (Colab free tier: 12 GB RAM, 2 CPUs)
    temp_dir    DuckDB spill directory (use local disk, e.g. /content/duckdb_tmp, not Drive)
    """

    raw_dir: Path
    out_dir: Path
    sample_pct: int = 10
    memory_limit: str = "8GB"
    threads: int = 2
    temp_dir: Optional[Path] = None

    def validate(self) -> "Config":
        if not isinstance(self.sample_pct, int) or isinstance(self.sample_pct, bool):
            raise ValueError("sample_pct must be an integer percentage")
        if not 1 <= self.sample_pct <= 100:
            raise ValueError(f"sample_pct must be in [1, 100], got {self.sample_pct}")
        lo, hi = RECOMMENDED_PCT_RANGE
        if not lo <= self.sample_pct <= hi:
            warnings.warn(
                f"sample_pct={self.sample_pct} is outside the recommended {lo}-{hi}% range; "
                "check Colab memory/runtime before proceeding.",
                stacklevel=2,
            )
        return self

    @property
    def sample_dir(self) -> Path:
        return Path(self.out_dir) / f"sample{self.sample_pct}"
