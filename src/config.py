"""Shared constants: paths, horizons, feature families.

Every feature is computed "as of end of day C", where C = expiry date E minus
the horizon h (days). Only rows dated <= C are used. See build_dataset.py.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PARQUET = ROOT / "data" / "parquet"
OUTPUTS = ROOT / "outputs"
ARTIFACTS = ROOT / "artifacts"
POWERBI = ROOT / "powerbi"

# Warning horizons: days before membership expiry at which risk is scored.
DEFAULT_HORIZONS = [30, 21, 14, 7]

# Churn label: no non-cancelled renewal with a later expiry within this many days of E.
CHURN_GRACE_DAYS = 30

# Feature windows (days, ending at C inclusive).
RECENT_WINDOW = 30      # main behavioural window
BASELINE_WINDOW = 90    # personal baseline window immediately before the main window

# Feature families (order matters for the cumulative ablation experiment).
FAMILIES = {
    "contract": [
        "anchor_payment_method_id", "anchor_plan_days", "anchor_list_price",
        "anchor_paid", "anchor_auto_renew", "anchor_discount",
        "anchor_discount_ratio", "n_cancel_since_anchor", "n_cancel_30d",
        "n_cancel_before", "n_plan_changes", "n_pay_method_changes",
        "n_distinct_plan_days", "n_distinct_pay_methods", "manual_tx_share",
        "days_since_anchor_tx", "plan_progress",
    ],
    "tenure_profile": [
        "tenure_days", "n_renewals", "city", "registered_via", "gender_code", "bd",
    ],
    "usage_level": [
        "secs_7", "secs_14", "secs_30", "unq_7", "unq_14", "unq_30",
        "plays_30", "completion_ratio_30", "skip_ratio_30",
        "avg_secs_per_active_day_30",
    ],
    "engagement": [
        "active_days_7", "active_days_14", "active_days_30",
        "days_since_last_active", "max_inactivity_gap_30",
    ],
    "trajectory": [
        "usage_ratio_7_vs_prev", "usage_ratio_14_vs_prev",
        "usage_slope_30", "usage_vs_baseline", "active_days_ratio_vs_baseline",
    ],
}
FAMILY_ORDER = ["contract", "tenure_profile", "usage_level", "engagement", "trajectory"]
FEATURES = [f for fam in FAMILY_ORDER for f in FAMILIES[fam]]

# Integer-coded identifiers: one-hot encoded for logistic regression, raw codes for trees.
CATEGORICAL = ["anchor_payment_method_id", "city", "registered_via"]

FEATURE_FAMILY = {f: fam for fam, fs in FAMILIES.items() for f in fs}

SEGMENTS = ["auto_renew_on", "manual_renew", "explicit_cancel"]

ID_COLS = ["event_id", "msno", "horizon", "expire_month", "expire_date", "cutoff_date", "segment"]
