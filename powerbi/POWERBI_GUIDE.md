# Power BI guide

Power BI Desktop runs on Windows only. The report itself (.pbix) has to be built in
Power BI Desktop; this folder gives you every data table and the measures.
`measures.dax` was written without access to Power BI and is untested.

## 1. Load the data

Run `.\run_all.ps1` (Windows) or `bash run_all.sh`, then in Power BI Desktop: **Get data > Text/CSV**, and load every
file in `powerbi/data/`. Keep the default table names (the file name without `.csv`).

## 2. Relationships (Model view)

| From (one side) | To (many side) | Note |
| --- | --- | --- |
| `horizons[horizon]` | `predictions[horizon]` | |
| `horizons[horizon]` | `value_of_lead_time[horizon]` | |
| `horizons[horizon]` | `shap_importance[horizon]`, `shap_family_importance[horizon]` | |
| `horizons[horizon]` | `metrics_by_horizon[horizon]`, `reliability[horizon]`, `conformal_summary[horizon]` | |
| `predictions[row_key]` | `shap_values_long[row_key]` | one customer, many features |
| `predictions[row_key]` | `shap_base[row_key]` | one to one |
| `predictions[row_key]` | `customer_features[row_key]` | one to one |
| `feature_catalog[feature]` | `shap_importance[feature]` | for descriptions and families |

Set cross-filter direction to single (one to many). Mark `horizons[label]` as
"Sort by column: sort_order".

## 3. Pages

**Page 1: Churn landscape (Unit 1: EDA, multivariate, perception)**
- Line chart from `eda_daily_usage`: axis `day_offset`, values `mean_secs`, legend `outcome`,
  small multiples by `segment`. Shows how churned and retained users diverge before the cutoff.
- Matrix heatmap from `correlation_matrix` (rows `feature_x`, columns `feature_y`,
  values `spearman`, diverging colour scale centred on 0), filtered to one family at a time.
- Scatter from `eda_sample`: e.g. `usage_slope_30` vs `days_since_last_active`, colour by `is_churn`.
- Cards from `cohort_report`: events at risk at each horizon.

**Page 2: Early-warning view**
- Slicer: `horizons[label]`.
- Cards: Customers Scored, Churn Rate, Flagged (High + Critical), Precision, Recall, Lift.
- Bar chart: customers by `risk_band`, with actual churn rate as a line.
- Line chart of `metrics_by_horizon[pr_auc]` by horizon for each `model`.
- Clustered bar from `ablation`: `pr_auc` by `families_up_to`, filtered by `segment`.

**Page 3: Why this customer (drill-through target)**
- Drill-through field: `predictions[row_key]`.
- Card: Customer Risk Text.
- Waterfall chart: `shap_values_long[feature]` on category, SHAP Contribution on Y,
  Top N filter 10 by Abs SHAP. Note: SHAP values are log-odds of the raw model score.
- Table from `customer_features` for the same customer.

**Page 4: Value and monitoring (Unit 4: probabilities, XAI, drift)**
- Slicers (single select): `k_pct`, `p_save`, `offer_cost_frac`, `lead_effect`.
- Line chart: `horizon` vs Profit (selected scenario) and Profit vs Random Contact.
- Reliability chart from `reliability`: `mean_predicted` vs `observed_rate`, plus a diagonal.
- Stacked 100% column from `shap_family_importance`: share by `family` per horizon.
- Heatmap from `drift_psi`: rows `feature`, columns `month`, values `psi`
  (conditional format at 0.10 and 0.25).
- Line chart from `drift_metrics`: `pr_auc` by `month`, legend `strategy`.

## 4. Design rules to follow (Unit 1)

- One risk colour scale on every page (for example light yellow to dark red).
- A colour-blind-safe categorical palette for model and segment colours.
- Zero-based axes for bars, labelled axes, no 3D, no decorative gradients.
- Direct data labels instead of legends where there are two or three series.
- Say what each chart does not show (see the limitations in the main README).

## 5. Full step-by-step

The click-by-click version, with the run commands and a troubleshooting table, is in the
"Run Guide and Power BI Build Steps" document that came with this project.
