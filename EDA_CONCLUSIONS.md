# LendingClub — EDA Findings & Feature Decision Log

**Project:** Loan Default Risk Prediction  
**EDA source:** `data/interim/lendingclub_2013_2016_36m_raw.parquet` (created from `archive.zip` → full raw Parquet)  
**Scope:** 2013–2016 issued loans, 36-month term, `Fully Paid` / `Charged Off` outcomes  
**Purpose:** Record **what we observed**, **what we decided**, and **what still needs testing**. This is a living report, not the preprocessing script.

## Data and evaluation context

- Full ZIP-derived source Parquet: **2,925,493 rows**, **142 columns** (original values stored as strings).
- Selected cohort: **869,540 loans**, **146 columns** (including `issue_date`, `term_months`, `issue_year`, `target`).
- Outcomes: **741,370 Fully Paid (target=0)** and **128,170 Charged Off (target=1)**; observed default fraction **14.74%** among the selected resolved loans.
- The older `lendingclub_2013_2016_36m.parquet` is a separate **142-column typed file**, not the source of this EDA. In the older file, many unavailable observations appeared as `-1`; the raw-cohort file retains them mainly as missing.
- All cohort observations may be used for **predictor-only descriptive** EDA (distribution, completeness, chronology, predictor correlations). **Target-dependent feature choices must not inspect 2016 outcomes**: 2016 is the intended chronological test period. The exact split will be implemented in Notebook 02.
- **Caveat:** Selecting only `Fully Paid` and `Charged Off` loans introduces outcome-selection bias; findings do not directly generalize to all loan applications or unresolved loans.

## 01.1 — Initial column and missingness audit

**Finding:** All 146 columns are represented in the inventory. Exactly **13 columns have 100% missingness** in this cohort. Many other columns have very high missingness for different reasons. The ZIP-derived cohort has few literal `-1` values: missingness should be measured as null/blank first, not blindly represented with `-1`.

**Decision — `SKIP_ALL_MISSING` (13):** Do not do distributions, outlier tests, or predictor correlations for these fields in the main cohort. Keep the documented exclusion names; do not physically remove source columns yet.

- `next_pymnt_d`
- `revol_bal_joint`
- `sec_app_earliest_cr_line`
- `sec_app_fico_range_low`, `sec_app_fico_range_high`
- `sec_app_inq_last_6mths`
- `sec_app_mort_acc`, `sec_app_open_acc`, `sec_app_open_act_il`
- `sec_app_revol_util`, `sec_app_num_rev_accts`
- `sec_app_chargeoff_within_12_mths`, `sec_app_collections_12_mths_ex_med`

**Note:** The secondary-applicant fields above have no observed values in this selected cohort. This is different from saying secondary-applicant data is never useful in other time periods or datasets.

## 01.2 — Column types and numerical validity

**Finding:** Initial semantic/value classification covered all 146 columns: **98 numerical**, **20 categorical**, **10 date**, **3 identifier**, **2 text**, **2 target**, and **11 `REVIEW_TYPE`** (the latter are entirely empty secondary-applicant fields). These are **data-type groups**, not approved predictors. `issue_year` and `term_months` were initially counted as numerical, but are **cohort metadata** rather than ordinary predictors.

- The original source stores most fields as strings by design, so data type is determined by field meaning (data dictionary) and validated against actual values; storage type alone does not determine semantics.
- `revol_util` is a percent-formatted string (e.g., `52.3%`). The first numeric validity test erroneously flagged its valid observations because direct float casting failed. Numeric checks must strip `%` **for validation only**. The raw file is unchanged.
- **Open check:** Ensure all remaining numerical candidates parse correctly; investigate any unexpected strings rather than converting or discarding them silently.

## 01.3 — Prediction-time eligibility screening

**Observed screening results (all 146 accounted for):**

| Initial category | Count | Action at this stage |
|---|---:|---|
| `EDA_CANDIDATE` | **80** | Investigate further: 70 numerical, 7 categorical, 1 date, 2 text |
| `EXCLUDE_LEAKAGE` | **31** | Exclude from predictor EDA: loan performance, later account updates, hardship/settlement/servicing |
| `SKIP_ALL_MISSING` | **13** | Exclude from detailed cohort EDA (listed in 01.1) |
| `REVIEW_LENDER_INFO` | **7** | Hold for separate prediction-time or lender-benchmark question |
| `METADATA` | **6** | Keep for cohort definition, dates and splits; not ordinary predictors |
| `SKIP_HIGH_MISSING` | **4** | >80% missing; set aside from primary detailed EDA, with reasons below |
| `EXCLUDE_IDENTIFIER` | **3** | Not predictors |
| `TARGET` | **2** | Outcome labels only |

**`EXCLUDE_LEAKAGE` (31), exactly as flagged by current screening:**

`collection_recovery_fee`, `debt_settlement_flag`, `deferral_term`, `hardship_amount`, `hardship_dpd`, `hardship_end_date`, `hardship_flag`, `hardship_last_payment_amount`, `hardship_length`, `hardship_loan_status`, `hardship_payoff_balance_amount`, `hardship_reason`, `hardship_start_date`, `hardship_status`, `hardship_type`, `last_credit_pull_d`, `last_fico_range_high`, `last_fico_range_low`, `last_pymnt_amnt`, `last_pymnt_d`, `orig_projected_additional_accrued_interest`, `out_prncp`, `out_prncp_inv`, `payment_plan_start_date`, `pymnt_plan`, `recoveries`, `total_pymnt`, `total_pymnt_inv`, `total_rec_int`, `total_rec_late_fee`, `total_rec_prncp`.

**`SKIP_HIGH_MISSING` (4), set aside, not mathematically proven useless:**

| Feature | Missingness | Reason |
|---|---:|---|
| `annual_inc_joint` | 99.37% | Joint-applicant-only information; unsuitable for general primary cohort |
| `dti_joint` | 99.37% | Same applicability concern |
| `verification_status_joint` | 99.37% | Same applicability concern |
| `mths_since_last_record` | 81.71% | Event-recency feature; potential no-public-record meaning. Set aside under the agreed >80% screening policy; possible reconsideration |

**`REVIEW_LENDER_INFO` (7):** `funded_amnt`, `funded_amnt_inv`, `grade`, `initial_list_status`, `installment`, `int_rate`, `sub_grade`. Their eligibility depends on when a prediction is meant to be made. They are not included in the independent borrower-information baseline without justification.

**`METADATA` (6):** `issue_d`, `issue_date`, `issue_year`, `term`, `term_months`, `policy_code`. `term` and `term_months` are fixed by cohort definition. Verify `policy_code` is constant in the new source before final exclusion.

**`EXCLUDE_IDENTIFIER` (3):** `Unnamed: 0`, `id`, `url`.  
**`TARGET` (2):** `loan_status`, `target`.

**Caution:** This is an *initial* eligibility screen. It does not establish that every remaining field is valid at the intended prediction time. For example, `verification_status` may depend on whether the prediction is before or after lender verification. We will return to such questions before final feature approval.

## 01.4 — Year-wise availability (development-year suitability)

**Analysis:** Complete 2013–2016 cohort; evaluate annual availability, especially in the **2013–2015 development years**, instead of relying on overall coverage.

**Strong finding:** **14 numeric fields** have **0% coverage in 2013**, **0% in 2014**, approximately **4.54–5.27% in 2015**, and predominantly high coverage in **2016**. The shared timing is characteristic of a later reporting/feature rollout, not borrower-specific event absence.

**Decision — `EXCLUDE_LATE_ROLLOUT` from primary model (14):**

`open_acc_6m`, `open_act_il`, `open_il_12m`, `open_il_24m`, `mths_since_rcnt_il`, `total_bal_il`, `il_util`, `open_rv_12m`, `open_rv_24m`, `max_bal_bc`, `all_util`, `inq_fi`, `total_cu_tl`, `inq_last_12m`.

**Reason:** Almost no usable information during training/development, while strong availability in 2016 could cause year/availability artifacts and instability. They may be investigated in a *different* later-period cohort, but not in the current primary model. No reason to run detailed numerical EDA on these 14 for this model.

**Different finding — four historical event-recency fields:**

| Field | 2013 coverage | 2014 | 2015 | 2016 | Current decision |
|---|---:|---:|---:|---:|---|
| `mths_since_last_delinq` | 43.53% | 51.08% | 52.01% | 53.16% | `REVIEW_EVENT_MISSINGNESS` |
| `mths_since_last_major_derog` | 20.49% | 29.25% | 30.20% | 29.36% | `REVIEW_EVENT_MISSINGNESS` |
| `mths_since_recent_bc_dlq` | 22.42% | 26.85% | 26.16% | 25.85% | `REVIEW_EVENT_MISSINGNESS` |
| `mths_since_recent_revol_delinq` | 30.21% | 36.47% | 36.55% | 36.66% | `REVIEW_EVENT_MISSINGNESS` |

**Interpretation:** Unlike the 14 late-rollout fields, these four have observed values in *every* year. Their missingness **may** mean there was no corresponding prior delinquency event; it might also reflect incomplete reporting. We have **not yet confirmed the missingness mechanism or predictive usefulness**. Do not automatically discard, fill with zero, or cap them.

**Status after this stage:** 70 initial numerical EDA candidates − 14 late-rollout exclusions = **56 numerical candidates**, of which **four** are pending a targeted event-missingness investigation. Alongside them remain seven categorical candidates, one date candidate (`earliest_cr_line`), and two text candidates.

## Next investigation — four delinquency-recency fields

To decide whether these four justify further analysis:

1. Check valid numerical ranges and whether missingness of the four fields overlaps.
2. Compare missing/present status with `delinq_2yrs` **as a broad proxy only**. This latter field measures delinquencies in a recent two-year window, whereas the four fields cover different event types and possibly older events. Disagreement does **not** prove an invalid record.
3. On the development period only, compare borrower counts and default rates for missing versus observed recency groups. Do not inspect 2016 default outcomes for feature selection.
4. Decide **USE / MAY USE / EXCLUDE** for each, recording reasoning and any proposed missingness indicators. Do not impute or remove observations during EDA.

### Pending results log

| Feature | Missingness plausibly event-related? | Distinct risk information? | Final EDA decision / evidence |
|---|---|---|---|
| `mths_since_last_delinq` | To investigate | To investigate | Pending |
| `mths_since_last_major_derog` | To investigate | To investigate | Pending |
| `mths_since_recent_bc_dlq` | To investigate | To investigate | Pending |
| `mths_since_recent_revol_delinq` | To investigate | To investigate | Pending |

---

**Operating rule:** Do not physically mutate the interim Parquet during EDA. Log exclusion decisions at the point they are justified. A real but extreme financial value is not a data error just because it fails an IQR rule. Missing-value and transformation parameters must be fitted later using training data only.
