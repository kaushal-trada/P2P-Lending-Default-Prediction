\
"""Streamlit interface for LendingClub default-risk prediction."""

from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loanrisk.predict import load_model, predict_full
from loanrisk.quick import predict_quick


st.set_page_config(
    page_title="Loan Default Risk Prediction",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# REFERENCE DATA
# ============================================================

US_STATES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}


FEATURE_DESCRIPTIONS = {
    "loan_amnt": "Requested loan amount in US dollars",
    "int_rate": "Annual interest rate (%)",
    "annual_inc": "Borrower's annual income in US dollars",
    "dti": "Debt-to-income ratio",
    "fico_range_low": "Lower bound of FICO credit score",
    "emp_length": "Length of employment",
    "home_ownership": "Home ownership status",
    "purpose": "Purpose of the loan",
    "addr_state": "US state of the borrower",

    "delinq_2yrs": "Delinquencies in the last 2 years",
    "inq_last_6mths": "Credit inquiries in the last 6 months",
    "mths_since_last_delinq": "Months since the last delinquency",
    "open_acc": "Number of open credit accounts",
    "pub_rec": "Number of derogatory public records",
    "revol_bal": "Total revolving credit balance",
    "revol_util": "Revolving credit utilization (%)",
    "total_acc": "Total number of credit accounts",

    "collections_12_mths_ex_med":
        "Collections in the last 12 months excluding medical",
    "acc_now_delinq": "Number of currently delinquent accounts",
    "tot_coll_amt": "Total collection amounts owed",
    "tot_cur_bal": "Total current balance across accounts",
    "total_rev_hi_lim": "Total revolving credit limit",

    "acc_open_past_24mths": "Accounts opened in the last 24 months",
    "avg_cur_bal": "Average current account balance",
    "bc_open_to_buy": "Available bankcard credit",
    "bc_util": "Bankcard utilization (%)",
    "chargeoff_within_12_mths": "Charge-offs in the last 12 months",
    "delinq_amnt": "Amount owed on delinquent accounts",

    "mo_sin_old_il_acct":
        "Months since the oldest installment account opened",
    "mo_sin_old_rev_tl_op":
        "Months since the oldest revolving account opened",
    "mo_sin_rcnt_rev_tl_op":
        "Months since the most recent revolving account opened",
    "mo_sin_rcnt_tl":
        "Months since the most recent credit account opened",
    "mort_acc": "Number of mortgage accounts",
    "mths_since_recent_bc":
        "Months since the most recent bankcard account opened",
    "mths_since_recent_inq":
        "Months since the most recent credit inquiry",

    "num_accts_ever_120_pd":
        "Accounts ever 120 or more days past due",
    "num_actv_bc_tl": "Number of active bankcard accounts",
    "num_actv_rev_tl": "Number of active revolving accounts",
    "num_bc_sats": "Number of satisfactory bankcard accounts",
    "num_bc_tl": "Total number of bankcard accounts",
    "num_il_tl": "Number of installment accounts",
    "num_op_rev_tl": "Number of open revolving accounts",
    "num_rev_accts": "Total revolving credit accounts",
    "num_tl_120dpd_2m":
        "Accounts 120 or more days past due in the last 2 months",
    "num_tl_30dpd":
        "Accounts currently 30 or more days past due",
    "num_tl_90g_dpd_24m":
        "Accounts 90 or more days past due in the last 24 months",
    "num_tl_op_past_12m":
        "Accounts opened in the last 12 months",

    "pct_tl_nvr_dlq":
        "Percentage of accounts never delinquent",
    "percent_bc_gt_75":
        "Percentage of bankcard accounts above 75% utilization",
    "pub_rec_bankruptcies": "Number of public bankruptcy records",
    "tax_liens": "Number of tax liens",
    "tot_hi_cred_lim": "Total high credit or credit limits",
    "total_bal_ex_mort": "Total balance excluding mortgages",
    "total_bc_limit": "Total bankcard credit limits",
    "total_il_high_credit_limit":
        "Total installment credit high limits",
}


# Standard FICO scale versus observed training coverage
FICO_MIN = 300
FICO_MAX = 850

TRAIN_FICO_MIN = 660
TRAIN_FICO_MAX = 845


# ============================================================
# MODEL HELPERS
# ============================================================

@st.cache_resource
def get_model():
    """Load the fitted model once."""
    return load_model()


def show_prediction(result, threshold):
    """Display prediction result for a single loan."""

    score = float(result["default_risk_score"].iloc[0])
    predicted = int(result["predicted_default"].iloc[0])

    c1, c2, c3 = st.columns(3)

    c1.metric("Default Risk Score", f"{score:.2%}")

    c2.metric(
        "Predicted Class",
        "Default (1)" if predicted else "Fully Paid (0)",
    )

    c3.metric("Decision Threshold", f"{threshold:.2%}")

    if predicted:
        st.warning("This loan is flagged for elevated default risk.")
    else:
        st.success("This loan is below the model's risk threshold.")

    st.caption(
        "Model prediction only. This is not a lending approval "
        "or rejection decision."
    )


def validate_required_values(df):
    """Validate essential fields in uploaded CSV records."""

    issues = []

    rules = {
        "loan_amnt": (0, float("inf")),
        "int_rate": (0, 100),
        "fico_range_low": (FICO_MIN, FICO_MAX),
    }

    for feature, (lower, upper) in rules.items():

        values = pd.to_numeric(df[feature], errors="coerce")

        invalid = (
            values.isna()
            | ~np.isfinite(values)
            | (values < lower)
            | (values > upper)
        )

        # Loan amount and interest rate cannot be zero
        if feature in {"loan_amnt", "int_rate"}:
            invalid |= values == 0

        for i in np.flatnonzero(invalid.to_numpy()):

            value = df[feature].iloc[i]

            if pd.isna(value) or str(value).strip() == "":
                message = "Missing value"

            elif feature == "loan_amnt":
                message = "Loan amount must be greater than zero"

            elif feature == "int_rate":
                message = "Interest rate must be > 0 and <= 100"

            else:
                message = "FICO out of range. Valid range: 300–850"

            issues.append({
                "CSV Row": int(i) + 2,
                "Feature": feature,
                "Value": value,
                "Error": message,
            })

    return pd.DataFrame(
        issues,
        columns=["CSV Row", "Feature", "Value", "Error"]
    )


def check_fico_training_coverage(fico):
    """Warn about valid FICO scores outside training coverage."""

    values = pd.to_numeric(fico, errors="coerce")

    valid = values.between(FICO_MIN, FICO_MAX)
    outside = valid & ~values.between(
        TRAIN_FICO_MIN, TRAIN_FICO_MAX
    )

    if outside.any():
        st.warning(
            f"{int(outside.sum()):,} loan(s) have valid FICO scores "
            f"outside the observed training range "
            f"({TRAIN_FICO_MIN}–{TRAIN_FICO_MAX}). "
            "Predictions for these borrowers may be less reliable."
        )

    return outside


# ============================================================
# LOAD MODEL
# ============================================================

try:
    bundle = get_model()

except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()


raw_cols = list(bundle["raw_columns"])
threshold = float(bundle["threshold"])


st.title("Loan Default Risk Prediction")

st.write(
    "Estimate loan default risk using a trained LendingClub "
    "classification model."
)

st.caption(
    f"Model: {bundle['model_name']} | "
    f"Raw features: {len(raw_cols)} | "
    f"Threshold: {threshold:.4f}"
)


full_tab, quick_tab = st.tabs([
    "Full Mode — CSV",
    "Quick Mode — 15 Inputs",
])



# ============================================================
# FULL MODE
# ============================================================

with full_tab:
    st.subheader("Full Feature Prediction")

    st.write(
        "Upload a CSV containing all required model features. "
        "Valid loans will be predicted; invalid loans will be "
        "skipped and reported."
    )

    with st.expander("View Required Columns and Their Meanings"):
        feature_reference = pd.DataFrame({
            "Feature": raw_cols,
            "Description": [
                FEATURE_DESCRIPTIONS.get(
                    col, "Refer to the LendingClub data dictionary"
                )
                for col in raw_cols
            ],
        })

        st.dataframe(
            feature_reference,
            use_container_width=True,
            hide_index=True,
            height=400,
        )

        template = pd.DataFrame(columns=raw_cols)

        st.download_button(
            "Download Empty CSV Template",
            data=template.to_csv(index=False),
            file_name="loan_input_template.csv",
            mime="text/csv",
        )

    uploaded_file = st.file_uploader(
        "Upload Loan CSV",
        type=["csv"],
        key="full_csv",
    )

    st.caption(
        "Valid FICO range: 300–850. Scores outside the observed "
        "training range (approximately 660–845) receive a warning."
    )

    if uploaded_file is None:
        st.session_state.pop("full_results", None)
        st.session_state.pop("full_file_hash", None)

    else:
        try:
            file_hash = hashlib.sha256(
                uploaded_file.getvalue()
            ).hexdigest()

            if st.session_state.get("full_file_hash") != file_hash:
                st.session_state.pop("full_results", None)
                st.session_state["full_file_hash"] = file_hash

            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.error("The uploaded CSV contains no records.")

            else:
                st.write(f"Uploaded records: **{len(df):,}**")

                missing_cols = [
                    col for col in raw_cols
                    if col not in df.columns
                ]

                if missing_cols:
                    st.error(
                        f"Missing {len(missing_cols)} required columns."
                    )
                    st.write(missing_cols)

                else:
                    # Validate essential fields
                    invalid = validate_required_values(df)

                    # Identify invalid and valid rows
                    invalid_indices = (
                        invalid["CSV Row"].to_numpy(dtype=int) - 2
                    )

                    valid_mask = ~df.index.isin(invalid_indices)
                    valid_df = df.loc[valid_mask].copy()

                    n_valid = len(valid_df)
                    n_invalid = len(df) - n_valid

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Loans", len(df))
                    c2.metric("Valid Loans", n_valid)
                    c3.metric("Invalid Loans", n_invalid)

                    # Display invalid rows
                    if not invalid.empty:
                        st.warning(
                            f"{n_invalid} loan(s) have invalid inputs "
                            "and will be skipped. Valid loans can "
                            "still be predicted."
                        )

                        st.dataframe(
                            invalid,
                            use_container_width=True,
                            hide_index=True,
                        )

                    # Warn about scores outside training coverage
                    check_fico_training_coverage(
                        df["fico_range_low"]
                    )

                    # Warn about nonpositive annual income
                    income = pd.to_numeric(
                        valid_df["annual_inc"],
                        errors="coerce",
                    )

                    unusual_income = (
                        income.isna() | (income <= 0)
                    )

                    if unusual_income.any():
                        st.warning(
                            f"{int(unusual_income.sum())} valid "
                            "loan(s) have missing or nonpositive "
                            "annual income."
                        )

                    # Report optional missing values
                    optional_cols = [
                        col for col in raw_cols
                        if col not in {
                            "loan_amnt",
                            "int_rate",
                            "fico_range_low",
                        }
                    ]

                    missing_values = int(
                        valid_df[optional_cols].isna().sum().sum()
                    )

                    if missing_values:
                        st.info(
                            f"{missing_values:,} missing optional "
                            "feature values will be handled by "
                            "the fitted preprocessing pipeline."
                        )

                    # Predict valid rows only
                    if st.button(
                        f"Predict {n_valid} Valid Loans",
                        type="primary",
                        disabled=(n_valid == 0),
                    ):
                        with st.spinner("Generating predictions..."):
                            result = predict_full(
                                valid_df.reset_index(drop=True),
                                bundle,
                            ).reset_index(drop=True)

                        # Keep ALL original CSV rows
                        output = df.reset_index(drop=True).copy()

                        # Remove any previous prediction columns
                        for col in [
                            "default_risk_score",
                            "predicted_default",
                            "threshold",
                            "validation_status",
                            "validation_error",
                            "CSV Row",
                        ]:
                            if col in output.columns:
                                output = output.drop(columns=col)

                        output.insert(
                            0, "CSV Row",
                            np.arange(len(output)) + 2,
                        )

                        output["validation_status"] = "Predicted"
                        output["validation_error"] = ""

                        output["default_risk_score"] = np.nan
                        output["predicted_default"] = pd.Series(
                            pd.NA,
                            index=output.index,
                            dtype="Int64",
                        )
                        output["threshold"] = np.nan

                        # Mark invalid rows and their reasons
                        for _, issue in invalid.iterrows():
                            pos = int(issue["CSV Row"]) - 2

                            output.at[
                                pos, "validation_status"
                            ] = "Skipped - Invalid Input"

                            message = (
                                f"{issue['Feature']}: {issue['Error']}"
                            )

                            previous = output.at[
                                pos, "validation_error"
                            ]

                            output.at[
                                pos, "validation_error"
                            ] = (
                                f"{previous}; {message}"
                                if previous else message
                            )

                        # Insert predictions into original row positions
                        valid_positions = np.flatnonzero(valid_mask)

                        output.loc[
                            valid_positions, "default_risk_score"
                        ] = result[
                            "default_risk_score"
                        ].to_numpy()

                        output.loc[
                            valid_positions, "predicted_default"
                        ] = result[
                            "predicted_default"
                        ].to_numpy(dtype=int)

                        output.loc[
                            valid_positions, "threshold"
                        ] = threshold

                        st.session_state["full_results"] = {
                            "output": output,
                            "predictions": result,
                            "valid_count": n_valid,
                            "invalid_count": n_invalid,
                        }

        except Exception as exc:
            st.session_state.pop("full_results", None)
            st.error(f"CSV processing failed: {exc}")

    # --------------------------------------------------------
    # Prediction results
    # --------------------------------------------------------

    if "full_results" in st.session_state:
        saved = st.session_state["full_results"]

        output = saved["output"]
        result = saved["predictions"]

        st.divider()
        st.subheader("Prediction Results")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Loans", len(output))
        c2.metric("Predicted", saved["valid_count"])
        c3.metric("Skipped", saved["invalid_count"])
        c4.metric(
            "Flagged as Default",
            int(result["predicted_default"].sum()),
        )

        if len(result) == 1:
            show_prediction(result, threshold)

        st.caption(
            "Skipped loans have no prediction score. "
            "Their errors are included in the results table."
        )

        # Put the important result columns first
        first_cols = [
            "CSV Row",
            "validation_status",
            "validation_error",
            "default_risk_score",
            "predicted_default",
            "threshold",
        ]

        remaining_cols = [
            col for col in output.columns
            if col not in first_cols
        ]

        output = output[first_cols + remaining_cols]

        st.dataframe(
            output,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download All Results (Including Invalid Rows)",
            data=output.to_csv(index=False),
            file_name="loan_predictions.csv",
            mime="text/csv",
        )


# ============================================================
# QUICK MODE
# ============================================================

with quick_tab:

    st.subheader("Quick Loan Risk Estimation")

    st.warning(
        "Experimental demonstration: Only 15 inputs are supplied. "
        "The remaining 40 features use historical training defaults. "
        "Quick Mode is less reliable than Full Mode."
    )

    with st.form("quick_prediction_form"):

        c1, c2, c3 = st.columns(3)

        # ----------------------------------------------------
        # Column 1: Financial information
        # ----------------------------------------------------

        with c1:

            st.markdown("#### Loan & Financial Information")

            loan_amnt = st.number_input(
                "Loan Amount ($)",
                min_value=0.0,
                value=15000.0,
                step=500.0,
            )

            annual_inc = st.number_input(
                "Annual Income ($)",
                min_value=0.0,
                value=60000.0,
                step=1000.0,
            )

            int_rate = st.number_input(
                "Interest Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=13.0,
                step=0.5,
            )

            dti = st.number_input(
                "Debt-to-Income Ratio",
                min_value=0.0,
                value=18.0,
                step=0.5,
            )

            fico_range_low = st.slider(
                "FICO Score (Lower Bound)",
                min_value=300,
                max_value=850,
                value=690,
                step=5,
                help=(
                    "Standard FICO range: 300–850. "
                    "The model was trained on a narrower "
                    "observed score range."
                ),
            )

        # ----------------------------------------------------
        # Column 2: Borrower information
        # ----------------------------------------------------

        with c2:

            st.markdown("#### Borrower Information")

            purpose = st.selectbox(
                "Loan Purpose",
                [
                    "debt_consolidation",
                    "credit_card",
                    "home_improvement",
                    "major_purchase",
                    "medical",
                    "car",
                    "small_business",
                    "moving",
                    "vacation",
                    "other",
                ],
            )

            emp_length = st.selectbox(
                "Employment Length",
                [
                    "10+ years",
                    "9 years",
                    "8 years",
                    "7 years",
                    "6 years",
                    "5 years",
                    "4 years",
                    "3 years",
                    "2 years",
                    "1 year",
                    "< 1 year",
                    "",
                ],
            )

            home_ownership = st.selectbox(
                "Home Ownership",
                ["RENT", "MORTGAGE", "OWN", "ANY"],
            )

            state_codes = sorted(
                US_STATES,
                key=lambda code: US_STATES[code],
            )

            addr_state = st.selectbox(
                "US State",
                options=state_codes,
                index=state_codes.index("CA"),
                format_func=lambda code: (
                    f"{US_STATES[code]} ({code})"
                ),
                help=(
                    "Select the state by name. "
                    "The two-letter code is sent to the model."
                ),
            )

            revol_util = st.number_input(
                "Revolving Utilization (%)",
                min_value=0.0,
                value=50.0,
                step=1.0,
            )

        # ----------------------------------------------------
        # Column 3: Credit history
        # ----------------------------------------------------

        with c3:

            st.markdown("#### Credit History")

            open_acc = st.number_input(
                "Open Credit Accounts",
                min_value=0,
                value=10,
            )

            total_acc = st.number_input(
                "Total Credit Accounts",
                min_value=0,
                value=25,
            )

            delinq_2yrs = st.number_input(
                "Delinquencies in Past 2 Years",
                min_value=0,
                value=0,
            )

            inq_last_6mths = st.number_input(
                "Credit Inquiries in Last 6 Months",
                min_value=0,
                value=1,
            )

            revol_bal = st.number_input(
                "Revolving Balance ($)",
                min_value=0.0,
                value=12000.0,
                step=500.0,
            )

        submitted = st.form_submit_button(
            "Estimate Default Risk",
            type="primary",
        )

    # --------------------------------------------------------
    # Quick Mode validation and prediction
    # --------------------------------------------------------

    if submitted:

        errors = []

        if loan_amnt <= 0:
            errors.append(
                "Loan amount must be greater than zero."
            )

        if int_rate <= 0:
            errors.append(
                "Interest rate must be greater than zero."
            )

        if total_acc < open_acc:
            errors.append(
                "Total credit accounts cannot be fewer "
                "than open credit accounts."
            )

        if annual_inc <= 0:
            st.warning(
                "Annual income is zero. This may reduce "
                "the reliability of the prediction."
            )

        if not (
            TRAIN_FICO_MIN
            <= fico_range_low
            <= TRAIN_FICO_MAX
        ):
            st.warning(
                f"FICO {fico_range_low} is valid but outside "
                f"the observed training range "
                f"({TRAIN_FICO_MIN}–{TRAIN_FICO_MAX}). "
                "Interpret the prediction cautiously."
            )

        if errors:

            for message in errors:
                st.error(message)

        else:

            quick_data = pd.DataFrame([{
                "loan_amnt": loan_amnt,
                "int_rate": int_rate,
                "purpose": purpose,
                "annual_inc": annual_inc,
                "emp_length": emp_length,
                "home_ownership": home_ownership,
                "addr_state": addr_state,
                "fico_range_low": fico_range_low,
                "dti": dti,
                "revol_util": revol_util,
                "open_acc": open_acc,
                "total_acc": total_acc,
                "delinq_2yrs": delinq_2yrs,
                "inq_last_6mths": inq_last_6mths,
                "revol_bal": revol_bal,
            }])

            try:

                result = predict_quick(
                    quick_data,
                    bundle,
                )

                st.divider()
                st.subheader("Quick Prediction Result")

                show_prediction(result, threshold)

            except Exception as exc:

                st.error(
                    f"Prediction failed: {exc}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Educational classical machine-learning project using "
    "historical LendingClub loan data. Predicted scores may be "
    "miscalibrated and should not be used for automated "
    "lending decisions."
)
