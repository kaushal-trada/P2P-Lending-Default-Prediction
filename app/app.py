
"""Streamlit interface for LendingClub default-risk prediction."""

from pathlib import Path
import sys

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


@st.cache_resource
def get_model():
    """Load the fitted model once per Streamlit session."""
    return load_model()


def show_prediction(result, threshold):
    """Display a single prediction."""

    score = float(result["default_risk_score"].iloc[0])
    predicted = int(result["predicted_default"].iloc[0])

    col1, col2, col3 = st.columns(3)

    col1.metric("Default Risk Score", f"{score:.2%}")
    col2.metric(
        "Predicted Class",
        "Default (1)" if predicted else "Fully Paid (0)",
    )
    col3.metric("Decision Threshold", f"{threshold:.2%}")

    if predicted:
        st.warning("This loan is flagged for elevated default risk.")
    else:
        st.success("This loan is below the model's risk threshold.")

    st.caption(
        "Model prediction only. This is not a lending approval "
        "or rejection decision."
    )


try:
    bundle = get_model()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()


raw_cols = list(bundle["raw_columns"])
threshold = float(bundle["threshold"])

st.title("Loan Default Risk Prediction")
st.write(
    "Predict loan default risk using a trained LendingClub "
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
        "Upload a CSV containing the 55 raw model features. "
        "Missing values inside existing columns are handled "
        "by the fitted preprocessing pipeline."
    )

    with st.expander("View required CSV column names"):
        st.code(", ".join(raw_cols), language="text")

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

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.error("The uploaded CSV contains no records.")
                st.stop()

            st.write(f"Uploaded records: **{len(df):,}**")

            missing_cols = [
                col for col in raw_cols if col not in df.columns
            ]

            if missing_cols:
                st.error(
                    f"Missing {len(missing_cols)} required columns:"
                )
                st.write(missing_cols)

            else:
                missing_values = int(
                    df[raw_cols].isna().sum().sum()
                )

                if missing_values > 0:
                    st.info(
                        f"{missing_values:,} missing feature values "
                        "detected. The fitted preprocessing pipeline "
                        "will handle them."
                    )

                if st.button(
                    "Run Full Prediction",
                    type="primary",
                ):
                    with st.spinner("Generating predictions..."):
                        result = predict_full(df, bundle)

                    st.session_state["full_results"] = {
                        "input": df.copy(),
                        "predictions": result.copy(),
                        "upload_name": uploaded_file.name,
                    }

        except Exception as exc:
            st.error(f"CSV processing failed: {exc}")

    if "full_results" in st.session_state:
        saved = st.session_state["full_results"]
        original = saved["input"]
        result = saved["predictions"]

        st.divider()
        st.subheader("Prediction Results")

        if len(result) == 1:
            show_prediction(result, threshold)

        else:
            col1, col2, col3 = st.columns(3)

            col1.metric("Loans", f"{len(result):,}")

            col2.metric(
                "Flagged as Default",
                f"{int(result['predicted_default'].sum()):,}",
            )

            col3.metric(
                "Mean Risk Score",
                f"{result['default_risk_score'].mean():.2%}",
            )

        # Retain original input columns and append predictions.
        output = original.reset_index(drop=True).copy()

        for col in [
            "default_risk_score",
            "predicted_default",
            "threshold",
        ]:
            if col in output.columns:
                output = output.drop(columns=[col])

        output["default_risk_score"] = (
            result["default_risk_score"].to_numpy()
        )
        output["predicted_default"] = (
            result["predicted_default"].to_numpy()
        )
        output["threshold"] = threshold

        st.dataframe(
            output,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Prediction Results",
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

        with c1:
            loan_amnt = st.number_input(
                "Loan Amount ($)",
                min_value=1.0,
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

            fico_range_low = st.number_input(
                "FICO Score (Lower Bound)",
                min_value=300,
                max_value=850,
                value=690,
            )

        with c2:
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

            addr_state = st.text_input(
                "US State Code",
                value="CA",
                max_chars=2,
            )

            revol_util = st.number_input(
                "Revolving Utilization (%)",
                min_value=0.0,
                value=50.0,
                step=1.0,
            )

        with c3:
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

    if submitted:

        quick_data = pd.DataFrame([{
            "loan_amnt": loan_amnt,
            "int_rate": int_rate,
            "purpose": purpose,
            "annual_inc": annual_inc,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "addr_state": addr_state.strip().upper(),
            "fico_range_low": fico_range_low,
            "dti": dti,
            "revol_util": revol_util,
            "open_acc": open_acc,
            "total_acc": total_acc,
            "delinq_2yrs": delinq_2yrs,
            "inq_last_6mths": inq_last_6mths,
            "revol_bal": revol_bal,
        }])

        if (
            len(quick_data["addr_state"].iloc[0]) != 2
            or not quick_data["addr_state"].iloc[0].isalpha()
        ):
            st.error("Enter a valid two-letter US state code.")

        else:
            try:
                result = predict_quick(quick_data, bundle)

                st.divider()
                st.subheader("Quick Prediction Result")

                show_prediction(result, threshold)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


st.divider()

st.caption(
    "Educational classical machine-learning project using "
    "historical LendingClub loan data. Predicted scores may be "
    "miscalibrated and should not be used for automated lending decisions."
)
