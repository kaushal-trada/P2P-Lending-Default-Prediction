
"""Quick-mode inference using 15 borrower inputs."""

import pandas as pd

from loanrisk.predict import predict_full


QUICK_FEATURES = [
    "loan_amnt",
    "int_rate",
    "purpose",
    "annual_inc",
    "emp_length",
    "home_ownership",
    "addr_state",
    "fico_range_low",
    "dti",
    "revol_util",
    "open_acc",
    "total_acc",
    "delinq_2yrs",
    "inq_last_6mths",
    "revol_bal",
]


def expand_quick_inputs(quick_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Expand 15 inputs into the full 55-feature schema."""

    if not isinstance(quick_df, pd.DataFrame):
        raise TypeError("Quick inputs must be a pandas DataFrame.")

    if not quick_df.columns.is_unique:
        raise ValueError("Duplicate input columns are not allowed.")

    raw_cols = list(bundle["raw_columns"])
    defaults = bundle["historical_defaults"]

    missing_inputs = [
        col for col in QUICK_FEATURES
        if col not in quick_df.columns
    ]

    if missing_inputs:
        raise ValueError(
            f"Missing Quick-mode columns: {missing_inputs}"
        )

    assert len(QUICK_FEATURES) == 15
    assert len(raw_cols) == 55

    missing_defaults = [
        col for col in raw_cols
        if col not in QUICK_FEATURES and col not in defaults
    ]

    if missing_defaults:
        raise ValueError(
            f"Missing historical defaults: {missing_defaults}"
        )

    full_df = quick_df[QUICK_FEATURES].copy()

    for col in raw_cols:
        if col not in QUICK_FEATURES:
            full_df[col] = defaults[col]

    return full_df[raw_cols]


def predict_quick(quick_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Predict using the frozen model and historical defaults."""

    full_df = expand_quick_inputs(quick_df, bundle)
    return predict_full(full_df, bundle)
