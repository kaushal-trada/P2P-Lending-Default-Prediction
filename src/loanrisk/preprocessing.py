
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import numpy as np


INDICATOR_FEATURES = [
    "delinq_recency_missing",
    "emp_length_missing",
    "recent_inq_missing",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "addr_state",
    "purpose",
]


def make_preprocessor(numeric_features):
    """Build the preprocessing pipeline used for XGBoost."""

    numeric_columns = (
        list(numeric_features)
        + ["emp_length_years", "int_rate"]
        + INDICATOR_FEATURES
    )

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="MISSING",
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32,
            ),
        ),
    ])

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0,
    )
