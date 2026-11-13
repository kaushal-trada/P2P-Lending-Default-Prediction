
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class LoanFeatureBuilder(BaseEstimator, TransformerMixin):
    """Prepare raw LendingClub features for the origination model."""

    def __init__(self, numeric_features, min_purpose_count=1000):
        self.numeric_features = numeric_features
        self.min_purpose_count = min_purpose_count

    @staticmethod
    def clean_purpose(series):
        return (
            series.astype("string")
            .str.strip()
            .str.lower()
            .replace("", pd.NA)
            .fillna("missing")
        )

    def fit(self, X, y=None):
        required = (
            list(self.numeric_features)
            + ["emp_length", "int_rate",
               "home_ownership", "addr_state", "purpose"]
        )

        missing = sorted(set(required) - set(X.columns))
        if missing:
            raise ValueError(f"Missing raw columns: {missing}")

        counts = self.clean_purpose(X["purpose"]).value_counts()

        self.common_purposes_ = set(
            counts[counts >= self.min_purpose_count].index
        )

        self.numeric_cols_ = (
            list(self.numeric_features)
            + ["emp_length_years", "int_rate"]
        )

        self.indicator_cols_ = [
            "delinq_recency_missing",
            "emp_length_missing",
            "recent_inq_missing",
        ]

        self.categorical_cols_ = [
            "home_ownership", "addr_state", "purpose"
        ]

        self.model_cols_ = (
            self.numeric_cols_
            + self.indicator_cols_
            + self.categorical_cols_
        )

        return self

    def transform(self, X):
        check_is_fitted(self, "model_cols_")

        required = (
            list(self.numeric_features)
            + ["emp_length", "int_rate",
               "home_ownership", "addr_state", "purpose"]
        )

        missing = sorted(set(required) - set(X.columns))
        if missing:
            raise ValueError(f"Missing raw columns: {missing}")

        result = pd.DataFrame(index=X.index)

        for col in self.numeric_features:
            values = X[col].astype("string").str.strip()

            if col == "revol_util":
                values = values.str.replace(
                    "%", "", regex=False
                )

            result[col] = pd.to_numeric(
                values, errors="coerce"
            ).astype(np.float32)

        result["dti"] = result["dti"].where(
            np.isfinite(result["dti"]) & (result["dti"] >= 0)
        )

        emp = X["emp_length"].astype("string").str.strip()

        years = pd.to_numeric(
            emp.str.extract(
                r"^(\d+)\s+years?$", expand=False
            ),
            errors="coerce",
        )

        years = years.mask(emp.eq("< 1 year").fillna(False), 0)
        years = years.mask(emp.eq("10+ years").fillna(False), 10)

        result["emp_length_years"] = years.astype(np.float32)

        result["int_rate"] = pd.to_numeric(
            X["int_rate"]
            .astype("string")
            .str.replace("%", "", regex=False)
            .str.strip(),
            errors="coerce",
        ).astype(np.float32)

        result["delinq_recency_missing"] = (
            result["mths_since_last_delinq"]
            .isna()
            .astype(np.int8)
        )

        result["emp_length_missing"] = (
            result["emp_length_years"]
            .isna()
            .astype(np.int8)
        )

        result["recent_inq_missing"] = (
            result["mths_since_recent_inq"]
            .isna()
            .astype(np.int8)
        )

        for col in ["home_ownership", "addr_state"]:
            result[col] = (
                X[col]
                .astype("string")
                .str.strip()
                .fillna("MISSING")
                .astype(object)
            )

        purpose = self.clean_purpose(X["purpose"])

        result["purpose"] = (
            purpose.where(
                purpose.isin(self.common_purposes_),
                "rare_other",
            ).astype(object)
        )

        return result[self.model_cols_]
