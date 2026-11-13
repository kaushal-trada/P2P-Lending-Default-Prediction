
from pathlib import Path
from datetime import date
import sys

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loanrisk.predict import load_model, predict_full


def run_tests():
    bundle = load_model()
    pipeline = bundle["pipeline"]
    raw_cols = bundle["raw_columns"]
    threshold = float(bundle["threshold"])

    print("Model:", bundle["model_name"])
    print("Raw features:", len(raw_cols))
    print("Threshold:", threshold)

    assert len(raw_cols) == 55
    assert "features" in pipeline.named_steps

    # Use a few 2015 validation records, not the 2016 test set.
    source = (
        ROOT
        / "data/interim/lendingclub_2013_2016_36m_raw.parquet"
    )

    X = (
        pl.scan_parquet(source)
        .filter(
            (pl.col("issue_date") >= date(2015, 10, 1))
            & (pl.col("issue_date") < date(2016, 1, 1))
        )
        .select(raw_cols)
        .limit(10)
        .collect()
        .to_pandas()
    )

    assert len(X) == 10

    # Test 1: Batch prediction
    batch = predict_full(X, bundle)

    assert len(batch) == 10
    assert np.isfinite(batch["default_risk_score"]).all()
    assert batch["default_risk_score"].between(0, 1).all()

    print("PASS: Batch prediction")

    # Test 2: Single-row prediction matches batch
    single = predict_full(X.iloc[[0]], bundle)

    np.testing.assert_allclose(
        single["default_risk_score"].to_numpy(),
        batch["default_risk_score"].iloc[:1].to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )

    print("PASS: Single vs batch consistency")

    # Test 3: Saved pipeline produces the same probabilities
    direct = pipeline.predict_proba(X)[:, 1]

    np.testing.assert_allclose(
        direct,
        batch["default_risk_score"].to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )

    print("PASS: Saved pipeline parity")

    # Test 4: Classification follows the saved threshold
    expected = (
        batch["default_risk_score"].to_numpy() >= threshold
    ).astype(int)

    np.testing.assert_array_equal(
        expected,
        batch["predicted_default"].to_numpy(),
    )

    print("PASS: Threshold classification")

    # Test 5: Missing numerical values and employment length
    sample = X.iloc[[0]].copy()

    for col in [
        "emp_length",
        "revol_util",
        "mths_since_last_delinq",
        "mths_since_recent_inq",
    ]:
        sample[col] = np.nan

    prepared = pipeline.named_steps["features"].transform(sample)

    assert np.isnan(prepared["emp_length_years"].iloc[0])
    assert prepared["emp_length_missing"].iloc[0] == 1
    assert prepared["delinq_recency_missing"].iloc[0] == 1
    assert prepared["recent_inq_missing"].iloc[0] == 1

    result = predict_full(sample, bundle)
    assert np.isfinite(result["default_risk_score"]).all()

    print("PASS: Missing-value handling")

    # Test 6: Extra CSV column does not affect predictions
    extra = X.copy()
    extra["unused_column"] = "example"

    extra_result = predict_full(extra, bundle)

    np.testing.assert_allclose(
        extra_result["default_risk_score"].to_numpy(),
        batch["default_risk_score"].to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )

    print("PASS: Extra-column handling")

    # Test 7: Missing feature column is rejected
    incomplete = X.drop(columns=["loan_amnt"])

    try:
        predict_full(incomplete, bundle)
    except (ValueError, KeyError):
        print("PASS: Missing-column validation")
    else:
        raise AssertionError(
            "Missing required feature column was not rejected"
        )
        
    
    # Test 8: Empty string vs actual NaN
    blank = X.iloc[[0]].copy()
    nan_value = X.iloc[[0]].copy()

    blank["emp_length"] = ""
    nan_value["emp_length"] = np.nan

    blank_features = pipeline.named_steps["features"].transform(blank)
    nan_features = pipeline.named_steps["features"].transform(nan_value)

    np.testing.assert_array_equal(
        blank_features["emp_length_missing"].to_numpy(),
        nan_features["emp_length_missing"].to_numpy(),
    )

    np.testing.assert_allclose(
        predict_full(blank, bundle)["default_risk_score"],
        predict_full(nan_value, bundle)["default_risk_score"],
        rtol=1e-6,
        atol=1e-8,
    )

    print("PASS: Blank string vs NaN")

    # Test 9: Every row predicted individually vs batch
    individual_scores = []

    for i in range(len(X)):
        result = predict_full(X.iloc[[i]], bundle)
        individual_scores.append(
            result["default_risk_score"].iloc[0]
        )

    np.testing.assert_allclose(
        individual_scores,
        batch["default_risk_score"].to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )

    print("PASS: Individual vs batch predictions")

    # Test 10: Prediction must not modify input DataFrame
    original = X.copy(deep=True)

    predict_full(X, bundle)

    assert X.equals(original)

    print("PASS: Input DataFrame unchanged")

    # Test 11: Column order should not matter
    reordered = X[X.columns[::-1]].copy()
    reordered_results = predict_full(reordered, bundle)

    np.testing.assert_allclose(
        reordered_results["default_risk_score"].to_numpy(),
        batch["default_risk_score"].to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )

    print("PASS: Column-order independence")


    print("\nALL PIPELINE TESTS PASSED")


if __name__ == "__main__":
    run_tests()
