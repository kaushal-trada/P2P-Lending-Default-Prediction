
"""Tests for the 15-feature Quick prediction mode."""

from pathlib import Path
from datetime import date
import sys

import numpy as np
import pandas as pd
import polars as pl
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loanrisk.predict import load_model, predict_full
from loanrisk.quick import (
    QUICK_FEATURES,
    expand_quick_inputs,
    predict_quick,
)


def run_tests():
    bundle = load_model()
    raw_cols = list(bundle["raw_columns"])
    defaults = bundle["historical_defaults"]

    source = (
        ROOT
        / "data/interim/lendingclub_2013_2016_36m_raw.parquet"
    )

    # Use development validation data, not the 2016 test set.
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

    quick = X[QUICK_FEATURES].copy()
    original = quick.copy(deep=True)

    # 1. Feature-count and schema checks
    assert len(QUICK_FEATURES) == 15
    assert len(raw_cols) == 55

    expanded = expand_quick_inputs(quick, bundle)

    assert expanded.shape == (10, 55)
    assert list(expanded.columns) == raw_cols

    print("PASS: 15-to-55 feature expansion")

    # 2. Manual inputs must remain unchanged
    assert_frame_equal(
        expanded[QUICK_FEATURES],
        quick,
    )

    # 3. All 40 hidden fields must use saved defaults
    hidden_cols = [
        col for col in raw_cols
        if col not in QUICK_FEATURES
    ]

    assert len(hidden_cols) == 40

    for col in hidden_cols:
        assert col in defaults
        assert pd.notna(defaults[col])
        assert expanded[col].eq(defaults[col]).all()

    print("PASS: Historical defaults for 40 fields")

    # 4. Valid Quick-mode predictions
    results = predict_quick(quick, bundle)

    assert len(results) == 10
    assert np.isfinite(results["default_risk_score"]).all()
    assert results["default_risk_score"].between(0, 1).all()

    print("PASS: Quick-mode batch prediction")

    # 5. Quick mode must match Full inference
    # when supplied with the same expanded inputs.
    direct = predict_full(expanded, bundle)

    np.testing.assert_allclose(
        results["default_risk_score"].to_numpy(),
        direct["default_risk_score"].to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )

    print("PASS: Quick vs expanded Full-mode parity")

    # 6. Individual and batch prediction consistency
    individual = [
        predict_quick(quick.iloc[[i]], bundle)[
            "default_risk_score"
        ].iloc[0]
        for i in range(len(quick))
    ]

    np.testing.assert_allclose(
        individual,
        results["default_risk_score"].to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )

    print("PASS: Single vs batch consistency")

    # 7. Empty employment length vs actual NaN
    blank = quick.iloc[[0]].copy()
    missing = quick.iloc[[0]].copy()

    blank["emp_length"] = ""
    missing["emp_length"] = np.nan

    p_blank = predict_quick(blank, bundle)
    p_missing = predict_quick(missing, bundle)

    np.testing.assert_allclose(
        p_blank["default_risk_score"].to_numpy(),
        p_missing["default_risk_score"].to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )

    print("PASS: Blank vs NaN employment length")

    # 8. Column order must not affect predictions
    shuffled = quick[QUICK_FEATURES[::-1]].copy()

    reordered = predict_quick(shuffled, bundle)

    np.testing.assert_allclose(
        reordered["default_risk_score"].to_numpy(),
        results["default_risk_score"].to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )

    print("PASS: Column-order independence")

    # 9. Missing manual-input column must be rejected
    incomplete = quick.drop(columns=["loan_amnt"])

    try:
        predict_quick(incomplete, bundle)
    except ValueError:
        print("PASS: Missing Quick-mode column rejected")
    else:
        raise AssertionError(
            "Quick mode accepted a missing input column."
        )

    # 10. Input must remain unchanged
    assert_frame_equal(quick, original)

    print("PASS: Original inputs unchanged")

    print("\nALL QUICK-MODE TESTS PASSED")


if __name__ == "__main__":
    run_tests()
