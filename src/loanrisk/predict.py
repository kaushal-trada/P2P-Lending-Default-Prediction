
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "saved_models/best_pipeline.pkl"


def load_model(path=MODEL_PATH):
    """Load the saved model pipeline and metadata."""

    bundle = joblib.load(path)

    required = {
        "pipeline",
        "threshold",
        "raw_columns",
        "model_name",
    }

    missing = required - set(bundle)

    if missing:
        raise ValueError(
            f"Model bundle missing entries: {sorted(missing)}"
        )

    return bundle


def predict_full(data, bundle):
    """Predict from complete raw LendingClub feature records."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Input DataFrame is empty.")

    if data.columns.has_duplicates:
        raise ValueError("Duplicate column names are not allowed.")

    required = bundle["raw_columns"]

    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(
            f"Missing {len(missing)} required features: {missing}"
        )

    # Preserve the exact feature order used during training.
    X = data.loc[:, required].copy()

    scores = np.asarray(
        bundle["pipeline"].predict_proba(X)
    )[:, 1]

    if not np.isfinite(scores).all():
        raise ValueError("Model produced non-finite scores.")

    threshold = float(bundle["threshold"])

    return pd.DataFrame(
        {
            "default_risk_score": scores,
            "predicted_default": (scores >= threshold).astype(int),
        },
        index=data.index,
    )
