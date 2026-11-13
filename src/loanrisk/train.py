
import argparse
import json
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    PredefinedSplit,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .features import LoanFeatureBuilder
from .preprocessing import make_preprocessor
from .inference_defaults import calculate_training_defaults


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data/interim/lendingclub_2013_2016_36m_raw.parquet"
)
FEATURE_PATH = ROOT / "config/feature_manifest.csv"
CONFIG_PATH = ROOT / "config/model_candidates.json"

MODEL_PATH = ROOT / "saved_models/best_pipeline.pkl"

REPORT_DIR = ROOT / "reports/modeling"

SEARCH_REPORT = REPORT_DIR / "parameter_search_results.csv"
COMPARISON_REPORT = REPORT_DIR / "train_model_comparison.csv"

CATEGORICAL_FEATURES = [
    "home_ownership",
    "addr_state",
    "purpose",
]

EXTRA_RAW_FEATURES = [
    "emp_length",
    "int_rate",
    *CATEGORICAL_FEATURES,
]


def load_configuration():
    """Load approved features and hyperparameter grids."""

    manifest = pd.read_csv(FEATURE_PATH)

    enabled = (
        manifest["origination_model"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    numeric_features = manifest.loc[
        enabled
        & manifest["model_status"].eq("CANDIDATE_NUMERIC"),
        "feature",
    ].tolist()

    selected_features = set(
        manifest.loc[enabled, "feature"]
    )

    raw_features = list(
        dict.fromkeys(numeric_features + EXTRA_RAW_FEATURES)
    )

    if len(numeric_features) != 50:
        raise ValueError(
            f"Expected 50 numeric features, got {len(numeric_features)}"
        )

    if len(raw_features) != 55:
        raise ValueError(
            f"Expected 55 raw features, got {len(raw_features)}"
        )

    if not set(raw_features).issubset(selected_features):
        raise ValueError(
            "Selected features do not match the approved manifest."
        )

    with open(CONFIG_PATH, encoding="utf-8") as file:
        configs = json.load(file)

    return numeric_features, raw_features, configs


def load_development_data(raw_features):
    """Load training and validation loans, excluding 2016."""

    columns = raw_features + ["issue_date", "target"]

    lf = pl.scan_parquet(DATA_PATH).select(columns)

    train_df = (
        lf.filter(
            (pl.col("issue_date") >= date(2013, 1, 1))
            & (pl.col("issue_date") < date(2015, 10, 1))
        )
        .collect()
        .to_pandas()
    )

    val_df = (
        lf.filter(
            (pl.col("issue_date") >= date(2015, 10, 1))
            & (pl.col("issue_date") < date(2016, 1, 1))
        )
        .collect()
        .to_pandas()
    )

    if len(train_df) != 457355 or len(val_df) != 88810:
        raise ValueError(
            "Development cohort differs from Notebook 04. "
            f"Train={len(train_df)}, validation={len(val_df)}"
        )

    X_train = train_df[raw_features].copy()
    X_val = val_df[raw_features].copy()

    y_train = train_df["target"].to_numpy(dtype=np.int8)
    y_val = val_df["target"].to_numpy(dtype=np.int8)

    train_dates = pd.to_datetime(train_df["issue_date"])

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Raw features:", len(raw_features))

    return X_train, X_val, y_train, y_val, train_dates


def build_pipeline(family, numeric_features, fixed_params):
    """Construct a complete pipeline for one model family."""

    feature_builder = LoanFeatureBuilder(
        numeric_features=list(numeric_features)
    )

    if family == "xgboost":
        return Pipeline([
            ("features", feature_builder),
            ("preprocessing", make_preprocessor(numeric_features)),
            ("model", XGBClassifier(**fixed_params)),
        ])

    if family == "catboost":
        return Pipeline([
            ("features", feature_builder),
            ("model", CatBoostClassifier(**fixed_params)),
        ])

    raise ValueError(f"Unknown model family: {family}")


def get_fit_params(family):
    """Supply CatBoost's categorical feature names during fitting."""

    if family == "catboost":
        return {
            "model__cat_features": CATEGORICAL_FEATURES,
        }

    return {}


def make_time_split(train_dates):
    """Use pre-April-2015 loans to tune on April-September 2015."""

    fold = np.where(
        train_dates < pd.Timestamp("2015-04-01"),
        -1,
        0,
    )

    return PredefinedSplit(test_fold=fold)


def search_hyperparameters(
    family,
    config,
    numeric_features,
    X_train,
    y_train,
    cv,
):
    """Search configurations using internal chronological PR-AUC."""

    pipeline = build_pipeline(
        family,
        numeric_features,
        config["fixed"],
    )

    n_configs = len(ParameterGrid(config["grid"]))

    print(f"\nSearching {family}: {n_configs} configurations")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=config["grid"],
        scoring="average_precision",
        cv=cv,
        refit=False,
        n_jobs=1,
        verbose=1,
        error_score="raise",
        return_train_score=False,
    )

    search.fit(
        X_train,
        y_train,
        **get_fit_params(family),
    )

    trials = pd.DataFrame(search.cv_results_)

    trials["model"] = family
    trials["params"] = trials["params"].astype(str)

    trials = trials[
        [
            "model",
            "rank_test_score",
            "mean_test_score",
            "mean_fit_time",
            "params",
        ]
    ].sort_values("rank_test_score")

    print("Best internal PR-AUC:", round(search.best_score_, 5))
    print("Best parameters:", search.best_params_)

    return (
        search.best_params_,
        float(search.best_score_),
        trials,
    )


def choose_threshold(y_true, probabilities):
    """Choose the threshold maximizing F1 on validation."""

    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12,
        )
    )

    best_index = int(np.argmax(f1_scores))

    return float(thresholds[best_index])


def evaluate_model(y_true, probabilities):
    """Evaluate ranking, classification and probability quality."""

    threshold = choose_threshold(y_true, probabilities)

    predictions = (probabilities >= threshold).astype(int)

    return {
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),
        "threshold": threshold,
        "f1": f1_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "brier": brier_score_loss(
            y_true, probabilities
        ),
    }


def fit_finalist(
    family,
    config,
    best_params,
    numeric_features,
    X_train,
    y_train,
    X_val,
    y_val,
):
    """Refit the best family configuration on all training loans."""

    pipeline = build_pipeline(
        family,
        numeric_features,
        config["fixed"],
    )

    pipeline.set_params(**best_params)

    start = time.perf_counter()

    pipeline.fit(
        X_train,
        y_train,
        **get_fit_params(family),
    )

    train_seconds = time.perf_counter() - start

    probabilities = pipeline.predict_proba(X_val)[:, 1]

    metrics = evaluate_model(y_val, probabilities)

    print(
        f"{family}: "
        f"PR-AUC={metrics['pr_auc']:.5f}, "
        f"ROC-AUC={metrics['roc_auc']:.5f}, "
        f"F1={metrics['f1']:.5f}"
    )

    return pipeline, metrics, train_seconds


def save_winner(bundle, X_val):
    """Save the winning pipeline and verify it survives reloading."""

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_path = MODEL_PATH.with_name(
        MODEL_PATH.name + ".tmp"
    )

    joblib.dump(
        bundle,
        temp_path,
        compress=3,
    )

    restored = joblib.load(temp_path)

    sample = X_val.iloc[:100]

    before = bundle["pipeline"].predict_proba(sample)[:, 1]

    after = restored["pipeline"].predict_proba(sample)[:, 1]

    np.testing.assert_allclose(
        before,
        after,
        rtol=1e-6,
        atol=1e-7,
    )

    temp_path.replace(MODEL_PATH)

    print("Model serialization: PASS")
    print("Saved:", MODEL_PATH)


def train(force=False):
    """Tune both families and save the best complete pipeline."""

    if MODEL_PATH.exists() and not force:
        raise FileExistsError(
            f"{MODEL_PATH} already exists. "
            "Use --force to explicitly replace it."
        )

    numeric_features, raw_features, configs = load_configuration()

    X_train, X_val, y_train, y_val, train_dates = (
        load_development_data(raw_features)
    )

    cv = make_time_split(train_dates)

    fitted_models = {}
    best_configs = {}
    comparison_rows = []
    search_tables = []

    for family in ["xgboost", "catboost"]:
        config = configs[family]

        best_params, internal_score, trials = (
            search_hyperparameters(
                family=family,
                config=config,
                numeric_features=numeric_features,
                X_train=X_train,
                y_train=y_train,
                cv=cv,
            )
        )

        search_tables.append(trials)

        pipeline, metrics, train_seconds = fit_finalist(
            family=family,
            config=config,
            best_params=best_params,
            numeric_features=numeric_features,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        fitted_models[family] = pipeline

        best_configs[family] = {
            "fixed": config["fixed"],
            "best_params": best_params,
            "internal_pr_auc": internal_score,
        }

        comparison_rows.append({
            "model": family,
            "internal_pr_auc": internal_score,
            **metrics,
            "train_seconds": train_seconds,
        })

    search_results = pd.concat(
        search_tables, ignore_index=True
    )

    comparison = (
        pd.DataFrame(comparison_rows)
        .sort_values(
            ["pr_auc", "roc_auc", "train_seconds"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    print("\nFinal validation comparison:")
    print(comparison.round(5).to_string(index=False))

    winner = comparison.iloc[0]

    winner_name = str(winner["model"])
    winning_pipeline = fitted_models[winner_name]

    default_values = calculate_training_defaults(
        X_train,
        numeric_features,
    )

    bundle = {
        "pipeline": winning_pipeline,
        "model_name": winner_name,
        "threshold": float(winner["threshold"]),
        "prediction_stage": "loan_origination_after_pricing",
        "raw_columns": raw_features,
        "numeric_features": numeric_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "historical_defaults": default_values,
        "selection_metric": "average_precision",
        "training_period": "2013-01 to 2015-09",
        "validation_period": "2015-10 to 2015-12",
        "internal_tuning_period": "2015-04 to 2015-09",
        "selected_configuration": best_configs[winner_name],
        "candidate_searches": best_configs,
        "validation_metrics": {
            "pr_auc": float(winner["pr_auc"]),
            "roc_auc": float(winner["roc_auc"]),
            "f1": float(winner["f1"]),
            "precision": float(winner["precision"]),
            "recall": float(winner["recall"]),
            "balanced_accuracy": float(
                winner["balanced_accuracy"]
            ),
            "brier": float(winner["brier"]),
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    search_results.to_csv(
        SEARCH_REPORT,
        index=False,
    )

    comparison.to_csv(
        COMPARISON_REPORT,
        index=False,
    )

    save_winner(bundle, X_val)

    print("\nTraining complete")
    print("Selected model:", winner_name)
    print("Validation PR-AUC:", round(winner["pr_auc"], 5))
    print("Threshold:", round(winner["threshold"], 5))
    print("Raw features:", len(raw_features))
    print("2016 holdout: untouched")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing final model.",
    )

    args = parser.parse_args()

    train(force=args.force)
