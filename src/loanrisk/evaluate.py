
"""Evaluate the frozen LendingClub model on the 2016 cohort."""

from pathlib import Path
from datetime import date
import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from loanrisk.predict import predict_full


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = ROOT / "saved_models/best_pipeline.pkl"
DEFAULT_DATA = (
    ROOT / "data/interim/lendingclub_2013_2016_36m_raw.parquet"
)
DEFAULT_OUTPUT = ROOT / "reports/evaluation/2016_frozen"


def load_test_data(path, raw_cols):
    """Load only the independent 2016 cohort."""

    columns = ["issue_date", "target"] + raw_cols

    df = (
        pl.scan_parquet(path)
        .filter(
            (pl.col("issue_date") >= date(2016, 1, 1))
            & (pl.col("issue_date") < date(2017, 1, 1))
        )
        .select(columns)
        .collect()
        .to_pandas()
    )

    if df.empty:
        raise ValueError("No 2016 records were found.")

    if df["target"].isna().any():
        raise ValueError("Test data contains missing target values.")

    if not df["target"].isin([0, 1, "0", "1"]).all():
        raise ValueError("Target must contain binary labels 0 and 1.")

    return df


def classification_metrics(y, p, threshold):
    """Calculate metrics using the frozen classification threshold."""

    predicted = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predicted, labels=[0, 1]
    ).ravel()

    metrics = {
        "samples": int(len(y)),
        "defaults": int(y.sum()),
        "default_rate": float(y.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predicted)
        ),
        "precision": float(
            precision_score(y, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predicted, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predicted, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "mean_predicted_score": float(p.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn)),
        "predicted_positive_rate": float(predicted.mean()),
    }

    return metrics, predicted


def calibration_analysis(y, p):
    """Group observations into predicted-score deciles."""

    df = pd.DataFrame({
        "target": y,
        "score": p,
    })

    # Decile 1 is the lowest predicted risk.
    df["calibration_decile"] = (
        pd.qcut(
            df["score"].rank(method="first"),
            q=10,
            labels=False,
        ) + 1
    )

    result = (
        df.groupby("calibration_decile", observed=True)
        .agg(
            loans=("target", "size"),
            defaults=("target", "sum"),
            mean_score=("score", "mean"),
            actual_default_rate=("target", "mean"),
        )
        .reset_index()
    )

    result["calibration_gap"] = (
        result["actual_default_rate"] - result["mean_score"]
    )

    return result


def risk_decile_analysis(y, p):
    """Calculate risk lift and cumulative default capture."""

    df = pd.DataFrame({
        "target": y,
        "score": p,
    })

    # Decile 1 = highest predicted risk.
    ascending_decile = pd.qcut(
        df["score"].rank(method="first"),
        q=10,
        labels=False,
    )

    df["risk_decile"] = 10 - ascending_decile

    result = (
        df.groupby("risk_decile", observed=True)
        .agg(
            loans=("target", "size"),
            defaults=("target", "sum"),
            mean_score=("score", "mean"),
            default_rate=("target", "mean"),
        )
        .reset_index()
        .sort_values("risk_decile")
    )

    base_rate = y.mean()

    result["lift"] = result["default_rate"] / base_rate

    result["cumulative_loans_pct"] = (
        result["loans"].cumsum() / len(y) * 100
    )

    result["defaults_captured_pct"] = (
        result["defaults"].cumsum() / y.sum() * 100
    )

    return result


def monthly_analysis(dates, y, p, threshold):
    """Evaluate discrimination and classification by issue month."""

    df = pd.DataFrame({
        "issue_date": pd.to_datetime(dates),
        "target": y,
        "score": p,
    })

    df["month"] = df["issue_date"].dt.strftime("%Y-%m")

    results = []

    for month, group in df.groupby("month", sort=True):
        actual = group["target"].to_numpy()
        scores = group["score"].to_numpy()

        predictions = (scores >= threshold).astype(int)

        results.append({
            "month": month,
            "loans": int(len(group)),
            "default_rate": float(actual.mean()),
            "mean_score": float(scores.mean()),
            "roc_auc": (
                float(roc_auc_score(actual, scores))
                if len(np.unique(actual)) == 2
                else np.nan
            ),
            "pr_auc": (
                float(average_precision_score(actual, scores))
                if actual.sum() > 0
                else np.nan
            ),
            "recall": float(
                recall_score(
                    actual, predictions, zero_division=0
                )
            ),
        })

    return pd.DataFrame(results)


def save_plots(y, p, threshold, calibration, risk, monthly, out):
    """Save evaluation visualizations."""

    # ROC curve
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"XGBoost AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("2016 ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=150)
    plt.close()

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {ap:.4f}")
    plt.axhline(
        y.mean(), linestyle="--",
        label=f"Baseline = {y.mean():.4f}"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("2016 Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "pr_curve.png", dpi=150)
    plt.close()

    # Calibration
    plt.figure(figsize=(7, 5))
    plt.plot(
        calibration["mean_score"],
        calibration["actual_default_rate"],
        "o-",
        label="Observed",
    )
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Mean Predicted Risk")
    plt.ylabel("Actual Default Rate")
    plt.title("2016 Calibration by Decile")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "calibration.png", dpi=150)
    plt.close()

    # Cumulative gains
    plt.figure(figsize=(7, 5))
    plt.plot(
        [0] + risk["cumulative_loans_pct"].tolist(),
        [0] + risk["defaults_captured_pct"].tolist(),
        marker="o",
        label="Model",
    )
    plt.plot([0, 100], [0, 100], "--", label="Random")
    plt.xlabel("Percentage of Loans Reviewed")
    plt.ylabel("Percentage of Defaults Captured")
    plt.title("2016 Cumulative Gains")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "cumulative_gains.png", dpi=150)
    plt.close()

    # Monthly ROC-AUC
    plt.figure(figsize=(9, 5))
    plt.plot(
        monthly["month"],
        monthly["roc_auc"],
        marker="o",
    )
    plt.xticks(rotation=45)
    plt.xlabel("Issue Month")
    plt.ylabel("ROC-AUC")
    plt.title("2016 Monthly Model Stability")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "monthly_stability.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the saved model on 2016 loans."
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Load frozen model. No refitting or training.
    bundle = joblib.load(args.model)

    raw_cols = list(bundle["raw_columns"])
    threshold = float(bundle["threshold"])

    print("Model:", bundle["model_name"])
    print("Model file:", args.model)
    print("Raw features:", len(raw_cols))
    print("Frozen threshold:", threshold)

    # 2016 evaluation cohort
    test_df = load_test_data(args.data, raw_cols)

    X = test_df[raw_cols]
    y = test_df["target"].astype(int).to_numpy()
    dates = test_df["issue_date"]

    print("\n2016 records:", len(test_df))
    print("Defaults:", int(y.sum()))
    print("Default rate:", round(float(y.mean()), 4))

    # Predict using the exact saved pipeline.
    result = predict_full(X, bundle)

    p = result["default_risk_score"].to_numpy(dtype=float)

    assert len(p) == len(y)
    assert np.isfinite(p).all()
    assert np.all((p >= 0) & (p <= 1))

    metrics, predicted = classification_metrics(
        y, p, threshold
    )

    np.testing.assert_array_equal(
        predicted,
        result["predicted_default"].to_numpy(dtype=int),
    )

    calibration = calibration_analysis(y, p)
    risk = risk_decile_analysis(y, p)
    monthly = monthly_analysis(dates, y, p, threshold)

    # Save metrics
    metrics["model_name"] = bundle["model_name"]
    metrics["evaluation_period"] = "2016-01 through 2016-12"

    with open(args.output / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save supporting tables
    pd.DataFrame(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ],
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    ).to_csv(args.output / "confusion_matrix.csv")

    calibration.to_csv(
        args.output / "calibration_deciles.csv", index=False
    )
    risk.to_csv(
        args.output / "risk_deciles.csv", index=False
    )
    monthly.to_csv(
        args.output / "monthly_stability.csv", index=False
    )

    # Save loan-level results without modifying original data.
    scored = pd.DataFrame({
        "issue_date": pd.to_datetime(dates).to_numpy(),
        "target": y,
        "default_risk_score": p,
        "threshold": threshold,
        "predicted_default": predicted,
    })

    scored.to_csv(
        args.output / "loan_predictions.csv",
        index=False,
    )

    save_plots(
        y, p, threshold, calibration, risk, monthly, args.output
    )

    print("\nFINAL 2016 EVALUATION")
    print("-" * 35)

    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]:
        print(f"{key:22s}: {metrics[key]:.5f}")

    print("\nConfusion matrix:")
    print(
        confusion_matrix(y, predicted, labels=[0, 1])
    )

    print("\nTop 10% risk lift:",
          round(float(risk.iloc[0]["lift"]), 4))

    print(
        "Defaults captured in top 10%:",
        round(float(risk.iloc[0]["defaults_captured_pct"]), 2),
        "%",
    )

    print("\nReports saved to:", args.output)
    print("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
