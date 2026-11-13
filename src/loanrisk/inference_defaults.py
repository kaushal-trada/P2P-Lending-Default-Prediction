
import pandas as pd


def calculate_training_defaults(X_train, numeric_features):
    """Calculate training-only defaults for optional simplified inference."""

    defaults = {}
    numeric_columns = set(numeric_features) | {"int_rate"}

    for column in X_train.columns:
        values = X_train[column]

        if column in numeric_columns:
            clean = (
                values.astype("string")
                .str.replace("%", "", regex=False)
                .str.strip()
            )

            numbers = pd.to_numeric(clean, errors="coerce")
            median = numbers.median()

            defaults[column] = (
                float(median) if pd.notna(median) else None
            )

        else:
            modes = values.dropna().astype(str).mode()

            defaults[column] = (
                str(modes.iloc[0]) if not modes.empty else None
            )

    return defaults
