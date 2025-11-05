from pathlib import Path
from typing import Tuple

import pandas as pd


def load_dataset(csv_path: Path, label_col: str = "Result") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the phishing dataset, separate features and label.

    If label_col is missing, try to auto-detect common target names.
    """
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        candidates = ["Result", "Label", "target", "Target", "class", "Class"]
        auto = next((c for c in candidates if c in df.columns), None)
        if auto is None:
            raise ValueError(
                f"Label column '{label_col}' not found and auto-detection failed. Available: {list(df.columns)}"
            )
        label_col = auto

    label_series = df[label_col]
    feature_df = df.drop(columns=[label_col])
    return df, feature_df, label_series


