from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class PreprocessInfo:
    numeric_columns: List[str]
    categorical_columns: List[str]
    label_encoders: Dict[str, LabelEncoder]
    scaler: StandardScaler


def preprocess_features(df_features: pd.DataFrame) -> Tuple[np.ndarray, PreprocessInfo]:
    """Fill missing values, encode categoricals, and standardize features.

    - Object dtype columns: LabelEncoder
    - Numeric dtype columns: fillna with median
    - Standardize all to zero mean and unit variance
    """
    df = df_features.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    label_encoders: Dict[str, LabelEncoder] = {}

    # Encode categoricals (if any)
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        numeric_cols.append(col)

    # Fill numeric NaNs with median
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols].values)

    info = PreprocessInfo(
        numeric_columns=numeric_cols,
        categorical_columns=cat_cols,
        label_encoders=label_encoders,
        scaler=scaler,
    )
    return X_scaled, info


