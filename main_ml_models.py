"""
Запуск машинного обучения на разных наборах признаков.

Наборы:
1. Полный — `Phishing_Websites_Data.csv`
2. Отобранный — `new_dataset1.csv`

Модели:
- Логистическая регрессия
- Дерево решений
- SVM (линейное ядро по умолчанию)

Результат: две таблицы (метрики Accuracy, Precision, Recall, F1, ROC-AUC),
сохраняемые в `outputs/ml_evaluation/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


DEFAULT_FULL_DATASET = Path("Phishing_Websites_Data.csv")
DEFAULT_SELECTED_DATASET = Path("new_dataset1.csv")
DEFAULT_OUTPUT_DIR = Path("outputs") / "ml_evaluation"
LABEL_COLUMN = "Result"
POSITIVE_LABEL = -1  # в исходном датасете -1 = фишинг
METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
MODEL_FACTORIES = {
    "Логистическая регрессия": lambda random_state: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    ),
    "Дерево решений": lambda random_state: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    ),
    "SVM": lambda random_state: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    ),
}


def load_feature_set(path: Path, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Колонка {label_col} не найдена в файле {path}")
    y = df[label_col]
    X = df.drop(columns=label_col)
    return X, y


def evaluate_dataset(
    dataset_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    test_size: float,
) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    rows = []
    prediction_columns: Dict[str, pd.Series] = {"Result": y_test}
    for model_name, factory in MODEL_FACTORIES.items():
        pipeline = factory(random_state)
        pipeline.fit(X_train, y_train)
        y_pred = pd.Series(pipeline.predict(X_test), index=y_test.index, name=model_name)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL),
            "Recall": recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL),
            "F1": f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL),
        }
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics["ROC-AUC"] = roc_auc_score(
                y_test.replace({-1: 1, 1: 0}), y_proba
            )
        else:
            metrics["ROC-AUC"] = float("nan")

        rows.append({"Модель": model_name, **metrics})
        prediction_columns[model_name] = y_pred

    df = pd.DataFrame(rows)
    metrics_df = df[["Модель", *METRIC_COLUMNS]]

    prediction_order = ["Логистическая регрессия", "Дерево решений", "SVM", "Result"]
    predictions_df = pd.DataFrame(prediction_columns)
    predictions_df = predictions_df[prediction_order]

    return metrics_df, predictions_df


def save_markdown(df: pd.DataFrame, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{name}.md"
    df.to_markdown(md_path, index=False)
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнение моделей на полном и отобранном наборах признаков."
    )
    parser.add_argument("--full-dataset", type=Path, default=DEFAULT_FULL_DATASET)
    parser.add_argument("--selected-dataset", type=Path, default=DEFAULT_SELECTED_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    datasets = {
        "all_features": (args.full_dataset, "Полный набор признаков"),
        "selected_features": (args.selected_dataset, "Отобранный набор (new_dataset1)"),
    }

    for file_key, (path, human_name) in datasets.items():
        if not path.exists():
            raise FileNotFoundError(f"Файл {path} не найден")
        X, y = load_feature_set(path, LABEL_COLUMN)
        print(f"\n=== {human_name} ===")
        metrics_table, predictions_table = evaluate_dataset(
            human_name, X, y, args.random_state, args.test_size
        )
        print(metrics_table.to_markdown(index=False))
        save_markdown(metrics_table, f"metrics_{file_key}", args.output_dir)
        save_markdown(
            predictions_table,
            f"predictions_{file_key}",
            args.output_dir,
        )


if __name__ == "__main__":
    main()

