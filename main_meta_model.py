"""
Мета-модель для комбинирования предсказаний базовых алгоритмов (Логистическая регрессия,
Дерево решений, SVM). Используются подготовленные таблицы
`predictions_all_features.md` и `predictions_selected_features.md`.

Шаги:
1. Загружаем таблицу, в которой есть столбцы предсказаний базовых моделей и `Result`.
2. Обучаем LogisticRegression (как простую мета-модель) на этих признаках.
3. Считаем метрики Accuracy, Precision, Recall, F1, ROC-AUC на тестовой части.
4. Сохраняем таблицу метрик (*.md) в `outputs/ml_evaluation`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


DEFAULT_PREDICTIONS_ALL = Path("outputs") / "ml_evaluation" / "predictions_all_features.md"
DEFAULT_PREDICTIONS_SELECTED = Path("outputs") / "ml_evaluation" / "predictions_selected_features.md"
DEFAULT_OUTPUT_DIR = Path("outputs") / "ml_evaluation"
LABEL_COLUMN = "Result"
POSITIVE_LABEL = -1
METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


def _read_markdown_table(path: Path) -> pd.DataFrame:
    """
    Парсинг markdown-таблицы, созданной через DataFrame.to_markdown.
    """
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|")]
    if not table_lines:
        raise ValueError(f"В файле {path} отсутствует markdown-таблица.")

    from io import StringIO

    buffer = "\n".join(table_lines)
    df = pd.read_csv(StringIO(buffer), sep="|", engine="python")
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")], errors="ignore")
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
    df = df.rename(columns=lambda c: c.strip())
    return df


def load_predictions(path: Path) -> pd.DataFrame:
    df = _read_markdown_table(path)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Не найден столбец {LABEL_COLUMN} в {path}")
    feature_cols = [col for col in df.columns if col != LABEL_COLUMN]
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().reset_index(drop=True)


def evaluate_meta_model(
    df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    feature_cols = [col for col in df.columns if col != LABEL_COLUMN]
    X = df[feature_cols]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    weights = pipeline.named_steps["model"].coef_.flatten()
    total_weight = weights.sum() if weights.sum() != 0 else 1.0
    normalized_weights = weights / total_weight
    weight_series = pd.Series(normalized_weights, index=feature_cols, name="Нормированный вес")
    weight_df = weight_series.reset_index().rename(columns={"index": "Алгоритм"})

    y_pred = pipeline.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL),
        "Recall": recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL),
        "F1": f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL),
    }

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics["ROC-AUC"] = roc_auc_score(
            y_test.replace({-1: 1, 1: 0}),
            y_proba,
        )
    else:
        metrics["ROC-AUC"] = float("nan")

    metrics_df = pd.DataFrame([metrics], index=["Мета-модель"])
    metrics_df = metrics_df[METRIC_COLUMNS]
    return metrics_df, metrics, weight_df


def save_markdown(df: pd.DataFrame, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{name}.md"
    df.to_markdown(md_path, index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Комбинирование базовых моделей через мета-классификатор."
    )
    parser.add_argument("--predictions-all", type=Path, default=DEFAULT_PREDICTIONS_ALL)
    parser.add_argument("--predictions-selected", type=Path, default=DEFAULT_PREDICTIONS_SELECTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    datasets = {
        "meta_all_features": args.predictions_all,
        "meta_selected_features": args.predictions_selected,
    }

    for name, path in datasets.items():
        if not path.exists():
            raise FileNotFoundError(f"Файл {path} не найден")
        df = load_predictions(path)
        metrics_df, metrics, weight_df = evaluate_meta_model(
            df,
            random_state=args.random_state,
            test_size=args.test_size,
        )
        print(f"\n=== {name} ===")
        print(metrics_df.to_markdown())
        save_markdown(metrics_df, name, args.output_dir)
        save_markdown(weight_df, f"{name}_weights", args.output_dir)


if __name__ == "__main__":
    main()

