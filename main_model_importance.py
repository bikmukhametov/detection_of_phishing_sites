"""
Оценка значимости моделей на основе их предсказаний.

Использует таблицы `predictions_all_features.md` и
`predictions_selected_features.md`, где столбцы соответствуют выходам моделей
и истинной метке `Result`. Для каждой таблицы обучаем RandomForestClassifier,
используя столбцы моделей как признаки, и оцениваем feature_importances_,
которые интерпретируются как значимость алгоритмов.

Результаты сохраняются в `outputs/ml_evaluation/model_importance_*.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


DEFAULT_PREDICTIONS_ALL = Path("outputs") / "ml_evaluation" / "predictions_all_features.md"
DEFAULT_PREDICTIONS_SELECTED = Path("outputs") / "ml_evaluation" / "predictions_selected_features.md"
DEFAULT_OUTPUT_DIR = Path("outputs") / "ml_evaluation"
LABEL_COLUMN = "Result"


def _read_markdown_table(path: Path) -> pd.DataFrame:
    """
    Простейший парсер markdown-таблиц, экспортированных методом DataFrame.to_markdown.
    Конвертирует в DataFrame, обрезая пустые колонки (первую/последнюю).
    """
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    table_lines: List[str] = [line for line in lines if line.startswith("|")]
    if not table_lines:
        raise ValueError(f"В файле {path} не обнаружена markdown-таблица")

    from io import StringIO

    buffer = "\n".join(table_lines)
    df = pd.read_csv(StringIO(buffer), sep="|", engine="python")
    # После split остаются пустые столбцы из-за крайних '|'
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")], errors="ignore")
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
    df = df.rename(columns=lambda c: c.strip())
    return df


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл {path}")
    df = _read_markdown_table(path)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Колонка {LABEL_COLUMN} отсутствует в {path}")
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    feature_cols = [col for col in df.columns if col != LABEL_COLUMN]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().reset_index(drop=True)


def compute_model_importance(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    feature_cols = [col for col in df.columns if col != LABEL_COLUMN]
    X = df[feature_cols]
    y = df[LABEL_COLUMN]

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X, y)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Алгоритм": feature_cols,
        "Значимость": importances,
    }).sort_values("Значимость", ascending=False).reset_index(drop=True)
    return importance_df


def save_table(df: pd.DataFrame, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{name}.md"
    df.to_markdown(md_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Оценка значимости алгоритмов на основе предсказаний."
    )
    parser.add_argument(
        "--predictions-all",
        type=Path,
        default=DEFAULT_PREDICTIONS_ALL,
        help="Путь к predictions_all_features.md",
    )
    parser.add_argument(
        "--predictions-selected",
        type=Path,
        default=DEFAULT_PREDICTIONS_SELECTED,
        help="Путь к predictions_selected_features.md",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Каталог для сохранения результатов",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    datasets = {
        "all_features": args.predictions_all,
        "selected_features": args.predictions_selected,
    }

    for key, path in datasets.items():
        df = load_predictions(path)
        importance_df = compute_model_importance(df, random_state=args.random_state)
        save_table(importance_df, f"model_importance_{key}", args.output_dir)
        print(f"Сохранено {key}:")
        print(importance_df.to_markdown(index=False))


if __name__ == "__main__":
    main()

