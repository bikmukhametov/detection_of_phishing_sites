"""
Отдельный скрипт для отбора признаков по весам модели.

Используем RandomForestClassifier из scikit-learn, чтобы получить веса (feature_importances_),
после чего выбираем топ-N признаков. Результат выводится в консоль и сохраняется
в CSV/Markdown в каталоге outputs/feature_weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.data_loader import load_dataset
from src.feature_labels import translate_feature_name

DEFAULT_DATA_PATH = Path("Phishing_Websites_Data.csv")
DEFAULT_OUTPUT_DIR = Path("outputs") / "feature_weights"


def build_feature_weight_model(random_state: int = 42) -> Pipeline:
    """Pipeline, который умеет обрабатывать пропуски и обучать модель с весами."""
    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("model", model),
        ]
    )
    return pipeline


def compute_feature_weights(
    features: pd.DataFrame,
    labels: pd.Series,
    random_state: int = 42,
) -> List[Tuple[str, float]]:
    """Возвращает список (feature_name, weight)."""
    pipeline = build_feature_weight_model(random_state=random_state)
    pipeline.fit(features, labels)
    model = pipeline.named_steps["model"]
    importances = model.feature_importances_
    return list(zip(features.columns, importances))


def save_results(
    sorted_weights: List[Tuple[str, float]],
    output_dir: Path,
    top_n: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "feature": feat,
                "feature_ru": translate_feature_name(feat),
                "weight": weight,
            }
            for feat, weight in sorted_weights
        ]
    )
    top_df = df.head(top_n)
    csv_path = output_dir / f"feature_weights_top_{top_n}.csv"
    md_path = output_dir / f"feature_weights_top_{top_n}.md"
    df_path = output_dir / "feature_weights_full.csv"

    df.to_csv(df_path, index=False, encoding="utf-8")
    top_df.to_csv(csv_path, index=False, encoding="utf-8")
    top_df.to_markdown(md_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Отбор признаков по весам RandomForestClassifier."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Путь к CSV с исходными данными (по умолчанию Phishing_Websites_Data.csv).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=36,
        help="Сколько признаков выводить в итоговом списке (10-15).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Фиксированное зерно для воспроизводимости.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Каталог для сохранения результатов.",
    )
    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError("Аргумент --top-n должен быть положительным.")

    df, feature_df, label_series = load_dataset(args.data_path, label_col="Result")
    weights = compute_feature_weights(feature_df, label_series, random_state=args.random_state)
    sorted_weights = sorted(weights, key=lambda x: x[1], reverse=True)

    top_slice = sorted_weights[: args.top_n]
    save_results(sorted_weights, args.output_dir, args.top_n)

    print("=" * 70)
    print(f"Топ-{args.top_n} признаков по весам RandomForestClassifier")
    print("=" * 70)
    for idx, (feat, weight) in enumerate(top_slice, start=1):
        print(f"{idx:>2}. {translate_feature_name(feat):<35} ({feat}) — {weight:.4f}")
    print("=" * 70)
    print(f"Полные результаты сохранены в: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

