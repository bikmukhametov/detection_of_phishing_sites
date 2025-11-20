"""
Скрипт строит таблицу совпадений отобранных признаков между алгоритмами из
`outputs/feature_selection/feature_selection_summary.md`.

По строкам — признаки (по-русски, как в отчёте), по столбцам — алгоритмы.
Если алгоритм содержит признак, ставим «+». Дополнительно считаем, в скольких
алгоритмах встречается признак.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


DEFAULT_MD_PATH = Path("outputs") / "feature_selection" / "feature_selection_summary.md"
DEFAULT_OUTPUT_DIR = Path("outputs") / "feature_selection"
ALGORITHM_SECTION_PREFIX = "## "


def parse_feature_lists(md_path: Path) -> Dict[str, List[str]]:
    """
    Возвращает словарь {algorithm_name: [feature1, feature2, ...]}.
    Ожидается, что признаки перечислены как нумерованный список вида `1. Признак`.
    """
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    features_by_algo: Dict[str, List[str]] = {}
    current_algo = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(ALGORITHM_SECTION_PREFIX) and len(stripped) > 3:
            current_algo = stripped[len(ALGORITHM_SECTION_PREFIX) :].strip()
            features_by_algo.setdefault(current_algo, [])
            continue

        if current_algo is None:
            continue

        if stripped.startswith("---"):
            current_algo = None
            continue

        if stripped and stripped[0].isdigit() and ". " in stripped:
            # Формат "1. Признак"
            parts = stripped.split(". ", 1)
            if len(parts) == 2:
                feature = parts[1].strip()
                if feature:
                    features_by_algo[current_algo].append(feature)

    # Фильтруем пустые
    return {k: v for k, v in features_by_algo.items() if v}


def build_overlap_table(features_by_algo: Dict[str, List[str]]) -> pd.DataFrame:
    """Строит таблицу совпадений и считает количество алгоритмов per feature."""
    algorithms = list(features_by_algo.keys())
    feature_union: Set[str] = set()
    for feats in features_by_algo.values():
        feature_union.update(feats)

    rows = []
    for feature in sorted(feature_union):
        row = {"Признак": feature}
        count = 0
        for algo in algorithms:
            has_feature = feature in features_by_algo.get(algo, [])
            row[algo] = "+" if has_feature else ""
            if has_feature:
                count += 1
        row["Количество алгоритмов"] = count
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Количество алгоритмов", "Признак"], ascending=[False, True])
    return df[["Признак", *algorithms, "Количество алгоритмов"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Строит таблицу совпадений признаков из feature_selection_summary.md."
    )
    parser.add_argument(
        "--md-path",
        type=Path,
        default=DEFAULT_MD_PATH,
        help="Путь к markdown-файлу отчёта отбора признаков.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Куда сохранить таблицы (CSV + Markdown).",
    )
    args = parser.parse_args()

    if not args.md_path.exists():
        raise FileNotFoundError(f"Не найден файл: {args.md_path}")

    features_by_algo = parse_feature_lists(args.md_path)
    if not features_by_algo:
        raise RuntimeError("Не удалось извлечь признаки из отчёта.")

    table_df = build_overlap_table(features_by_algo)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "feature_overlap_table.csv"
    md_path = args.output_dir / "feature_overlap_table.md"

    table_df.to_csv(csv_path, index=False, encoding="utf-8")
    table_df.to_markdown(md_path, index=False)

    print(f"Сохранено:\n- {csv_path}\n- {md_path}")


if __name__ == "__main__":
    main()

