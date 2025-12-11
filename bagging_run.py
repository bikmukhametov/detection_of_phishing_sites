"""
Run comparison of sklearn BaggingClassifier vs custom bagging.

Produces outputs in `outputs/bagging/`:
- `bagging_summary.md` — markdown report
- `bagging_metrics.png` — bar chart of metrics
- `bagging_roc.png` — ROC curves comparison

Usage: python bagging_run.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.data_loader import load_dataset
from src.preprocess import preprocess_features
from src.bagging import CustomBaggingClassifier


def metrics_dict(y_true, y_pred, y_proba):
    # Handle binary vs multiclass labels
    unique = np.unique(y_true)
    binary = len(unique) == 2
    if binary:
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            roc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            roc = float('nan')
    else:
        # multiclass: use macro averages; ROC-AUC not computed here
        precision = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        recall = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        roc = float('nan')

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc,
    }


def run():
    out = Path('outputs') / 'bagging'
    out.mkdir(parents=True, exist_ok=True)

    # Load and preprocess
    df, features_df, labels = load_dataset('Phishing_Websites_Data.csv', label_col='Result')
    X, _ = preprocess_features(features_df)
    y = labels.values

    # Map labels to integer codes to avoid dtype/label mismatches
    unique_labels = np.unique(y)
    label_map = {lab: idx for idx, lab in enumerate(unique_labels)}
    y = np.array([label_map[v] for v in y])

    # Use a train/test split for speed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Helper function to run both bagging variants
    def run_comparison(base_estimator, estimator_name):
        """Run bagging comparison for given base estimator."""
        sklearn_bag = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=30,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            bootstrap_features=False,
            random_state=42,
        )
        sklearn_bag.fit(X_train, y_train)
        sklearn_proba = sklearn_bag.predict_proba(X_test)[:, 1]
        sklearn_pred = sklearn_bag.predict(X_test)
        sklearn_metrics = metrics_dict(y_test, sklearn_pred, sklearn_proba)

        custom = CustomBaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=30,
            max_samples=0.8,
            max_features=0.6,
            bootstrap=True,
            bootstrap_features=False,
            random_state=42,
        )
        custom.fit(X_train, y_train)
        custom_proba = custom.predict_proba(X_test)[:, 1]
        custom_pred = custom.predict(X_test)
        custom_metrics = metrics_dict(y_test, custom_pred, custom_proba)

        return sklearn_metrics, custom_metrics

    # Run with LogisticRegression
    sklearn_metrics_lr, custom_metrics_lr = run_comparison(
        LogisticRegression(max_iter=1000)
    )

    # Run with DecisionTree
    sklearn_metrics_dt, custom_metrics_dt = run_comparison(
        DecisionTreeClassifier(max_depth=5)
    )

    # Save comparison plots for LR
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    sklearn_vals_lr = [sklearn_metrics_lr[m] for m in metrics_names]
    custom_vals_lr = [custom_metrics_lr[m] for m in metrics_names]

    plt.figure(figsize=(10, 5))
    x = np.arange(len(metrics_names))
    width = 0.35
    plt.bar(x - width/2, sklearn_vals_lr, width, label='sklearn Bagging')
    plt.bar(x + width/2, custom_vals_lr, width, label='Custom Bagging')
    plt.xticks(x, metrics_names)
    plt.ylim(0, 1)
    plt.ylabel('Значение')
    plt.title('Bagging: сравнение метрик (Логистическая регрессия)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / 'bagging_metrics_lr.png', dpi=150)
    plt.close()

    # Save comparison plots for DT
    sklearn_vals_dt = [sklearn_metrics_dt[m] for m in metrics_names]
    custom_vals_dt = [custom_metrics_dt[m] for m in metrics_names]

    plt.figure(figsize=(10, 5))
    x = np.arange(len(metrics_names))
    width = 0.35
    plt.bar(x - width/2, sklearn_vals_dt, width, label='sklearn Bagging')
    plt.bar(x + width/2, custom_vals_dt, width, label='Custom Bagging')
    plt.xticks(x, metrics_names)
    plt.ylim(0, 1)
    plt.ylabel('Значение')
    plt.title('Bagging: сравнение метрик (Дерево решений)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / 'bagging_metrics_dt.png', dpi=150)
    plt.close()

    # Write markdown report on Russian
    with open(out / 'bagging_summary.md', 'w', encoding='utf-8') as f:
        f.write('# Сравнение Bagging\n\n')
        f.write('В этом отчёте сравниваются две пользовательские реализации базовых оценщиков: логистическая регрессия и дерево решений с использованием sklearn.BaggingClassifier и CustomBaggingClassifier.\n\n')

        f.write('## Результаты: Логистическая регрессия\n\n')
        f.write('| Реализация | Accuracy | Precision | Recall | F1 | ROC-AUC |\n')
        f.write('|---|---:|---:|---:|---:|---:|\n')
        f.write(f'| sklearn Bagging | {sklearn_metrics_lr["accuracy"]:.4f} | {sklearn_metrics_lr["precision"]:.4f} | {sklearn_metrics_lr["recall"]:.4f} | {sklearn_metrics_lr["f1"]:.4f} | {sklearn_metrics_lr["roc_auc"]:.4f} |\n')
        f.write(f'| Custom Bagging | {custom_metrics_lr["accuracy"]:.4f} | {custom_metrics_lr["precision"]:.4f} | {custom_metrics_lr["recall"]:.4f} | {custom_metrics_lr["f1"]:.4f} | {custom_metrics_lr["roc_auc"]:.4f} |\n')

        f.write('\n## Результаты: Дерево решений\n\n')
        f.write('| Реализация | Accuracy | Precision | Recall | F1 | ROC-AUC |\n')
        f.write('|---|---:|---:|---:|---:|---:|\n')
        f.write(f'| sklearn Bagging | {sklearn_metrics_dt["accuracy"]:.4f} | {sklearn_metrics_dt["precision"]:.4f} | {sklearn_metrics_dt["recall"]:.4f} | {sklearn_metrics_dt["f1"]:.4f} | {sklearn_metrics_dt["roc_auc"]:.4f} |\n')
        f.write(f'| Custom Bagging | {custom_metrics_dt["accuracy"]:.4f} | {custom_metrics_dt["precision"]:.4f} | {custom_metrics_dt["recall"]:.4f} | {custom_metrics_dt["f1"]:.4f} | {custom_metrics_dt["roc_auc"]:.4f} |\n')

        f.write('\n## Как работает sklearn.BaggingClassifier\n\n')
        f.write('`BaggingClassifier` из scikit-learn строит ансамбль базовых оценщиков, обучая каждый оценщик на случайно выбранной подвыборке объектов (и при необходимости — подвыборке признаков). Итоговое предсказание получается агрегированием прогнозов от всех оценщиков: усреднением вероятностей или большинством голосов.\n\n')

        f.write('Ключевые атрибуты:\n')
        f.write('- `estimators_` — список обученных оценщиков\n')
        f.write('- `estimators_features_` — индексы признаков, использованные каждым оценщиком (если `max_features` < 1)\n')
        f.write('- `oob_score_` — оценка вне выборки (если включено)\n\n')

        f.write('## Как работает пользовательский CustomBaggingClassifier\n\n')
        f.write('Пользовательская реализация выполняет следующие шаги:\n\n')
        f.write('- Для каждого из `n_estimators` создаётся случайная подвыборка образцов (по параметру `max_samples`) и подвыборка признаков (по параметру `max_features`).\n')
        f.write('- Клонируется переданный `base_estimator` и обучается на этой подвыборке.\n')
        f.write('- Сохраняется обученный оценщик и соответствующие индексы признаков.\n')
        f.write('- При предсказании усредняются предсказанные вероятности положительного класса от всех оценщиков; итоговое бинарное решение получается применением порога 0.5.\n\n')

        f.write('## Пример использования\n\n')
        f.write('```python\n')
        f.write('from src.bagging import CustomBaggingClassifier\n')
        f.write('clf = CustomBaggingClassifier(n_estimators=30, max_samples=0.8, max_features=0.6)\n')
        f.write('clf.fit(X_train, y_train)\n')
        f.write('pred = clf.predict(X_test)\n')
        f.write('proba = clf.predict_proba(X_test)[:, 1]\n')
        f.write('```\n\n')

        f.write('## Параметры и их влияние\n\n')
        f.write('- `n_estimators`: число базовых оценщиков; увеличение повышает стабильность, но увеличивает время обучения.\n')
        f.write('- `max_samples`: доля обучающих объектов, используемая для каждого оценщика; меньшие значения увеличивают разнообразие ансамбля.\n')
        f.write('- `max_features`: доля признаков, используемая для каждого оценщика; управляет разнообразием на уровне признаков.\n')
        f.write('- `bootstrap`: использовать ли выборку с возвращением для объектов.\n')
        f.write('- `bootstrap_features`: использовать ли выборку с возвращением для признаков.\n\n')

        f.write('## Рекомендации\n\n')
        f.write('- Для быстрых экспериментов рекомендуются `n_estimators=20..50`.\n')
        f.write('- Если в данных много признаков, снижение `max_features` (например, до 0.5) может повысить разнообразие и устойчивость модели.\n')
        f.write('- Для оценки можно включить `oob_score` в `sklearn.BaggingClassifier` и сравнить результаты с кросс-валидацией.\n\n')

        f.write('## Сохранённые файлы\n\n')
        f.write('- `bagging_metrics_lr.png` — столбчатая диаграмма сравнения метрик для логистической регрессии\n')
        f.write('- `bagging_metrics_dt.png` — столбчатая диаграмма сравнения метрик для дерева решений\n')
        f.write('- `bagging_summary.md` — этот файл\n')

    print('Bagging comparison saved to', out)


if __name__ == '__main__':
    run()
