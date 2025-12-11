"""
Run comparison of sklearn BaggingClassifier vs custom bagging on SELECTED FEATURES.

Produces outputs in `outputs/bagging_selected/`:
- `bagging_summary.md` — markdown report
- `bagging_metrics_lr.png` — bar chart of metrics for logistic regression
- `bagging_metrics_dt.png` — bar chart of metrics for decision tree
- `bagging_predictions_pca_lr.png` — PCA visualization of predictions (LR)
- `bagging_predictions_pca_dt.png` — PCA visualization of predictions (DT)

Usage: python bagging_selected_run.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
import seaborn as sns
from matplotlib.lines import Line2D

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


def plot_predictions_pca(X_test, y_test, y_pred_sklearn, y_pred_custom, estimator_name, out_file):
    """Plot predictions in 2D using PCA reduction, comparing sklearn vs custom bagging.
    
    Args:
        X_test: Test feature matrix
        y_test: True labels
        y_pred_sklearn: Predictions from sklearn BaggingClassifier
        y_pred_custom: Predictions from CustomBaggingClassifier
        estimator_name: Name of base estimator (e.g., "Логистическая регрессия")
        out_file: Path to save the plot
    """
    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_test)
    
    # Create dataframe for visualization
    plot_df = pd.DataFrame(X2, columns=['x', 'y'])
    plot_df['true_label'] = y_test
    plot_df['sklearn_pred'] = y_pred_sklearn
    plot_df['custom_pred'] = y_pred_custom
    plot_df['is_phishing_true'] = y_test == 1
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Color palette for classes
    palette = {0: '#1f77b4', 1: '#ff7f0e'}  # blue for legitimate, orange for phishing
    
    # Plot 1: sklearn predictions
    ax = axes[0]
    for pred_class in [0, 1]:
        class_data = plot_df[plot_df['sklearn_pred'] == pred_class]
        is_phishing = class_data['is_phishing_true']
        
        # Correct predictions (circles)
        correct = class_data[is_phishing == pred_class]
        ax.scatter(correct['x'], correct['y'], c=palette[pred_class], marker='o', s=50,
                  alpha=0.7, edgecolor='k', linewidth=0.5, label=f'Предсказание: {pred_class}')
        
        # Incorrect predictions (X markers)
        incorrect = class_data[is_phishing != pred_class]
        if len(incorrect) > 0:
            ax.scatter(incorrect['x'], incorrect['y'], c=palette[pred_class], marker='x', s=50,
                      alpha=0.9, linewidth=1.5)
    
    explained_var = pca.explained_variance_ratio_
    ax.set_xlabel(f'ГК1 ({explained_var[0]*100:.1f}%)')
    ax.set_ylabel(f'ГК2 ({explained_var[1]*100:.1f}%)')
    ax.set_title(f'sklearn Bagging — {estimator_name}')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: custom predictions
    ax = axes[1]
    for pred_class in [0, 1]:
        class_data = plot_df[plot_df['custom_pred'] == pred_class]
        is_phishing = class_data['is_phishing_true']
        
        # Correct predictions (circles)
        correct = class_data[is_phishing == pred_class]
        ax.scatter(correct['x'], correct['y'], c=palette[pred_class], marker='o', s=50,
                  alpha=0.7, edgecolor='k', linewidth=0.5, label=f'Предсказание: {pred_class}')
        
        # Incorrect predictions (X markers)
        incorrect = class_data[is_phishing != pred_class]
        if len(incorrect) > 0:
            ax.scatter(incorrect['x'], incorrect['y'], c=palette[pred_class], marker='x', s=50,
                      alpha=0.9, linewidth=1.5)
    
    ax.set_xlabel(f'ГК1 ({explained_var[0]*100:.1f}%)')
    ax.set_ylabel(f'ГК2 ({explained_var[1]*100:.1f}%)')
    ax.set_title(f'Custom Bagging — {estimator_name}')
    ax.grid(True, alpha=0.3)
    
    # Create custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, 
               markeredgecolor='k', markeredgewidth=0.5, label='Класс 0 (Легит.)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=10,
               markeredgecolor='k', markeredgewidth=0.5, label='Класс 1 (Фишинг)'),
        Line2D([0], [0], marker='x', color='gray', markersize=10, markeredgewidth=1.5, 
               label='Ошибка предсказания'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    fig.suptitle(f'PCA визуализация предсказаний Bagging', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()


def run():
    out = Path('outputs') / 'bagging_selected'
    out.mkdir(parents=True, exist_ok=True)

    # Load and preprocess
    df, features_df, labels = load_dataset('new_dataset1.csv', label_col='Result')
    X, _ = preprocess_features(features_df)
    y = labels.values

    # Map labels to integer codes to avoid dtype/label mismatches
    unique_labels = np.unique(y)
    label_map = {lab: idx for idx, lab in enumerate(unique_labels)}
    y = np.array([label_map[v] for v in y])

    # Use a train/test split for speed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Helper function to run both bagging variants
    def run_comparison(base_estimator):
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
            max_features=0.8,
            bootstrap=True,
            bootstrap_features=False,
            random_state=42,
        )
        custom.fit(X_train, y_train)
        custom_proba = custom.predict_proba(X_test)[:, 1]
        custom_pred = custom.predict(X_test)
        custom_metrics = metrics_dict(y_test, custom_pred, custom_proba)

        return sklearn_metrics, custom_metrics, sklearn_pred, custom_pred

    # Run with LogisticRegression
    sklearn_metrics_lr, custom_metrics_lr, y_pred_sklearn_lr, y_pred_custom_lr = run_comparison(
        LogisticRegression(max_iter=1000)
    )

    # Run with DecisionTree
    sklearn_metrics_dt, custom_metrics_dt, y_pred_sklearn_dt, y_pred_custom_dt = run_comparison(
        DecisionTreeClassifier(max_depth=5)
    )

    # Save PCA visualizations
    plot_predictions_pca(X_test, y_test, y_pred_sklearn_lr, y_pred_custom_lr,
                         'Логистическая регрессия', out / 'bagging_predictions_pca_lr.png')
    plot_predictions_pca(X_test, y_test, y_pred_sklearn_dt, y_pred_custom_dt,
                         'Дерево решений', out / 'bagging_predictions_pca_dt.png')

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
    plt.title('Bagging на отобранных признаках: сравнение метрик (Логистическая регрессия)')
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
    plt.title('Bagging на отобранных признаках: сравнение метрик (Дерево решений)')
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
