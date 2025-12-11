"""
Cluster-based Bagging: train on clear clusters, test on ambiguous ones.

This script:
1. Loads clustering results from kmeans_k=7 and kmeans_k=8
2. Identifies "clear" clusters (phishing/legitimate ratio > 80% or < 20%)
3. Identifies "ambiguous" clusters (20% - 80% phishing)
4. Trains bagging models on clear clusters, tests on ambiguous clusters

Usage: python cluster_based_bagging_run.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import pickle

from src.data_loader import load_dataset
from src.preprocess import preprocess_features
from src.bagging import CustomBaggingClassifier
 


def metrics_dict(y_true, y_pred, y_proba):
    """Calculate metrics, handling binary and multiclass labels."""
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


def load_cluster_assignments(k):
    """Load cluster assignments from saved pickle file."""
    # Prefer precomputed labels if available (older outputs)
    pkl_path = Path('outputs') / f'kmeans_k_{k}' / 'cluster_labels.pkl'
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    # Otherwise, will return None and caller may compute labels on the fly
    return None


def read_cluster_summary_selected(k):
    """Read cluster_summary.csv from selected_pipeline for k and return phishing ratios dict."""
    csv_path = Path('outputs') / 'selected_pipeline' / f'kmeans_k_{k}' / 'cluster_summary.csv'
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        # Expect columns: кластер, ..., процент фишинга в кластере
        ratio_col = [c for c in df.columns if 'фишинга' in c][0]
        cluster_col = df.columns[0]
        ratios = {int(row[cluster_col]): float(row[ratio_col]) / 100.0 for _, row in df.iterrows() if row[cluster_col] != 'ИТОГО'}
        return ratios
    except Exception:
        return None


def get_cluster_stats(labels, cluster_ids, y):
    """Calculate phishing ratio per cluster."""
    stats = {}
    unique_labels = np.unique(y)
    label_map = {lab: idx for idx, lab in enumerate(unique_labels)}
    y_mapped = np.array([label_map[v] for v in y])
    
    for cluster_id in np.unique(cluster_ids):
        mask = cluster_ids == cluster_id
        cluster_y = y_mapped[mask]
        phishing_count = np.sum(cluster_y == 1)  # assuming 1 is phishing
        total_count = len(cluster_y)
        phishing_ratio = phishing_count / total_count if total_count > 0 else 0
        stats[cluster_id] = {
            'total': total_count,
            'phishing_ratio': phishing_ratio,
            'phishing_count': phishing_count
        }
    return stats


def run():
    out = Path('outputs') / 'cluster_based_bagging'
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    df, features_df, labels = load_dataset('Phishing_Websites_Data.csv', label_col='Result')
    X, _ = preprocess_features(features_df)
    y = labels.values

    # Map labels to integer codes
    unique_labels = np.unique(y)
    label_map = {lab: idx for idx, lab in enumerate(unique_labels)}
    y_mapped = np.array([label_map[v] for v in y])

    # Load clustering results (k=7 and k=8)
    print("Loading cluster assignments...")
    clusters_k7 = load_cluster_assignments(7)
    clusters_k8 = load_cluster_assignments(8)

    if clusters_k7 is None or clusters_k8 is None:
        print("Clustering files not found. Running clustering...")
        # Run clustering if not found
        from sklearn.cluster import KMeans
        kmeans_7 = KMeans(n_clusters=7, random_state=42, n_init=10)
        kmeans_8 = KMeans(n_clusters=8, random_state=42, n_init=10)
        clusters_k7 = kmeans_7.fit_predict(X)
        clusters_k8 = kmeans_8.fit_predict(X)
        # Save for future use
        Path('outputs/kmeans_k_7').mkdir(parents=True, exist_ok=True)
        Path('outputs/kmeans_k_8').mkdir(parents=True, exist_ok=True)
        with open(Path('outputs/kmeans_k_7/cluster_labels.pkl'), 'wb') as f:
            pickle.dump(clusters_k7, f)
        with open(Path('outputs/kmeans_k_8/cluster_labels.pkl'), 'wb') as f:
            pickle.dump(clusters_k8, f)

    # Analyze clusters: find clear vs ambiguous
    print("\nAnalyzing cluster composition...")
    
    for k, clusters in [(7, clusters_k7), (8, clusters_k8)]:
        stats = get_cluster_stats(labels, clusters, y_mapped)

        # Try to read user-selected cluster summaries from selected_pipeline
        selected_ratios = read_cluster_summary_selected(k)

        clear_clusters = []
        ambiguous_clusters = []

        # iterate cluster ids present in the clustering
        for cluster_id in np.unique(clusters):
            # prefer ratio from selected_pipeline summary if available
            if selected_ratios and cluster_id in selected_ratios:
                ratio = selected_ratios[cluster_id]
            else:
                ratio = stats.get(cluster_id, {}).get('phishing_ratio', 0)

            if ratio >= 0.80 or ratio <= 0.20:
                clear_clusters.append(int(cluster_id))
            else:
                ambiguous_clusters.append(int(cluster_id))
        
        print(f"\nK={k}:")
        print(f"  Clear clusters (phishing ratio <= 0.2 or >= 0.8): {clear_clusters}")
        print(f"  Ambiguous clusters (0.2 < phishing ratio < 0.8): {ambiguous_clusters}")
        
        # Create train/test split based on clusters
        train_mask = np.isin(clusters, clear_clusters)
        test_mask = np.isin(clusters, ambiguous_clusters)
        
        X_train = X[train_mask]
        y_train = y_mapped[train_mask]
        X_test = X[test_mask]
        y_test = y_mapped[test_mask]
        
        print(f"  Training set size: {len(X_train)}")
        print(f"  Test set size: {len(X_test)}")
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  Skipping k={k} (not enough data)")
            continue

        # Helper function to run comparison
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

        # Save plots for this k value
        k_out = out / f'kmeans_k_{k}'
        k_out.mkdir(parents=True, exist_ok=True)

        # Metrics comparison plots
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
        plt.title(f'Bagging: сравнение метрик (k={k}, Логистическая регрессия)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(k_out / 'bagging_metrics_lr.png', dpi=150)
        plt.close()

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
        plt.title(f'Bagging: сравнение метрик (k={k}, Дерево решений)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(k_out / 'bagging_metrics_dt.png', dpi=150)
        plt.close()

        # Write markdown report
        with open(k_out / 'cluster_based_bagging_report.md', 'w', encoding='utf-8') as f:
            f.write(f'# Cluster-Based Bagging (K={k})\n\n')
            f.write('Отчёт о сравнении sklearn и custom Bagging на явных и неясных кластерах.\n\n')

            f.write('## Стратегия\n\n')
            f.write('- **Обучающая выборка**: кластеры, где преобладает один класс (фишинг >80% или <20%)\n')
            f.write(f'- **Тестовая выборка**: кластеры с "неясной" композицией (20%-80% фишинга)\n')
            f.write(f'- **Явные кластеры**: {clear_clusters}\n')
            f.write(f'- **Неясные кластеры**: {ambiguous_clusters}\n')
            f.write(f'- **Размер обучающей выборки**: {len(X_train)}\n')
            f.write(f'- **Размер тестовой выборки**: {len(X_test)}\n\n')

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

            f.write('\n## Анализ\n\n')
            f.write('Модель была обучена на "явных" кластерах и протестирована на "неясных" кластерах.\n')
            f.write('Это проверяет, насколько хорошо модель может классифицировать примеры на границе между фишингом и не-фишингом.\n\n')

            f.write('## Сохранённые файлы\n\n')
            f.write('- `bagging_metrics_lr.png` — сравнение метрик для логистической регрессии\n')
            f.write('- `bagging_metrics_dt.png` — сравнение метрик для дерева решений\n')
            f.write('- `cluster_based_bagging_report.md` — этот файл\n')

        print(f"  Results saved to {k_out}")

    print(f"\nCluster-based bagging analysis complete!")
    print(f"Results saved to {out}")


if __name__ == '__main__':
    run()
