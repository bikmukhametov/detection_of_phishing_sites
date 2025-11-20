"""
Запуск полного пайплайна анализа кластеров и визуализаций для набора new_dataset1.csv,
без повторного выполнения алгоритмов отбора признаков.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.data_loader import load_dataset
from src.preprocess import preprocess_features
from src.clustering import run_kmeans_set, run_birch_set
from src.evaluation import build_cluster_summary_tables
from src.plots import (
    plot_low_cardinality_proportions_paged,
    plot_clusters_pca,
    plot_cluster_means_with_phishing,
    plot_chi2_feature_importance,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main():
    data_path = Path("new_dataset1.csv")
    label_name = "Result"
    out_root = Path("outputs") / "selected_pipeline"
    ensure_dir(out_root)

    df, feature_df, label_series = load_dataset(data_path, label_col=label_name)
    X_scaled, preprocess_info = preprocess_features(feature_df)

    chi2_out = out_root / "chi2_feature_importance.png"
    plot_chi2_feature_importance(
        feature_df=feature_df,
        label_series=label_series,
        out_file=chi2_out,
    )

    k_values = [2, 3, 4, 5, 6, 7, 8, 9]
    kmeans_runs = run_kmeans_set(X_scaled, k_values)
    birch_runs = run_birch_set(X_scaled, k_values)

    all_runs = {}
    all_runs.update({f"kmeans_k={k}": res for k, res in kmeans_runs.items()})
    all_runs.update({f"birch_k={k}": res for k, res in birch_runs.items()})

    phishing_positive_value = -1

    for run_name, run_info in all_runs.items():
        safe_run_name = run_name.replace("=", "_")
        run_dir = out_root / safe_run_name
        ensure_dir(run_dir)

        clusters = run_info["labels"]

        summary_df = build_cluster_summary_tables(
            df_index=feature_df.index,
            clusters=clusters,
            label_series=label_series,
            phishing_positive_value=phishing_positive_value,
        )
        summary_df.to_csv(run_dir / "cluster_summary.csv", index=False, encoding="utf-8")
        summary_df.to_markdown(run_dir / "cluster_summary.md", index=False)

        plot_low_cardinality_proportions_paged(
            df_features=feature_df,
            clusters=clusters,
            title_prefix=f"{run_name} — доли признаков",
            out_dir=run_dir / "plots_low_card_props",
            features_per_page=9,
            max_unique=5,
        )

        plot_cluster_means_with_phishing(
            X_scaled=X_scaled,
            feature_names=list(feature_df.columns),
            clusters=clusters,
            label_series=label_series,
            df_index=feature_df.index,
            phishing_positive_value=phishing_positive_value,
            title_prefix=f"{run_name} — средние значения кластеров",
            out_dir=run_dir / "plots_cluster_means",
        )

        model = run_info["model"]
        cluster_centers = None
        if hasattr(model, "cluster_centers_"):
            unique_clusters = sorted(np.unique(clusters[clusters >= 0]))
            centers_list = []
            for cl in unique_clusters:
                if cl < len(model.cluster_centers_):
                    centers_list.append(model.cluster_centers_[cl])
                else:
                    centers_list.append(X_scaled[clusters == cl].mean(axis=0))
            cluster_centers = np.array(centers_list)
        else:
            unique_clusters = sorted(np.unique(clusters[clusters >= 0]))
            cluster_centers = np.array([
                X_scaled[clusters == cl].mean(axis=0) for cl in unique_clusters
            ])

        k = len(np.unique(clusters[clusters >= 0]))
        plot_clusters_pca(
            X_scaled=X_scaled,
            clusters=clusters,
            label_series=label_series,
            df_index=feature_df.index,
            phishing_positive_value=phishing_positive_value,
            cluster_centers=cluster_centers,
            title=f"{run_name} — Визуализация PCA (k={k})",
            out_file=run_dir / "pca_visualization.png",
        )

    print(f"Selected pipeline results saved to: {os.path.abspath(out_root)}")


if __name__ == "__main__":
    main()

