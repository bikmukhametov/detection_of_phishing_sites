import os
import warnings
from pathlib import Path
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

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
    # Fixed configuration per project requirements
    data_path = Path("Phishing_Websites_Data.csv")
    label_name = "Result"  # 1: законный, -1: фишинг
    out_root = Path("outputs")
    ensure_dir(out_root)

    df, feature_df, label_series = load_dataset(data_path, label_col=label_name)

    X_scaled, preprocess_info = preprocess_features(feature_df)

    # Plot chi-square feature importance (once, not per cluster)
    chi2_out = out_root / "chi2_feature_importance.png"
    plot_chi2_feature_importance(
        feature_df=feature_df,
        label_series=label_series,
        out_file=chi2_out,
    )

    # Run clustering variants
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    kmeans_runs = run_kmeans_set(X_scaled, k_values)
    birch_runs = run_birch_set(X_scaled, k_values)

    # Collect all runs for uniform processing
    all_runs = {}
    all_runs.update({f"kmeans_k={k}": res for k, res in kmeans_runs.items()})
    all_runs.update({f"birch_k={k}": res for k, res in birch_runs.items()})

    phishing_positive_value = -1  # -1 — фишинг, 1 — законный

    # For each run: save tables and plots
    for run_name, run_info in all_runs.items():
        run_dir = out_root / run_name
        ensure_dir(run_dir)

        clusters = run_info["labels"]

        # Tables
        summary_df = build_cluster_summary_tables(
            df_index=feature_df.index,
            clusters=clusters,
            label_series=label_series,
            phishing_positive_value=phishing_positive_value,
        )
        summary_csv = run_dir / "cluster_summary.csv"
        summary_md = run_dir / "cluster_summary.md"
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
        summary_df.to_markdown(summary_md, index=False)

        # Plots — proportions for low-cardinality features on raw data
        plot_low_cardinality_proportions_paged(
            df_features=feature_df,
            clusters=clusters,
            title_prefix=f"{run_name} — доли признаков",
            out_dir=run_dir / "plots_low_card_props",
            features_per_page=9,
            max_unique=5,
        )

        # Plots — cluster means with phishing overlay
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

        # Plots — PCA visualization for clusters
        model = run_info["model"]
        cluster_centers = None
        
        # Get cluster centers if available
        # For KMeans: cluster_centers_[i] corresponds to cluster i
        # For BIRCH: compute centers manually as mean of points in each cluster
        if hasattr(model, "cluster_centers_"):
            # KMeans: reorder centers to match sorted unique clusters
            unique_clusters = sorted(np.unique(clusters[clusters >= 0]))
            # KMeans centers are indexed by cluster label, so we can index directly
            # But we need to reorder them to match sorted unique_clusters
            centers_list = []
            for cl in unique_clusters:
                if cl < len(model.cluster_centers_):
                    centers_list.append(model.cluster_centers_[cl])
                else:
                    # Fallback: compute mean manually
                    centers_list.append(X_scaled[clusters == cl].mean(axis=0))
            cluster_centers = np.array(centers_list)
        else:
            # For BIRCH or other methods, compute cluster centers as mean of points in each cluster
            # Sort clusters to ensure consistent ordering
            unique_clusters = sorted(np.unique(clusters[clusters >= 0]))
            cluster_centers = np.array([
                X_scaled[clusters == cl].mean(axis=0) for cl in unique_clusters
            ])
        
        # Determine number of clusters for title
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

    print(f"Done. Results saved to: {os.path.abspath(out_root)}")


if __name__ == "__main__":
    main()


