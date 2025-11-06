import os
from pathlib import Path

from src.data_loader import load_dataset
from src.preprocess import preprocess_features
from src.clustering import run_kmeans_set, run_birch_set, run_dbscan_set
from src.evaluation import build_cluster_summary_tables
from src.plots import (
    plot_cluster_feature_means_paged,
    plot_low_cardinality_proportions_paged,
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

    # Run clustering variants
    k_values = [2, 3, 4]
    kmeans_runs = run_kmeans_set(X_scaled, k_values)
    birch_runs = run_birch_set(X_scaled, k_values)
    dbscan_runs = run_dbscan_set(X_scaled, eps_list=[0.3, 0.5, 0.7], min_samples=5)

    # Collect all runs for uniform processing
    all_runs = {}
    all_runs.update({f"kmeans_k={k}": res for k, res in kmeans_runs.items()})
    all_runs.update({f"birch_k={k}": res for k, res in birch_runs.items()})
    for key, res in dbscan_runs.items():
        all_runs[f"dbscan_{key}"] = res

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

        # Plots — feature means on scaled data (all features)
        plot_cluster_feature_means_paged(
            X_scaled=X_scaled,
            feature_names=list(feature_df.columns),
            clusters=clusters,
            title_prefix=f"{run_name} — scaled feature means",
            out_dir=run_dir / "plots_means_scaled",
            features_per_page=15,
        )

        # Plots — proportions for low-cardinality features on raw data
        plot_low_cardinality_proportions_paged(
            df_features=feature_df,
            clusters=clusters,
            title_prefix=f"{run_name} — low-cardinality proportions",
            out_dir=run_dir / "plots_low_card_props",
            features_per_page=10,
            max_unique=5,
        )

    print(f"Done. Results saved to: {os.path.abspath(out_root)}")


if __name__ == "__main__":
    main()


