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
from src.feature_selection import (
    add_del_algorithm,
    genetic_algorithm,
    stochastic_search_with_adaptation,
    evaluate_feature_set,
)
from src.feature_analysis import (
    save_feature_selection_results,
    plot_algorithm_comparison,
    plot_quality_vs_feature_count,
    plot_convergence_curves,
    plot_feature_count_progression,
    create_summary_markdown,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_feature_selection_analysis(X_scaled, feature_df, label_series, output_dir: Path):
    """
    Run all feature selection algorithms and generate analysis.
    
    Args:
        X_scaled: Scaled feature matrix
        feature_df: Original feature DataFrame
        label_series: Target labels
        output_dir: Output directory for results
    """
    ensure_dir(output_dir)
    
    feature_names = list(feature_df.columns)
    y = label_series.values
    
    print("\n" + "="*60)
    print("АНАЛИЗ ОТБОРА ПРИЗНАКОВ")
    print("="*60)
    
    # Collect results from all algorithms
    results = {}
    
    # 1. Add-Del Algorithm
    print("\nЗапуск алгоритма Add-Del...")
    try:
        selected_indices_addel, stats_addel = add_del_algorithm(
            X_scaled, y, feature_names,
            max_iterations=25,
            patience=10,
            test_size=0.2
        )
        
        selected_names_addel = [feature_names[i] for i in selected_indices_addel]
        metrics_addel = evaluate_feature_set(X_scaled, y, selected_indices_addel)
        
        results['Add-Del'] = {
            'selected_features': selected_indices_addel,
            'selected_feature_names': selected_names_addel,
            'metrics': metrics_addel,
            'stats': stats_addel,
        }
        print(f"Add-Del: выбрано {len(selected_indices_addel)} признаков, F1={metrics_addel['f1']:.4f}")
    except Exception as e:
        print(f"Ошибка в Add-Del: {e}")
    
    # 2. Genetic Algorithm
    print("\nЗапуск генетического алгоритма...")
    try:
        selected_indices_ga, stats_ga = genetic_algorithm(
            X_scaled, y, feature_names,
            population_size=40,
            generations=30,
            mutation_rate=0.12,
            crossover_prob=0.85,
            patience=5,
            test_size=0.2
        )
        
        selected_names_ga = [feature_names[i] for i in selected_indices_ga]
        metrics_ga = evaluate_feature_set(X_scaled, y, selected_indices_ga)
        
        results['Genetic Algorithm'] = {
            'selected_features': selected_indices_ga,
            'selected_feature_names': selected_names_ga,
            'metrics': metrics_ga,
            'stats': stats_ga,
        }
        print(f"Genetic Algorithm: выбрано {len(selected_indices_ga)} признаков, F1={metrics_ga['f1']:.4f}")
    except Exception as e:
        print(f"Ошибка в Genetic Algorithm: {e}")
    
    # 3. Stochastic Search with Adaptation (SPA)
    print("\nЗапуск стохастического поиска с адаптацией (СПА)...")
    try:
        selected_indices_spa, stats_spa = stochastic_search_with_adaptation(
            X_scaled, y, feature_names,
            j0=1,
            T=15,
            r=20,
            h=0.03,
            d=5,
            test_size=0.2,
        )

        selected_names_spa = [feature_names[i] for i in selected_indices_spa]
        metrics_spa = evaluate_feature_set(X_scaled, y, selected_indices_spa)

        results['Stochastic Search (SPA)'] = {
            'selected_features': selected_indices_spa,
            'selected_feature_names': selected_names_spa,
            'metrics': metrics_spa,
            'stats': stats_spa,
        }
        print(f"SPA: выбрано {len(selected_indices_spa)} признаков, F1={metrics_spa['f1']:.4f}")
    except Exception as e:
        print(f"Ошибка в SPA: {e}")
    
    if not results:
        print("\n⚠ Не удалось запустить ни один алгоритм отбора признаков!")
        return
    
    # Save and visualize results
    print("\nСохранение результатов...")
    save_feature_selection_results(results, output_dir)
    
    print("Создание графиков сравнения...")
    plot_algorithm_comparison(results, output_dir)
    plot_quality_vs_feature_count(results, output_dir)
    plot_convergence_curves(results, output_dir)
    plot_feature_count_progression(results, output_dir)
    
    print("Создание специфичных графиков для алгоритмов...")
    from src.feature_analysis import plot_algorithm_specific_visualizations
    plot_algorithm_specific_visualizations(results, output_dir)
    
    create_summary_markdown(results, output_dir, feature_names)
    
    print(f"Результаты отбора признаков сохранены в: {output_dir}")
    print("\n" + "="*60)


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
    k_values = [2, 3, 4, 5, 6, 7, 8, 9]
    kmeans_runs = run_kmeans_set(X_scaled, k_values)
    birch_runs = run_birch_set(X_scaled, k_values)

    # Collect all runs for uniform processing
    all_runs = {}
    all_runs.update({f"kmeans_k={k}": res for k, res in kmeans_runs.items()})
    all_runs.update({f"birch_k={k}": res for k, res in birch_runs.items()})

    phishing_positive_value = -1  # -1 — фишинг, 1 — законный

    # Run feature selection analysis
    feature_selection_out = out_root / "feature_selection"
    run_feature_selection_analysis(X_scaled, feature_df, label_series, feature_selection_out)

    # For each run: save tables and plots
    for run_name, run_info in all_runs.items():
        # Replace "=" with "_" in run_name to avoid Windows file path issues
        safe_run_name = run_name.replace("=", "_")
        run_dir = out_root / safe_run_name
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


