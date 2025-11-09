from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _paginate(items: List[str], page_size: int) -> List[List[str]]:
    return [items[i : i + page_size] for i in range(0, len(items), page_size)]


def plot_low_cardinality_proportions_paged(
    df_features: pd.DataFrame,
    clusters: np.ndarray,
    title_prefix: str,
    out_dir: Path,
    features_per_page: int = 10,
    max_unique: int = 5,
) -> None:
    """For low-cardinality (categorical-like) features, plot per-cluster proportions of top values.

    - For each selected feature, compute distribution of its values within each cluster
    - Plot as grouped bars; paginate features across multiple figures
    """
    _ensure_dir(out_dir)

    df = df_features.copy()
    df["cluster"] = clusters

    # choose low-cardinality features
    low_card_features = [c for c in df_features.columns if df_features[c].nunique(dropna=False) <= max_unique]
    if not low_card_features:
        return

    feature_pages = _paginate(low_card_features, features_per_page)

    for page_idx, features in enumerate(feature_pages, start=1):
        # Build a tall table: (feature, cluster, category, proportion)
        all_rows = []
        for feat in features:
            # For stability, keep top up-to 5 categories by global frequency
            top_vals = df[feat].value_counts(dropna=False).index.tolist()[:5]
            # proportions by cluster
            for cl, grp in df.groupby("cluster"):
                denom = max(len(grp), 1)
                for val in top_vals:
                    prop = (grp[feat] == val).sum() / denom
                    all_rows.append({
                        "feature": feat,
                        "cluster": cl,
                        "category": str(val),
                        "proportion": prop,
                    })

        tall = pd.DataFrame(all_rows)
        if tall.empty:
            continue

        # One figure with multiple subplots
        nrows = len(features)
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(11, max(2.6 * nrows, 4)), sharex=True)
        if nrows == 1:
            axes = [axes]

        for ax, feat in zip(axes, features):
            df_feat = tall[tall["feature"] == feat]
            sns.barplot(
                data=df_feat,
                x="cluster",
                y="proportion",
                hue="category",
                ax=ax,
                palette="tab10",
            )
            ax.set_title(
                textwrap.fill(str(feat), width=80),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
            )
            ax.set_ylabel("Доля")
            ax.set_xlabel("Кластер")
            ax.legend(title="Категория", loc="upper right", fontsize=8)

        fig.suptitle(
            textwrap.fill(f"{title_prefix} — страница {page_idx}", width=100),
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", alpha=0.95),
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_file = out_dir / f"low_card_page_{page_idx}.png"
        plt.savefig(out_file, dpi=150)
        plt.close(fig)


def plot_clusters_pca(
    X_scaled: np.ndarray,
    clusters: np.ndarray,
    label_series: pd.Series,
    df_index: pd.Index,
    phishing_positive_value: int = -1,
    cluster_centers: Optional[np.ndarray] = None,
    title: str = "Визуализация кластеризации с использованием PCA",
    out_file: Path = None,
    figsize: tuple = (9, 7),
) -> None:
    """Plot clusters in 2D using PCA reduction with site type markers.

    Args:
        X_scaled: Scaled feature matrix
        clusters: Cluster labels for each sample
        label_series: True labels (phishing/legitimate)
        df_index: Index of the original dataframe (for aligning labels)
        phishing_positive_value: Value indicating phishing sites (default: -1)
        cluster_centers: Optional cluster centers (if available from model)
        title: Plot title
        out_file: Optional path to save the plot
        figsize: Figure size
    """
    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)

    # Transform cluster centers if provided
    centers_2 = None
    if cluster_centers is not None:
        centers_2 = pca.transform(cluster_centers)

    # Prepare data for visualization
    plot_df = pd.DataFrame(X2, columns=["x", "y"], index=df_index)
    plot_df["cluster"] = clusters
    
    # Align label_series with data index (same as in evaluation.py)
    labels = label_series.reindex(df_index)
    plot_df["label"] = labels.values
    
    # Identify phishing and legitimate sites
    plot_df["is_phishing"] = plot_df["label"] == phishing_positive_value

    # Number of clusters
    k = len(np.unique(clusters[clusters >= 0]))  # Exclude noise label -1 if present
    palette = sns.color_palette("tab10", n_colors=k)

    # Create plot
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot points separately for phishing and legitimate sites
    unique_clusters = sorted(np.unique(clusters[clusters >= 0]))
    
    # Plot phishing sites (squares)
    phishing_df = plot_df[plot_df["is_phishing"]]
    if len(phishing_df) > 0:
        for idx, cluster_label in enumerate(unique_clusters):
            cluster_data = phishing_df[phishing_df["cluster"] == cluster_label]
            if len(cluster_data) > 0:
                color_idx = idx % len(palette)
                ax.scatter(
                    cluster_data["x"],
                    cluster_data["y"],
                    marker="s",  # square for phishing
                    s=40,
                    c=[palette[color_idx]],
                    alpha=0.8,
                    edgecolor="k",
                    linewidth=0.15,
                    label=f"Cluster {cluster_label} (Phishing)" if idx == 0 else "",
                )
    
    # Plot legitimate sites (circles)
    legitimate_df = plot_df[~plot_df["is_phishing"]]
    if len(legitimate_df) > 0:
        for idx, cluster_label in enumerate(unique_clusters):
            cluster_data = legitimate_df[legitimate_df["cluster"] == cluster_label]
            if len(cluster_data) > 0:
                color_idx = idx % len(palette)
                ax.scatter(
                    cluster_data["x"],
                    cluster_data["y"],
                    marker="o",  # circle for legitimate
                    s=40,
                    c=[palette[color_idx]],
                    alpha=0.8,
                    edgecolor="k",
                    linewidth=0.15,
                    label=f"Cluster {cluster_label} (Legitimate)" if idx == 0 else "",
                )

    # Plot cluster centers if available
    if centers_2 is not None:
        # Ensure we have the same number of centers as clusters
        n_centers = min(len(centers_2), len(unique_clusters))
        for idx in range(n_centers):
            cluster_label = unique_clusters[idx]
            cx, cy = centers_2[idx]
            # Get color index - seaborn uses sorted order for palette mapping
            color_idx = idx % len(palette)
            plt.scatter(
                cx,
                cy,
                marker="*",
                s=350,
                c=[palette[color_idx]],
                edgecolor="black",
                linewidth=1.5,
                zorder=10,
            )
            plt.text(cx + 0.05, cy + 0.05, f"C{cluster_label}", fontsize=12, fontweight="bold")

    explained_variance = pca.explained_variance_ratio_
    plt.title(title)
    plt.xlabel(f"Главная компонента 1 ({explained_variance[0]*100:.2f}%)")
    plt.ylabel(f"Главная компонента 2 ({explained_variance[1]*100:.2f}%)")
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = []
    # Add cluster colors with appropriate markers
    for idx, cluster_label in enumerate(unique_clusters):
        color_idx = idx % len(palette)
        # Use circle as default marker for cluster legend
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[color_idx],
                   markersize=10, markeredgecolor="k", markeredgewidth=0.5, label=f"Кластер {cluster_label}")
        )
    # Add separator and site type markers
    legend_elements.append(
        Line2D([0], [0], marker="", color="w", label="")  # Empty line for spacing
    )
    legend_elements.append(
        Line2D([0], [0], marker="s", color="w", markerfacecolor="darkred",
               markersize=10, markeredgecolor="k", markeredgewidth=0.5, label="Фишинг (квадрат)")
    )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="darkgreen",
               markersize=10, markeredgecolor="k", markeredgewidth=0.5, label="Законный (круг)")
    )
    plt.legend(handles=legend_elements, loc="best", fontsize=9, framealpha=0.9)
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if out_file:
        _ensure_dir(out_file.parent)
        plt.savefig(out_file, dpi=150)
    plt.close()


def plot_chi2_feature_importance(
    feature_df: pd.DataFrame,
    label_series: pd.Series,
    title: str = 'Сила связи признаков с меткой "Result" (по статистике Хи-квадрат)',
    out_file: Path = None,
    figsize: tuple = (12, 10),
) -> pd.DataFrame:
    """Plot chi-square statistics for feature importance.

    Args:
        feature_df: DataFrame with features
        label_series: Target labels
        title: Plot title
        out_file: Optional path to save the plot
        figsize: Figure size

    Returns:
        DataFrame with chi-square statistics sorted by Chi2_Statistic
    """
    # Prepare data
    X = feature_df.apply(pd.to_numeric, errors="coerce").dropna(axis=1)
    y = label_series

    chi2_results = []

    for feature in X.columns:
        # Create contingency table
        contingency_table = pd.crosstab(X[feature], y)

        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        chi2_results.append(
            {
                "Feature": feature,
                "Chi2_Statistic": chi2,
                "P_Value": p_value,
            }
        )

    df_chi2 = pd.DataFrame(chi2_results).sort_values(by="Chi2_Statistic", ascending=False).reset_index(drop=True)

    # Visualization
    plt.figure(figsize=figsize)
    sns.barplot(
        x="Feature",
        y="Chi2_Statistic",
        hue="Feature",
        data=df_chi2,
        palette=sns.color_palette("viridis", len(df_chi2)),
        legend=False,
    )

    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.3)
    plt.title(title, fontsize=16)
    plt.ylabel("Статистика Хи-квадрат (мера силы связи)", fontsize=14)
    plt.xlabel("Признак", fontsize=14)
    plt.tight_layout()

    if out_file:
        _ensure_dir(out_file.parent)
        plt.savefig(out_file, dpi=150)
    plt.close()

    return df_chi2


