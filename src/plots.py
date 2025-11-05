from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _paginate(items: List[str], page_size: int) -> List[List[str]]:
    return [items[i : i + page_size] for i in range(0, len(items), page_size)]


def plot_cluster_feature_means_paged(
    X_scaled: np.ndarray,
    feature_names: List[str],
    clusters: np.ndarray,
    title_prefix: str,
    out_dir: Path,
    features_per_page: int = 15,
) -> None:
    """Plot cluster-wise means of scaled features.

    Split features into several figures to avoid scale crowding.
    """
    _ensure_dir(out_dir)

    df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    df_scaled["cluster"] = clusters

    feature_pages = _paginate(feature_names, features_per_page)

    for page_idx, features in enumerate(feature_pages, start=1):
        means = df_scaled.groupby("cluster")[features].mean()

        plt.figure(figsize=(max(8, len(features) * 0.6), 6))
        sns.heatmap(means.T, cmap="vlag", center=0, annot=False, cbar_kws={"shrink": 0.8})
        title_text = textwrap.fill(f"{title_prefix} — page {page_idx}", width=90)
        plt.title(
            title_text,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", alpha=0.9),
        )
        plt.xlabel("cluster")
        plt.ylabel("feature")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_file = out_dir / f"means_page_{page_idx}.png"
        plt.savefig(out_file, dpi=150)
        plt.close()


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
            ax.set_ylabel("proportion")
            ax.legend(title="category", loc="upper right", fontsize=8)

        fig.suptitle(
            textwrap.fill(f"{title_prefix} — page {page_idx}", width=100),
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", alpha=0.95),
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_file = out_dir / f"low_card_page_{page_idx}.png"
        plt.savefig(out_file, dpi=150)
        plt.close(fig)


