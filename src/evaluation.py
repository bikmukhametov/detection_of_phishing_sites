from typing import Iterable

import numpy as np
import pandas as pd


def build_cluster_summary_tables(
    df_index: Iterable,
    clusters: np.ndarray,
    label_series: pd.Series,
    phishing_positive_value: int = 1,
) -> pd.DataFrame:
    """Build summary table per cluster:
    - count
    - percent of total
    - percent phishing in cluster
    - percent of all phishing accounted by this cluster
    """
    cluster_labels = pd.Series(clusters, index=df_index, name="cluster")
    labels = label_series.reindex(df_index)

    total_n = len(cluster_labels)
    total_phishing = (labels == phishing_positive_value).sum()
    if total_phishing == 0:
        total_phishing = 1  # avoid div by zero; indicates labels may be different

    rows = []
    for cl in sorted(cluster_labels.dropna().unique()):
        mask = cluster_labels == cl
        n_in_cluster = int(mask.sum())
        phishing_in_cluster = int((labels[mask] == phishing_positive_value).sum())

        rows.append(
            {
                "cluster": int(cl),
                "count": n_in_cluster,
                "percent_of_total": 100.0 * n_in_cluster / total_n,
                "phishing_percent_in_cluster": 100.0 * phishing_in_cluster / max(n_in_cluster, 1),
                "phishing_share_of_all_phishing": 100.0 * phishing_in_cluster / total_phishing,
            }
        )

    # Handle DBSCAN noise label (-1) ordering last
    df = pd.DataFrame(rows)
    if (df["cluster"] == -1).any():
        df = pd.concat(
            [df[df["cluster"] != -1].sort_values("cluster"), df[df["cluster"] == -1]],
            ignore_index=True,
        )
    else:
        df = df.sort_values("cluster").reset_index(drop=True)

    # Add totals row at bottom
    totals = {
        "cluster": "TOTAL",
        "count": int(total_n),
        "percent_of_total": 100.0,
        "phishing_percent_in_cluster": np.nan,
        "phishing_share_of_all_phishing": 100.0,
    }
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    return df


