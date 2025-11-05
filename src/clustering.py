from typing import Dict, List

import numpy as np
from sklearn.cluster import Birch, DBSCAN, KMeans


def run_kmeans_set(X_scaled: np.ndarray, k_values: List[int]) -> Dict[int, dict]:
    results: Dict[int, dict] = {}
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(X_scaled)
        results[k] = {"model": model, "labels": labels}
    return results


def run_birch_set(X_scaled: np.ndarray, k_values: List[int]) -> Dict[int, dict]:
    results: Dict[int, dict] = {}
    for k in k_values:
        model = Birch(n_clusters=k)
        labels = model.fit_predict(X_scaled)
        results[k] = {"model": model, "labels": labels}
    return results


def run_dbscan_set(
    X_scaled: np.ndarray, eps_list: List[float], min_samples: int = 5
) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    for eps in eps_list:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
        key = f"eps={eps}_min={min_samples}"
        results[key] = {"model": model, "labels": labels}
    return results


