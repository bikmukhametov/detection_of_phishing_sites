from typing import Dict, List

import numpy as np
from sklearn.cluster import Birch, KMeans


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


