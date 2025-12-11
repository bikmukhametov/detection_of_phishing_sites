"""
Custom bagging implementation (simple) and helpers.

Provides:
- CustomBaggingClassifier: ensemble with random sampling of rows and features

This implementation is intentionally small and clear for educational use.
"""
from copy import deepcopy
from typing import List, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import accuracy_score


class CustomBaggingClassifier(BaseEstimator, ClassifierMixin):
    """Simple bagging ensemble with random sampling of samples and features.

    Parameters
    - base_estimator: sklearn-like estimator (must implement fit/predict_proba)
    - n_estimators: number of base estimators
    - max_samples: fraction (0-1] of samples to draw for each estimator
    - max_features: fraction (0-1] of features to draw for each estimator
    - bootstrap: whether to sample with replacement for samples
    - bootstrap_features: whether to sample features with replacement (usually False)
    - random_state: int or None
    """

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        n_estimators: int = 10,
        max_samples: float = 0.8,
        max_features: float = 0.8,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        random_state: Optional[int] = None,
    ):
        self.base_estimator = base_estimator or LogisticRegression(max_iter=1000)
        self.n_estimators = int(n_estimators)
        self.max_samples = float(max_samples)
        self.max_features = float(max_features)
        self.bootstrap = bool(bootstrap)
        self.bootstrap_features = bool(bootstrap_features)
        self.random_state = random_state

        # attributes to be filled after fit
        self.estimators_: List[BaseEstimator] = []
        self.features_: List[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        n_draw = max(1, int(np.ceil(self.max_samples * n_samples)))
        n_feats = max(1, int(np.ceil(self.max_features * n_features)))

        self.estimators_ = []
        self.features_ = []

        for i in range(self.n_estimators):
            # sample sample indices
            if self.bootstrap:
                sample_idx = rng.randint(0, n_samples, size=n_draw)
            else:
                sample_idx = rng.choice(n_samples, size=n_draw, replace=False)

            # sample feature indices
            if self.bootstrap_features:
                feat_idx = rng.randint(0, n_features, size=n_feats)
            else:
                feat_idx = rng.choice(n_features, size=n_feats, replace=False)

            clf = clone(self.base_estimator)
            X_sub = X[sample_idx][:, feat_idx]
            y_sub = y[sample_idx]
            clf.fit(X_sub, y_sub)

            self.estimators_.append(clf)
            self.features_.append(feat_idx)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # average predicted probabilities for class 1
        probs = []
        for clf, feat_idx in zip(self.estimators_, self.features_):
            X_sub = X[:, feat_idx]
            p = clf.predict_proba(X_sub)[:, 1]
            probs.append(p)
        probs = np.vstack(probs)  # shape (n_estimators, n_samples)
        avg = probs.mean(axis=0)
        # return two-column proba
        return np.vstack([1 - avg, avg]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
