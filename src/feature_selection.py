"""
Feature selection algorithms for phishing detection.

Implements three feature selection methods:
1. Add-Del (Incremental/Decremental) - Sequential forward selection with backward elimination
2. Genetic Algorithm - Evolutionary search with crossover and mutation
3. Stochastic Search without Adaptation (SSA) - Random search with probability updates
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")


def compute_quality_criterion(y_true, y_pred, y_pred_proba=None):
    """
    Compute quality criterion Q - the error rate on training set.
    
    Q(w) = 1/l * sum[a(x_i, w) * y_i < 0]
    
    Where a(x_i, w) is the prediction and y_i is the true label.
    We compute it as 1 - accuracy (error rate).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
    
    Returns:
        Q value (error rate)
    """
    accuracy = accuracy_score(y_true, y_pred)
    error_rate = 1.0 - accuracy
    return error_rate


def train_classifier(X_train, y_train, X_test, y_test):
    """
    Train logistic regression classifier and return predictions.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Tuple of (y_pred, y_pred_proba, model)
    """
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_proba, model


class FeatureSelectionTracker:
    """Track feature selection process and collect statistics."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = np.array(feature_names)
        self.iterations = []
        self.selected_features_history = []
        self.quality_history = []
        
    def record_iteration(self, iteration: int, selected_indices: np.ndarray, quality: float):
        """Record statistics for an iteration."""
        self.iterations.append(iteration)
        self.selected_features_history.append(selected_indices.copy())
        self.quality_history.append(quality)
    
    def get_selected_feature_names(self, iteration_idx: int = -1):
        """Get feature names for a specific iteration."""
        if iteration_idx < len(self.selected_features_history):
            indices = self.selected_features_history[iteration_idx]
            return self.feature_names[indices]
        return []


def add_del_algorithm(X, y, feature_names: List[str], 
                      max_iterations: int = 100, 
                      patience: int = 5,
                      test_size: float = 0.2) -> Tuple[np.ndarray, Dict]:
    """
    Add-Del (Sequential Addition-Deletion) Algorithm
    
    Combines forward selection (Add phase) with backward elimination (Del phase).
    Continues until no improvement or patience exceeded.
    
    Algorithm:
    1. Start with empty set J0
    2. Add phase: Incrementally add features that minimize Q (until |J| < n)
    3. Del phase: Remove features that minimize Q (while |J| > 0)
    4. Stop when Q doesn't decrease for d iterations
    
    Args:
        X: Feature matrix (n_samples x n_features)
        y: Labels
        feature_names: List of feature names
        max_iterations: Maximum iterations per phase
        patience: Number of iterations without improvement before stopping
        test_size: Proportion of data to use for testing
    
    Returns:
        Tuple of (selected_feature_indices, statistics_dict)
    """
    n_samples, n_features = X.shape
    
    # Split data
    split_idx = int(n_samples * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    tracker = FeatureSelectionTracker(feature_names)
    
    J_t = np.array([], dtype=int)  # Current feature set
    Q_star = float('inf')
    t_star = 0
    t = 0
    d = patience
    
    best_Q = float('inf')
    best_J = np.array([], dtype=int)
    
    # Add phase
    add_iteration = 0
    while len(J_t) < n_features and add_iteration < max_iterations:
        add_iteration += 1
        t += 1
        
        # Find feature to add that minimizes Q
        best_f = None
        best_Q_f = float('inf')
        
        for f in range(n_features):
            if f not in J_t:
                J_candidate = np.sort(np.append(J_t, f))
                X_train_subset = X_train[:, J_candidate]
                X_test_subset = X_test[:, J_candidate]
                
                try:
                    y_pred, _, _ = train_classifier(X_train_subset, y_train, X_test_subset, y_test)
                    Q_f = compute_quality_criterion(y_test, y_pred)
                    
                    if Q_f < best_Q_f:
                        best_Q_f = Q_f
                        best_f = f
                except:
                    continue
        
        if best_f is not None:
            J_t = np.sort(np.append(J_t, best_f))
            
            # Record statistics
            tracker.record_iteration(t, J_t, best_Q_f)
            
            if best_Q_f < Q_star:
                t_star = t
                Q_star = best_Q_f
                best_J = J_t.copy()
            
            if t - t_star >= d:
                break
        else:
            break
    
    # Del phase
    del_iteration = 0
    while len(J_t) > 0 and del_iteration < max_iterations:
        del_iteration += 1
        t += 1
        
        # Find feature to remove that minimizes Q
        best_f = None
        best_Q_f = float('inf')
        
        for f in J_t:
            J_candidate = J_t[J_t != f]
            
            if len(J_candidate) == 0:
                continue
            
            X_train_subset = X_train[:, J_candidate]
            X_test_subset = X_test[:, J_candidate]
            
            try:
                y_pred, _, _ = train_classifier(X_train_subset, y_train, X_test_subset, y_test)
                Q_f = compute_quality_criterion(y_test, y_pred)
                
                if Q_f < best_Q_f:
                    best_Q_f = Q_f
                    best_f = f
            except:
                continue
        
        if best_f is not None:
            J_t = J_t[J_t != best_f]
            
            # Record statistics
            if len(J_t) > 0:
                tracker.record_iteration(t, J_t, best_Q_f)
                
                if best_Q_f < Q_star:
                    t_star = t
                    Q_star = best_Q_f
                    best_J = J_t.copy()
                
                if t - t_star >= d:
                    break
        else:
            break
    
    if len(best_J) == 0:
        best_J = J_t.copy()
    
    return best_J, {
        'algorithm': 'Add-Del',
        'tracker': tracker,
        'iterations': t,
        'final_quality': Q_star,
        'selected_count': len(best_J),
    }


def genetic_algorithm(X, y, feature_names: List[str],
                     population_size: int = 50,
                     generations: int = 50,
                     mutation_rate: float = 0.1,
                     crossover_prob: float = 0.8,
                     selection_size: Optional[int] = None,
                     patience: int = 10,
                     test_size: float = 0.2) -> Tuple[np.ndarray, Dict]:
    """
    Genetic Algorithm for feature selection
    
    Evolves a population of binary feature masks using:
    - Selection: Tournament selection based on fitness (negative Q)
    - Crossover: Uniform crossover between parents
    - Mutation: Bit flip mutation
    
    Args:
        X: Feature matrix (n_samples x n_features)
        y: Labels
        feature_names: List of feature names
        population_size: Size of population
        generations: Number of generations to evolve
        mutation_rate: Probability of mutation per feature
        crossover_prob: Probability of crossover
        test_size: Proportion of data to use for testing
    
    Returns:
        Tuple of (selected_feature_indices, statistics_dict)
    """
    n_samples, n_features = X.shape
    
    # Split data
    split_idx = int(n_samples * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    tracker = FeatureSelectionTracker(feature_names)
    
    def fitness(chromosome):
        """Evaluate fitness of a chromosome (inverse of Q)."""
        selected = np.where(chromosome == 1)[0]
        
        if len(selected) == 0:
            return 0.0  # No features selected
        
        X_train_subset = X_train[:, selected]
        X_test_subset = X_test[:, selected]
        
        try:
            y_pred, _, _ = train_classifier(X_train_subset, y_train, X_test_subset, y_test)
            Q = compute_quality_criterion(y_test, y_pred)
            # Fitness = inverse of error, with penalty for many features
            return 1.0 / (Q + 1e-6) - len(selected) * 0.001
        except:
            return 0.0
    
    # Initialize population - ensure diversity with some full feature set
    population = np.random.randint(0, 2, size=(population_size, n_features))
    population[0] = np.ones(n_features)  # Start with all features
    
    best_fitness = -float('inf')
    best_chromosome = None
    
    # Determine selection_size (number of individuals to use as parents)
    if selection_size is None:
        selection_size = max(2, population_size // 4)

    # Track best and early stopping per pseudocode
    Q_star = float('inf')
    t_star = 0

    for gen in range(1, generations + 1):
        # Evaluate fitness -> lower Q is better, but fitness is higher when Q is lower
        fitnesses = np.array([fitness(chrom) for chrom in population])
        # Rank population by fitness descending (equivalently Q ascending)
        ranked_idx = np.argsort(-fitnesses)
        # Rt: selected parents (top selection_size)
        Rt_idx = ranked_idx[:selection_size]
        Rt = population[Rt_idx]

        # Best individual in current generation
        current_best_idx = ranked_idx[0]
        current_best_chrom = population[current_best_idx].copy()
        current_selected = np.where(current_best_chrom == 1)[0]
        if len(current_selected) > 0:
            X_subset = X_test[:, current_selected]
            y_pred, _, _ = train_classifier(X_train[:, current_selected], y_train, X_subset, y_test)
            Q_curr = compute_quality_criterion(y_test, y_pred)
        else:
            Q_curr = float('inf')

        # Update global best Q* and t*
        if Q_curr < Q_star:
            Q_star = Q_curr
            t_star = gen
            best_chromosome = current_best_chrom.copy()
            # record iteration (use generation index)
            tracker.record_iteration(gen, current_selected, Q_curr)

        # If no improvement for patience generations -> stop and return best
        if gen - t_star >= patience:
            break

        # Produce offspring via crossover/mutation
        # Per pseudocode: R_{t+1} := offspring ∪ R_t  (we do not trim population here)
        new_population = []
        # Keep parents (Rt) as part of next generation
        for r in Rt:
            new_population.append(r.copy())

        # Generate children by crossing parents from Rt
        # We'll generate population_size children pairs (can be adjusted)
        num_children_pairs = max(1, population_size)
        for _ in range(num_children_pairs):
            parents_idx = np.random.choice(len(Rt), size=2, replace=True)
            parent1 = Rt[parents_idx[0]].copy()
            parent2 = Rt[parents_idx[1]].copy()

            if np.random.random() < crossover_prob:
                mask = np.random.randint(0, 2, n_features)
                child1 = np.where(mask, parent1, parent2)
                child2 = np.where(mask, parent2, parent1)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            # Mutation
            for child in [child1, child2]:
                mutation_mask = np.random.random(n_features) < mutation_rate
                child[mutation_mask] = 1 - child[mutation_mask]
                new_population.append(child)

        # Now set population to the union (Rt U offspring) exactly per pseudocode
        try:
            population = np.array(new_population)
        except Exception:
            population = np.array(new_population[:population_size])
    
    selected_indices = np.where(best_chromosome == 1)[0]
    if len(selected_indices) == 0:
        selected_indices = np.array([0])
    
    # Final quality
    X_subset = X_test[:, selected_indices]
    y_pred, _, _ = train_classifier(X_train[:, selected_indices], y_train, X_subset, y_test)
    final_Q = compute_quality_criterion(y_test, y_pred)
    
    return selected_indices, {
        'algorithm': 'Genetic Algorithm',
        'tracker': tracker,
        'generations': generations,
        'final_quality': final_Q,
        'selected_count': len(selected_indices),
    }


def stochastic_search_with_adaptation(
    X,
    y,
    feature_names: List[str],
    j0: int = 1,
    T: int = 30,
    r: int = 10,
    h: float = 0.05,
    d: int = 3,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, Dict]:
    """
    Stochastic Search without Adaptation (SSA)
    
    Performs random search over feature subsets with uniform probability.
    No learning or probability adaptation across iterations.
    
    Algorithm:
    1. Initialize equal probabilities p_i = 1/n for each feature
    2. For each iteration t:
       a. Sample r random subsets of size j from distribution {p_1, ..., p_n}
       b. Find best subset J_t^min with lowest Q
       c. Find worst subset J_t^max with highest Q
       d. Adjust probabilities for features in J_t^max (but no adaptation in pure SSA)
       e. Find best overall J_t*
    3. Return best solution found
    
    Args:
        X: Feature matrix (n_samples x n_features)
        y: Labels
        feature_names: List of feature names
        j0: minimal subset size
        T: iterations per complexity
        r: samples per iteration
        h: punishment step
        d: patience/early-stop across complexities
        test_size: Proportion of data to use for testing
    
    Returns:
        Tuple of (selected_feature_indices, statistics_dict)
    """
    n_samples, n_features = X.shape

    split_idx = int(n_samples * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    tracker = FeatureSelectionTracker(feature_names)

    # Initialize equal probabilities
    p = np.ones(n_features) / n_features

    best_J = None
    best_Q = float('inf')

    # For each complexity j
    # Track best across complexities for early stopping
    Q_global_star = float('inf')
    j_star = j0

    for j in range(j0, n_features + 1):
        best_J_for_j = None
        best_Q_for_j = float('inf')
        for t in range(1, T + 1):
            J_min = None
            Q_min = float('inf')
            J_max = None
            Q_max = -float('inf')

            for _ in range(r):
                probs = p.copy()
                if probs.sum() == 0:
                    probs = np.ones_like(probs) / len(probs)
                else:
                    probs = probs / probs.sum()

                try:
                    selected = np.random.choice(n_features, size=j, replace=False, p=probs)
                except Exception:
                    selected = np.random.choice(n_features, size=j, replace=False)

                selected = np.sort(selected)
                X_train_subset = X_train[:, selected]
                X_test_subset = X_test[:, selected]

                try:
                    y_pred, _, _ = train_classifier(X_train_subset, y_train, X_test_subset, y_test)
                    Q = compute_quality_criterion(y_test, y_pred)

                    if Q < Q_min:
                        Q_min = Q
                        J_min = selected.copy()
                    if Q > Q_max:
                        Q_max = Q
                        J_max = selected.copy()

                    if Q < best_Q:
                        best_Q = Q
                        best_J = selected.copy()
                        tracker.record_iteration((j - j0) * T + t, best_J, best_Q)
                except Exception:
                    continue

            # punishment for features in J_max
            if J_max is not None:
                H = 0.0
                for s in J_max:
                    delta = min(p[s], h)
                    p[s] = max(0.0, p[s] - delta)
                    H += delta

                # reward features in J_min proportionally
                if J_min is not None and H > 0:
                    add_val = H / max(1, len(J_min))
                    for s in J_min:
                        p[s] = p[s] + add_val

                # normalize
                p = np.clip(p, 0.0, None)
                if p.sum() == 0:
                    p = np.ones_like(p) / len(p)
                else:
                    p = p / p.sum()

        # After T iterations for this complexity j, find best found for complexity j
        if best_J is not None:
            best_J_for_j = best_J.copy()
            best_Q_for_j = best_Q

        # Update global best across complexities and check early stopping per pseudocode
        if best_Q_for_j < Q_global_star:
            Q_global_star = best_Q_for_j
            j_star = j

        if j - j_star >= d:
            # return the best found for complexity j_star
            if best_J_for_j is None:
                best_J_for_j = np.array([0])
            return best_J_for_j, {
                'algorithm': 'Stochastic Search with Adaptation (SPA)',
                'tracker': tracker,
                'iterations': (j - j0 + 1) * T,
                'final_quality': Q_global_star,
                'selected_count': len(best_J_for_j),
            }

    if best_J is None:
        best_J = np.array([0])

    return best_J, {
        'algorithm': 'Stochastic Search with Adaptation (SPA)',
        'tracker': tracker,
        'iterations': (n_features - j0 + 1) * T,
        'final_quality': best_Q,
        'selected_count': len(best_J),
    }


def evaluate_feature_set(X, y, selected_features: np.ndarray, 
                        test_size: float = 0.2,
                        cv_folds: int = 5) -> Dict:
    """
    Evaluate quality of selected features using various metrics.
    
    Args:
        X: Feature matrix
        y: Labels
        selected_features: Indices of selected features
        test_size: Test set proportion
        cv_folds: Number of cross-validation folds
    
    Returns:
        Dictionary with metrics
    """
    if len(selected_features) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'roc_auc': 0.0,
            'error_rate': 1.0,
        }
    
    X_subset = X[:, selected_features]
    
    # Split data
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X_subset[:split_idx], X_subset[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    try:
        y_pred, y_pred_proba, model = train_classifier(X_train, y_train, X_test, y_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        error_rate = 1.0 - accuracy
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'error_rate': error_rate,
        }
    except Exception as e:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'roc_auc': 0.0,
            'error_rate': 1.0,
        }
