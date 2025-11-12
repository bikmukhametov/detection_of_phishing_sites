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
        self.phase_history = []  # For Add-Del: 'add' or 'del'
        self.complexity_history = []  # For SPA: complexity j
        
    def record_iteration(self, iteration: int, selected_indices: np.ndarray, quality: float, 
                        phase: str = None, complexity: int = None):
        """Record statistics for an iteration."""
        self.iterations.append(iteration)
        self.selected_features_history.append(selected_indices.copy())
        self.quality_history.append(quality)
        self.phase_history.append(phase)
        self.complexity_history.append(complexity)
    
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
    
    # Record initial state (empty set)
    if len(J_t) == 0:
        # Evaluate empty set
        Q_empty = 1.0  # Error rate for empty set (no features = 100% error)
        tracker.record_iteration(0, J_t, Q_empty, phase='init')
        Q_star = Q_empty
        best_Q = Q_empty
    
    # Outer loop: repeat Add and Del phases while Q decreases (according to pseudocode)
    max_outer_iterations = 10  # Prevent infinite loops
    outer_iter = 0
    
    while outer_iter < max_outer_iterations:
        outer_iter += 1
        Q_before_cycle = Q_star
        
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
                
                # Record statistics - record every iteration according to pseudocode
                tracker.record_iteration(t, J_t, best_Q_f, phase='add')
                
                if best_Q_f < Q_star:
                    t_star = t
                    Q_star = best_Q_f
                    best_J = J_t.copy()
                
                if t - t_star >= d:
                    break
            else:
                # Even if no feature found, record current state
                if len(J_t) > 0:
                    # Re-evaluate current set
                    X_train_subset = X_train[:, J_t]
                    X_test_subset = X_test[:, J_t]
                    try:
                        y_pred, _, _ = train_classifier(X_train_subset, y_train, X_test_subset, y_test)
                        Q_current = compute_quality_criterion(y_test, y_pred)
                        tracker.record_iteration(t, J_t, Q_current, phase='add')
                    except:
                        pass
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
                
                # Record statistics - record every iteration according to pseudocode
                if len(J_t) > 0:
                    tracker.record_iteration(t, J_t, best_Q_f, phase='del')
                    
                    if best_Q_f < Q_star:
                        t_star = t
                        Q_star = best_Q_f
                        best_J = J_t.copy()
                    
                    if t - t_star >= d:
                        break
                else:
                    # Record empty set
                    tracker.record_iteration(t, J_t, best_Q_f, phase='del')
                    if best_Q_f < Q_star:
                        t_star = t
                        Q_star = best_Q_f
                        best_J = J_t.copy()
            else:
                # Even if no feature to remove, record current state
                if len(J_t) > 0:
                    X_train_subset = X_train[:, J_t]
                    X_test_subset = X_test[:, J_t]
                    try:
                        y_pred, _, _ = train_classifier(X_train_subset, y_train, X_test_subset, y_test)
                        Q_current = compute_quality_criterion(y_test, y_pred)
                        tracker.record_iteration(t, J_t, Q_current, phase='del')
                    except:
                        pass
                break
        
        # Check if Q improved in this cycle
        if Q_star >= Q_before_cycle:
            # No improvement, stop outer loop
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
    t_star = 0
    Q_star = float('inf')
    
    # Determine selection_size (number of individuals to keep each generation)
    if selection_size is None:
        selection_size = max(2, population_size // 4)

    for gen in range(generations):
        # Evaluate fitness
        fitnesses = np.array([fitness(chrom) for chrom in population])

        # Rank population and select top individuals (R_t)
        ranked_idx = np.argsort(-fitnesses)
        Rt_idx = ranked_idx[:selection_size]
        Rt = population[Rt_idx]

        # Track best - record best solution of each generation (J¹_t according to pseudocode)
        current_best_idx = ranked_idx[0]
        best_chromosome_gen = population[current_best_idx].copy()
        selected_indices_gen = np.where(best_chromosome_gen == 1)[0]
        
        # Record best solution of this generation
        if len(selected_indices_gen) > 0:
            X_subset = X_test[:, selected_indices_gen]
            y_pred, _, _ = train_classifier(X_train[:, selected_indices_gen], y_train,
                                           X_subset, y_test)
            Q_gen = compute_quality_criterion(y_test, y_pred)
            tracker.record_iteration(gen + 1, selected_indices_gen, Q_gen)
        else:
            # Empty set
            tracker.record_iteration(gen + 1, selected_indices_gen, 1.0)
        
        # Update global best
        if fitnesses[current_best_idx] > best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_chromosome = best_chromosome_gen.copy()
        
        # Update Q_star and t_star for early stopping (step 6-7 of pseudocode)
        if len(selected_indices_gen) > 0:
            if Q_gen < Q_star:
                t_star = gen + 1
                Q_star = Q_gen
        
        # Check termination condition (step 7: if t - t* >= d then return)
        if (gen + 1) - t_star >= patience:
            break

        # Elitism: keep the best individual
        new_population = [population[current_best_idx].copy()]

        # Fill the rest of the population by breeding from Rt
        while len(new_population) < population_size:
            # Choose two parents from Rt (uniform)
            parents_idx = np.random.choice(len(Rt), size=2, replace=True)
            parent1 = Rt[parents_idx[0]].copy()
            parent2 = Rt[parents_idx[1]].copy()

            # Crossover
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
    d: int = 5,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, Dict]:
    """
    Stochastic Search with Adaptation (SPA)
    
    Performs random search over feature subsets with probability adaptation.
    Adapts probabilities based on best and worst solutions found.
    
    Algorithm:
    1. Initialize equal probabilities p_i = 1/n for each feature
    2. For each complexity j from j0 to n:
       a. For each iteration t = 1 to T:
          - Sample r random subsets of size j from distribution {p_1, ..., p_n}
          - Find best subset J_min with lowest Q
          - Find worst subset J_max with highest Q
          - Punish features in J_max, reward features in J_min
       b. Record best set J_j for complexity j
       c. Update global best if Q(J_j) < Q*
       d. Stop if j - j* >= d
    3. Return best solution J_j*
    
    Args:
        X: Feature matrix (n_samples x n_features)
        y: Labels
        feature_names: List of feature names
        j0: Starting complexity (minimum feature set size)
        T: Number of iterations per complexity level
        r: Number of random samples per iteration
        h: Penalty/reward step size
        d: Patience parameter for early stopping
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
    j_star = j0
    Q_star = float('inf')

    # For each complexity j
    for j in range(j0, n_features + 1):
        J_j = None  # Best set for complexity j
        Q_j = float('inf')  # Best Q for complexity j
        
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

                    # Track best overall
                    if Q < best_Q:
                        best_Q = Q
                        best_J = selected.copy()
                    
                    # Track best for this complexity j
                    if Q < Q_j:
                        Q_j = Q
                        J_j = selected.copy()
                except Exception:
                    continue

            # Record best set found in this iteration t (for better visualization)
            if J_min is not None:
                iteration_num = (j - j0) * T + t
                # Only record if it's an improvement or first iteration of this complexity
                if not tracker.selected_features_history or Q_min < tracker.quality_history[-1] or t == 1:
                    tracker.record_iteration(iteration_num, J_min, Q_min, complexity=j)

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
        
        # Also record final best set for complexity j (J_j according to pseudocode step 9)
        if J_j is not None:
            iteration_num = (j - j0) * T + T  # End of iteration for complexity j
            # Only record if it's different from last recorded
            if not tracker.selected_features_history or not np.array_equal(tracker.selected_features_history[-1], J_j):
                tracker.record_iteration(iteration_num, J_j, Q_j, complexity=j)
            
            # Update global best (step 10)
            if Q_j < Q_star:
                j_star = j
                Q_star = Q_j
                best_J = J_j.copy()
            
            # Check termination condition (step 11)
            if j - j_star >= d:
                break

    # Return J_j* according to pseudocode
    if best_J is None:
        best_J = np.array([0])
    
    return best_J, {
        'algorithm': 'Stochastic Search with Adaptation (SPA)',
        'tracker': tracker,
        'iterations': (n_features - j0 + 1) * T,
        'final_quality': Q_star if Q_star < float('inf') else best_Q,
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
