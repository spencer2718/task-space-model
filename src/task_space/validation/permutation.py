"""
Permutation tests and cross-validation for validation regression.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional

from .regression import simple_regression


@dataclass
class PermutationResult:
    """Results from permutation test."""
    observed_t: float
    observed_r2: float
    null_t_mean: float
    null_t_std: float
    null_r2_mean: float
    null_r2_std: float
    p_value: float
    percentile: float
    n_permutations: int


@dataclass
class CrossValidationResult:
    """Results from cross-validation."""
    train_r2_mean: float
    train_r2_std: float
    test_r2_mean: float
    test_r2_std: float
    overfit_ratio: float
    n_folds: int


def run_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray = None,
    n_permutations: int = 1000,
    seed: int = 42,
) -> PermutationResult:
    """
    Run permutation test to assess significance.

    Permutes y to break relationship with x while preserving marginal distribution.

    Args:
        x: (n,) regressor (e.g., similarity)
        y: (n,) outcome (e.g., comovement)
        cluster_ids: (n,) cluster identifiers (optional)
        n_permutations: Number of permutations
        seed: Random seed

    Returns:
        PermutationResult with observed and null distribution statistics
    """
    np.random.seed(seed)

    # Observed statistics
    obs_result = simple_regression(x, y, cluster_ids)
    observed_t = obs_result.t[1]  # t-stat on x
    observed_r2 = obs_result.r2

    # Null distribution
    null_t = []
    null_r2 = []

    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        perm_result = simple_regression(x, y_perm, cluster_ids)
        null_t.append(perm_result.t[1])
        null_r2.append(perm_result.r2)

    null_t = np.array(null_t)
    null_r2 = np.array(null_r2)

    # P-value: fraction of null t-stats more extreme than observed
    p_value = (np.abs(null_t) >= np.abs(observed_t)).mean()

    # Percentile of observed t in null distribution
    percentile = (null_t <= observed_t).mean() * 100

    return PermutationResult(
        observed_t=float(observed_t),
        observed_r2=float(observed_r2),
        null_t_mean=float(null_t.mean()),
        null_t_std=float(null_t.std()),
        null_r2_mean=float(null_r2.mean()),
        null_r2_std=float(null_r2.std()),
        p_value=float(p_value),
        percentile=float(percentile),
        n_permutations=n_permutations,
    )


def run_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> CrossValidationResult:
    """
    Run k-fold cross-validation.

    Args:
        x: (n,) regressor
        y: (n,) outcome
        n_folds: Number of folds
        seed: Random seed

    Returns:
        CrossValidationResult with train and test R^2
    """
    np.random.seed(seed)
    n = len(y)

    # Shuffle indices
    indices = np.random.permutation(n)
    fold_size = n // n_folds

    train_r2s = []
    test_r2s = []

    for fold in range(n_folds):
        # Test indices for this fold
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        # Split data
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit on train
        train_result = simple_regression(x_train, y_train)
        train_r2s.append(train_result.r2)

        # Predict on test
        beta = train_result.beta
        y_pred = beta[0] + beta[1] * x_test
        ss_res = np.sum((y_test - y_pred)**2)
        ss_tot = np.sum((y_test - y_test.mean())**2)
        test_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        test_r2s.append(test_r2)

    train_r2_mean = np.mean(train_r2s)
    test_r2_mean = np.mean(test_r2s)

    # Overfit ratio: how much worse is test vs train
    overfit_ratio = train_r2_mean / test_r2_mean if test_r2_mean > 0 else np.inf

    return CrossValidationResult(
        train_r2_mean=float(train_r2_mean),
        train_r2_std=float(np.std(train_r2s)),
        test_r2_mean=float(test_r2_mean),
        test_r2_std=float(np.std(test_r2s)),
        overfit_ratio=float(overfit_ratio),
        n_folds=n_folds,
    )


def run_random_baseline_comparison(
    x: np.ndarray,
    y: np.ndarray,
    n_activities: int,
    n_random: int = 100,
    seed: int = 42,
) -> dict:
    """
    Compare observed similarity to random embeddings.

    Generates random embeddings and computes their predictive power.

    Args:
        x: (n,) observed similarity (flattened upper triangle)
        y: (n,) observed comovement (flattened upper triangle)
        n_activities: Number of activities (for generating random embeddings)
        n_random: Number of random baselines
        seed: Random seed

    Returns:
        Dict with observed stats, random distribution, and percentile
    """
    np.random.seed(seed)

    # Observed
    obs_result = simple_regression(x, y)
    obs_t = obs_result.t[1]
    obs_r2 = obs_result.r2

    # Random baselines
    # Note: This is a simplified version - caller should provide
    # the machinery to convert random embeddings to similarity matrix
    random_t = []
    random_r2 = []

    for _ in range(n_random):
        # Random similarity values with same distribution
        x_random = np.random.permutation(x)
        rand_result = simple_regression(x_random, y)
        random_t.append(rand_result.t[1])
        random_r2.append(rand_result.r2)

    random_t = np.array(random_t)
    random_r2 = np.array(random_r2)

    return {
        'observed_t': float(obs_t),
        'observed_r2': float(obs_r2),
        'random_t_mean': float(random_t.mean()),
        'random_t_max': float(random_t.max()),
        'random_r2_mean': float(random_r2.mean()),
        'random_r2_max': float(random_r2.max()),
        't_percentile': float((random_t <= obs_t).mean() * 100),
        'r2_percentile': float((random_r2 <= obs_r2).mean() * 100),
        'n_random': n_random,
    }
