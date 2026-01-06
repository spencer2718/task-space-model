"""
Spearman correlation utilities for mobility validation.

This module centralizes Spearman correlation methodologies for evaluating
how well model predictions match observed transition patterns. Previously
these methods were duplicated across experiment scripts.

Two main approaches are provided:
1. Model probability methods (v0.7.0c style): Use fitted model P(j|i)
2. Raw 1/distance methods (v0.7.0.3b style): Use geometry directly

Key functions:
- aggregate_spearman_model_prob: Correlation over all destinations
- per_origin_spearman_model_prob: Per-origin correlations averaged
- aggregate_spearman_inv_distance: Geometry-only, common destinations
- per_origin_spearman_inv_distance: Per-origin geometry correlations

Usage:
    from task_space.validation.spearman import (
        aggregate_spearman_model_prob,
        per_origin_spearman_model_prob,
    )

    result = aggregate_spearman_model_prob(
        holdout_df=holdout,
        prob_matrix=P,
        census_codes=codes,
    )
    print(f"Aggregate Spearman: {result['spearman']}")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SpearmanResult:
    """
    Result from Spearman correlation computation.

    Attributes:
        spearman: Correlation coefficient (-1 to 1)
        p_value: Two-tailed p-value (if applicable)
        n_destinations: Number of destinations evaluated
        n_origins: Number of origins in sample
        method: Description of methodology used
    """
    spearman: float
    p_value: Optional[float]
    n_destinations: int
    n_origins: int
    method: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "spearman": round(self.spearman, 4) if self.spearman is not None else None,
            "p_value": round(self.p_value, 4) if self.p_value is not None else None,
            "n_destinations": self.n_destinations,
            "n_origins": self.n_origins,
            "method": self.method,
        }


@dataclass
class PerOriginSpearmanResult:
    """
    Result from per-origin Spearman correlation computation.

    Attributes:
        mean_spearman: Mean correlation across origins
        median_spearman: Median correlation across origins
        std_spearman: Standard deviation of correlations
        n_origins_evaluated: Number of origins with enough destinations
        correlations: List of individual origin correlations
        min_destinations_filter: Minimum destinations required per origin
        method: Description of methodology used
    """
    mean_spearman: float
    median_spearman: float
    std_spearman: float
    n_origins_evaluated: int
    correlations: List[float]
    min_destinations_filter: int
    method: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "mean_spearman": round(self.mean_spearman, 4),
            "median_spearman": round(self.median_spearman, 4),
            "std_spearman": round(self.std_spearman, 4),
            "n_origins_evaluated": self.n_origins_evaluated,
            "min_destinations_filter": self.min_destinations_filter,
            "method": self.method,
        }


# =============================================================================
# Model Probability Methods (v0.7.0c style)
# =============================================================================

def aggregate_spearman_model_prob(
    holdout_df: pd.DataFrame,
    prob_matrix: np.ndarray,
    census_codes: List[int],
    *,
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
) -> SpearmanResult:
    """
    Compute aggregate Spearman using model probabilities over all destinations.

    Methodology (from v0.7.0c):
    1. Weight model P(j|i) by origin counts in holdout
    2. Compute Spearman over ALL destinations (447)
    3. Compare predicted ranks to observed destination counts

    This evaluates how well the model ranks destinations by aggregate inflow.

    Args:
        holdout_df: Holdout transitions DataFrame
        prob_matrix: (n_occ, n_occ) probability matrix P(j|i)
        census_codes: List of Census occupation codes (matrix row/col labels)
        origin_col: Column name for origin occupation
        dest_col: Column name for destination occupation

    Returns:
        SpearmanResult with correlation and metadata

    Example:
        >>> result = aggregate_spearman_model_prob(holdout, P, codes)
        >>> print(f"Spearman: {result.spearman:.4f}")
    """
    code_to_idx = {c: i for i, c in enumerate(census_codes)}
    n_occ = len(census_codes)

    # Get origin distribution from holdout
    origin_counts = holdout_df[origin_col].value_counts()
    valid_origins = [o for o in origin_counts.index if o in code_to_idx]

    if not valid_origins:
        return SpearmanResult(
            spearman=np.nan,
            p_value=np.nan,
            n_destinations=n_occ,
            n_origins=0,
            method="model_probability_aggregate",
        )

    # Weighted average of predicted probabilities
    predicted_probs = np.zeros(n_occ)
    for origin in valid_origins:
        count = origin_counts[origin]
        i = code_to_idx[origin]
        predicted_probs += count * prob_matrix[i, :]
    predicted_probs /= predicted_probs.sum() + 1e-15

    # Build observed probability vector
    observed_dest_counts = holdout_df[dest_col].value_counts()
    observed_total = observed_dest_counts.sum()

    observed_probs = np.zeros(n_occ)
    for dest, count in observed_dest_counts.items():
        if dest in code_to_idx:
            j = code_to_idx[dest]
            observed_probs[j] = count / observed_total

    # Spearman over ALL destinations
    obs_ranks = stats.rankdata(-observed_probs)
    pred_ranks = stats.rankdata(-predicted_probs)
    spearman, p_value = stats.spearmanr(obs_ranks, pred_ranks)

    return SpearmanResult(
        spearman=float(spearman),
        p_value=float(p_value),
        n_destinations=n_occ,
        n_origins=len(valid_origins),
        method="model_probability_aggregate_all_destinations",
    )


def per_origin_spearman_model_prob(
    holdout_df: pd.DataFrame,
    prob_matrix: np.ndarray,
    census_codes: List[int],
    *,
    min_destinations: int = 5,
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
) -> PerOriginSpearmanResult:
    """
    Compute per-origin Spearman using model probabilities.

    Methodology (from v0.7.0c):
    For each origin with >= min_destinations observed transitions,
    compute Spearman between observed counts and model probability vector.

    This evaluates how well the model ranks destinations for each origin.

    Args:
        holdout_df: Holdout transitions DataFrame
        prob_matrix: (n_occ, n_occ) probability matrix P(j|i)
        census_codes: List of Census occupation codes
        min_destinations: Minimum observed destinations to include origin
        origin_col: Column name for origin occupation
        dest_col: Column name for destination occupation

    Returns:
        PerOriginSpearmanResult with mean, median, and individual correlations

    Example:
        >>> result = per_origin_spearman_model_prob(holdout, P, codes, min_destinations=5)
        >>> print(f"Mean Spearman: {result.mean_spearman:.4f} (n={result.n_origins_evaluated})")
    """
    code_to_idx = {c: i for i, c in enumerate(census_codes)}
    n_occ = len(census_codes)

    origin_counts = holdout_df[origin_col].value_counts()
    per_origin_rhos = []

    for origin in origin_counts.index:
        if origin not in code_to_idx:
            continue
        i = code_to_idx[origin]

        # Observed destinations for this origin
        origin_holdout = holdout_df[holdout_df[origin_col] == origin]
        origin_dest_counts = origin_holdout[dest_col].value_counts()

        if len(origin_dest_counts) < min_destinations:
            continue

        # Build vectors over ALL destinations
        obs_vec = np.zeros(n_occ)
        for dest, count in origin_dest_counts.items():
            if dest in code_to_idx:
                obs_vec[code_to_idx[dest]] = count

        pred_vec = prob_matrix[i, :]

        # Spearman
        if obs_vec.sum() > 0:
            corr, _ = stats.spearmanr(obs_vec, pred_vec)
            if not np.isnan(corr):
                per_origin_rhos.append(float(corr))

    if not per_origin_rhos:
        return PerOriginSpearmanResult(
            mean_spearman=np.nan,
            median_spearman=np.nan,
            std_spearman=np.nan,
            n_origins_evaluated=0,
            correlations=[],
            min_destinations_filter=min_destinations,
            method="model_probability_per_origin",
        )

    return PerOriginSpearmanResult(
        mean_spearman=float(np.mean(per_origin_rhos)),
        median_spearman=float(np.median(per_origin_rhos)),
        std_spearman=float(np.std(per_origin_rhos)),
        n_origins_evaluated=len(per_origin_rhos),
        correlations=per_origin_rhos,
        min_destinations_filter=min_destinations,
        method="model_probability_per_origin_all_destinations",
    )


# =============================================================================
# Inverse Distance Methods (v0.7.0.3b style)
# =============================================================================

def aggregate_spearman_inv_distance(
    holdout_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    census_codes: List[int],
    *,
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
) -> SpearmanResult:
    """
    Compute aggregate Spearman using raw 1/distance over common destinations.

    Methodology (from v0.7.0.3b):
    1. Sum 1/distance across all origins for each destination
    2. Compute Spearman over COMMON destinations only (non-zero in both)

    This is a geometry-only prediction without fitted parameters.

    Args:
        holdout_df: Holdout transitions DataFrame
        distance_matrix: (n_occ, n_occ) distance matrix
        census_codes: List of Census occupation codes
        origin_col: Column name for origin occupation
        dest_col: Column name for destination occupation

    Returns:
        SpearmanResult with correlation over common destinations

    Example:
        >>> result = aggregate_spearman_inv_distance(holdout, d_wass, codes)
        >>> print(f"Spearman: {result.spearman:.4f} (n_common={result.n_destinations})")
    """
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    # Observed inflows by destination
    observed_inflows = holdout_df[dest_col].value_counts().to_dict()

    # Geometry-based prediction: sum(1/d) over all origins
    origins = holdout_df[origin_col].unique()
    geo_scores = {}

    for origin in origins:
        if origin not in code_to_idx:
            continue
        i = code_to_idx[origin]

        for j, dest in enumerate(census_codes):
            if dest == origin:
                continue
            d = distance_matrix[i, j]
            if d > 0:
                geo_scores[dest] = geo_scores.get(dest, 0) + 1 / d

    # Common destinations (non-zero in both)
    common = set(geo_scores.keys()) & set(observed_inflows.keys())

    if len(common) < 3:
        return SpearmanResult(
            spearman=np.nan,
            p_value=np.nan,
            n_destinations=len(common),
            n_origins=len([o for o in origins if o in code_to_idx]),
            method="inverse_distance_aggregate_common",
        )

    pred_vec = [geo_scores[d] for d in common]
    obs_vec = [observed_inflows[d] for d in common]

    spearman, p_value = stats.spearmanr(pred_vec, obs_vec)

    return SpearmanResult(
        spearman=float(spearman),
        p_value=float(p_value),
        n_destinations=len(common),
        n_origins=len([o for o in origins if o in code_to_idx]),
        method="inverse_distance_aggregate_common_destinations",
    )


def per_origin_spearman_inv_distance(
    holdout_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    census_codes: List[int],
    *,
    min_destinations: int = 3,
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
) -> PerOriginSpearmanResult:
    """
    Compute per-origin Spearman using raw 1/distance over common destinations.

    Methodology (from v0.7.0.3b):
    For each origin with >= min_destinations observed, compute Spearman
    between observed counts and 1/distance (common destinations only).

    Note: This method tends to show higher correlations than model probability
    because it only evaluates over observed destinations (survivorship bias).

    Args:
        holdout_df: Holdout transitions DataFrame
        distance_matrix: (n_occ, n_occ) distance matrix
        census_codes: List of Census occupation codes
        min_destinations: Minimum observed destinations to include origin
        origin_col: Column name for origin occupation
        dest_col: Column name for destination occupation

    Returns:
        PerOriginSpearmanResult with mean, median, and individual correlations

    Example:
        >>> result = per_origin_spearman_inv_distance(holdout, d_wass, codes)
        >>> print(f"Mean Spearman: {result.mean_spearman:.4f}")
    """
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    origins = holdout_df[origin_col].unique()
    per_origin_rhos = []

    for origin in origins:
        if origin not in code_to_idx:
            continue
        i = code_to_idx[origin]

        # Observed destinations for this origin
        origin_df = holdout_df[holdout_df[origin_col] == origin]
        observed_counts = origin_df[dest_col].value_counts().to_dict()

        if len(observed_counts) < min_destinations:
            continue

        # Predicted scores: 1/distance for observed destinations
        predicted_scores = {}
        for dest in observed_counts.keys():
            if dest in code_to_idx and dest != origin:
                d = distance_matrix[i, code_to_idx[dest]]
                if d > 0:
                    predicted_scores[dest] = 1.0 / d

        # Common destinations
        common = set(observed_counts.keys()) & set(predicted_scores.keys())
        if len(common) < min_destinations:
            continue

        obs_vec = [observed_counts[d] for d in common]
        pred_vec = [predicted_scores[d] for d in common]

        rho, _ = stats.spearmanr(pred_vec, obs_vec)
        if not np.isnan(rho):
            per_origin_rhos.append(float(rho))

    if not per_origin_rhos:
        return PerOriginSpearmanResult(
            mean_spearman=np.nan,
            median_spearman=np.nan,
            std_spearman=np.nan,
            n_origins_evaluated=0,
            correlations=[],
            min_destinations_filter=min_destinations,
            method="inverse_distance_per_origin",
        )

    return PerOriginSpearmanResult(
        mean_spearman=float(np.mean(per_origin_rhos)),
        median_spearman=float(np.median(per_origin_rhos)),
        std_spearman=float(np.std(per_origin_rhos)),
        n_origins_evaluated=len(per_origin_rhos),
        correlations=per_origin_rhos,
        min_destinations_filter=min_destinations,
        method="inverse_distance_per_origin_common_destinations",
    )


# =============================================================================
# Bootstrap Utilities
# =============================================================================

def spearman_with_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute Spearman correlation with bootstrap confidence interval.

    Args:
        x: First variable (array-like)
        y: Second variable (array-like)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for interval (e.g., 0.95)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of:
            - Spearman correlation coefficient
            - (lower, upper) confidence interval bounds

    Example:
        >>> rho, (lo, hi) = spearman_with_bootstrap(x, y, n_bootstrap=1000)
        >>> print(f"Spearman: {rho:.3f} [{lo:.3f}, {hi:.3f}]")
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    rng = np.random.default_rng(random_state)

    # Point estimate
    rho, _ = stats.spearmanr(x, y)

    # Bootstrap
    bootstrap_rhos = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_rho, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(boot_rho):
            bootstrap_rhos.append(boot_rho)

    if not bootstrap_rhos:
        return float(rho), (np.nan, np.nan)

    # Percentile confidence interval
    alpha = 1 - confidence_level
    lo = float(np.percentile(bootstrap_rhos, 100 * alpha / 2))
    hi = float(np.percentile(bootstrap_rhos, 100 * (1 - alpha / 2)))

    return float(rho), (lo, hi)


# =============================================================================
# Model Probability Matrix Construction
# =============================================================================

def compute_model_probability_matrix(
    d_semantic: np.ndarray,
    d_institutional: np.ndarray,
    *,
    gamma_sem: float,
    gamma_inst: float,
) -> np.ndarray:
    """
    Compute conditional logit probability matrix P(j|i).

    Utility: U(i,j) = -gamma_sem * d_sem(i,j) - gamma_inst * d_inst(i,j)
    Probability: P(j|i) = exp(U(i,j)) / sum_k exp(U(i,k))

    Self-transitions are excluded (diagonal set to 0 probability).

    Args:
        d_semantic: (n, n) semantic distance matrix
        d_institutional: (n, n) institutional distance matrix
        gamma_sem: Coefficient on semantic distance (positive = disutility)
        gamma_inst: Coefficient on institutional distance

    Returns:
        (n, n) probability matrix where rows sum to 1

    Example:
        >>> P = compute_model_probability_matrix(d_wass, d_inst, gamma_sem=8.9, gamma_inst=0.14)
        >>> assert np.allclose(P.sum(axis=1), 1.0)
    """
    # Compute utility
    utility = -gamma_sem * d_semantic - gamma_inst * d_institutional

    # Exclude self-transitions
    np.fill_diagonal(utility, -np.inf)

    # Softmax (numerically stable)
    max_util = np.nanmax(utility, axis=1, keepdims=True)
    exp_utility = np.exp(utility - max_util)
    np.fill_diagonal(exp_utility, 0.0)

    row_sums = exp_utility.sum(axis=1, keepdims=True)
    prob_matrix = exp_utility / (row_sums + 1e-15)

    return prob_matrix


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data classes
    "SpearmanResult",
    "PerOriginSpearmanResult",
    # Model probability methods
    "aggregate_spearman_model_prob",
    "per_origin_spearman_model_prob",
    # Inverse distance methods
    "aggregate_spearman_inv_distance",
    "per_origin_spearman_inv_distance",
    # Utilities
    "spearman_with_bootstrap",
    "compute_model_probability_matrix",
]
