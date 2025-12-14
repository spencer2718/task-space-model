"""
Validation regression utilities.

SINGLE implementation of clustered standard errors.
All validation code must use this module.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import scipy.stats


@dataclass
class RegressionResult:
    """Results from validation regression."""
    beta: np.ndarray           # Coefficients
    se: np.ndarray             # Standard errors (clustered)
    t: np.ndarray              # t-statistics
    p: np.ndarray              # p-values
    r2: float                  # R-squared
    n_pairs: int               # Number of observations
    n_clusters: int            # Number of clusters
    variable_names: list[str]  # Names of variables


def compute_clustered_se(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    """
    Compute cluster-robust standard errors.

    THIS IS THE SINGLE IMPLEMENTATION. Use this everywhere.

    Args:
        X: (n, k) design matrix (must include constant if desired)
        residuals: (n,) OLS residuals
        cluster_ids: (n,) cluster identifiers

    Returns:
        (k,) standard errors
    """
    n, k = X.shape
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    # Bread: (X'X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)

    # Meat: sum of cluster score outer products
    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = cluster_ids == c
        X_c = X[mask]
        e_c = residuals[mask]
        score_c = X_c.T @ e_c
        meat += np.outer(score_c, score_c)

    # Small-sample adjustment (HC1-style)
    adjustment = n_clusters / (n_clusters - 1) * (n - 1) / (n - k)

    # Sandwich
    V = XtX_inv @ meat @ XtX_inv * adjustment

    return np.sqrt(np.diag(V))


def run_validation_regression(
    similarity: np.ndarray,
    comovement: np.ndarray,
    sim_codes: list[str],
    comove_codes: list[str],
    crosswalk: dict[str, str],
    cluster_by: str = 'origin',
    controls: Optional[pd.DataFrame] = None,
) -> RegressionResult:
    """
    Run validation regression of comovement on similarity.

    Args:
        similarity: (n_sim, n_sim) similarity/overlap matrix
        comovement: (n_comove, n_comove) comovement matrix
        sim_codes: O*NET-SOC codes for similarity rows/cols
        comove_codes: SOC codes for comovement rows/cols
        crosswalk: O*NET-SOC -> SOC mapping (dict)
        cluster_by: 'origin' or 'destination'
        controls: Optional DataFrame with control variables
            Must have 'origin_soc', 'dest_soc' columns for merging

    Returns:
        RegressionResult
    """
    # Build pair dataset
    pairs = _build_pair_dataset(similarity, comovement, sim_codes, comove_codes, crosswalk)

    if len(pairs) == 0:
        raise ValueError("No valid occupation pairs found for regression")

    # Merge controls if provided
    control_cols = []
    if controls is not None:
        pairs = pairs.merge(controls, on=['origin_soc', 'dest_soc'], how='left')
        control_cols = [c for c in controls.columns if c not in ['origin_soc', 'dest_soc']]

    # Design matrix
    y = pairs['y'].values
    X_data = pairs[['x'] + control_cols].values
    X = np.column_stack([np.ones(len(y)), X_data])
    variable_names = ['const', 'similarity'] + control_cols

    # OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta

    # Clustered SEs
    cluster_col = 'origin_soc' if cluster_by == 'origin' else 'dest_soc'
    _, cluster_ids = np.unique(pairs[cluster_col].values, return_inverse=True)
    se = compute_clustered_se(X, residuals, cluster_ids)

    # Statistics
    t = beta / se
    df = len(np.unique(cluster_ids)) - 1
    p = 2 * (1 - scipy.stats.t.cdf(np.abs(t), df=df))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return RegressionResult(
        beta=beta,
        se=se,
        t=t,
        p=p,
        r2=r2,
        n_pairs=len(y),
        n_clusters=len(np.unique(cluster_ids)),
        variable_names=variable_names,
    )


def _build_pair_dataset(
    similarity: np.ndarray,
    comovement: np.ndarray,
    sim_codes: list[str],
    comove_codes: list[str],
    crosswalk: dict[str, str],
) -> pd.DataFrame:
    """Build DataFrame of occupation pairs with similarity and comovement."""
    # Create mappings
    sim_code_to_idx = {c: i for i, c in enumerate(sim_codes)}
    comove_code_to_idx = {c: i for i, c in enumerate(comove_codes)}

    pairs = []
    for i, onet_i in enumerate(sim_codes):
        soc_i = crosswalk.get(onet_i)
        if soc_i is None or soc_i not in comove_code_to_idx:
            continue
        ci = comove_code_to_idx[soc_i]

        for j, onet_j in enumerate(sim_codes):
            if i >= j:
                continue
            soc_j = crosswalk.get(onet_j)
            if soc_j is None or soc_j not in comove_code_to_idx:
                continue
            cj = comove_code_to_idx[soc_j]

            comove_val = comovement[ci, cj]
            if np.isnan(comove_val):
                continue

            pairs.append({
                'origin_soc': soc_i,
                'dest_soc': soc_j,
                'x': similarity[i, j],
                'y': comove_val,
            })

    return pd.DataFrame(pairs)


def simple_regression(
    x: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray = None,
) -> RegressionResult:
    """
    Simple bivariate regression with optional clustering.

    Args:
        x: (n,) regressor
        y: (n,) outcome
        cluster_ids: (n,) cluster identifiers (optional)

    Returns:
        RegressionResult
    """
    n = len(y)
    X = np.column_stack([np.ones(n), x])

    # OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta

    # Standard errors
    if cluster_ids is not None:
        se = compute_clustered_se(X, residuals, cluster_ids)
        n_clusters = len(np.unique(cluster_ids))
        df = n_clusters - 1
    else:
        ss_res = np.sum(residuals**2)
        var_resid = ss_res / (n - 2)
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(var_resid * np.diag(XtX_inv))
        n_clusters = n
        df = n - 2

    # Statistics
    t = beta / se
    p = 2 * (1 - scipy.stats.t.cdf(np.abs(t), df=df))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return RegressionResult(
        beta=beta,
        se=se,
        t=t,
        p=p,
        r2=r2,
        n_pairs=n,
        n_clusters=n_clusters,
        variable_names=['const', 'x'],
    )
