"""
Phase A: Raw Binary Overlap Baseline (v0.5.0)

Implements the sparsity hypothesis floor: does counting shared activities
predict wage comovement without any geometric structure?

Binary Jaccard overlap: |A_i ∩ A_j| / |A_i ∪ A_j|
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .domain import OccupationMeasures


@dataclass
class BinaryOverlapResult:
    """
    Binary overlap computation result.

    Attributes:
        overlap_matrix: Symmetric matrix of Jaccard overlaps, shape (n_occ, n_occ)
        occupation_codes: List of O*NET-SOC codes (row/column labels)
        threshold: Binarization threshold (τ)
        stats: Distribution statistics
        sparsity_stats: Activity sparsity statistics
    """
    overlap_matrix: np.ndarray
    occupation_codes: list[str]
    threshold: float
    stats: dict[str, float]
    sparsity_stats: dict[str, float]


@dataclass
class BaselineRegressionResult:
    """
    Baseline validation regression result.

    Attributes:
        beta: Coefficient on binary overlap
        se: Clustered standard error
        t_stat: t-statistic
        pvalue: Two-sided p-value
        r_squared: R-squared
        n_pairs: Number of occupation pairs
        n_clusters: Number of clusters (occupations)
        ci_lower: 95% CI lower bound
        ci_upper: 95% CI upper bound
        passes: True if beta > 0 and p < 0.10
    """
    beta: float
    se: float
    t_stat: float
    pvalue: float
    r_squared: float
    n_pairs: int
    n_clusters: int
    ci_lower: float
    ci_upper: float
    passes: bool


def compute_binary_overlap(
    measures: OccupationMeasures,
    threshold: float = 0.0,
) -> BinaryOverlapResult:
    """
    Compute binary Jaccard overlap between all occupation pairs.

    Binarizes the occupation-activity weight matrix at threshold τ,
    then computes Jaccard similarity for each pair.

    Args:
        measures: OccupationMeasures with raw_matrix (unnormalized weights)
        threshold: Binarization threshold τ. Default 0 = any positive weight.

    Returns:
        BinaryOverlapResult with Jaccard overlap matrix and statistics.
    """
    # Use raw (unnormalized) matrix for binarization
    W = measures.raw_matrix  # Shape: (J, N_activities)

    # Binarize: B[j, a] = 1 if W[j, a] > threshold else 0
    B = (W > threshold).astype(np.float64)

    # Compute activity counts per occupation
    activity_counts = B.sum(axis=1)

    # Jaccard overlap: |A_i ∩ A_j| / |A_i ∪ A_j|
    # intersection[i,j] = B[i] · B[j] = number of shared activities
    intersection = B @ B.T

    # union[i,j] = |A_i| + |A_j| - |A_i ∩ A_j|
    row_sums = activity_counts.reshape(-1, 1)
    union = row_sums + row_sums.T - intersection

    # Avoid division by zero (happens if both occupations have 0 activities)
    union[union == 0] = 1

    overlap_matrix = intersection / union

    # Compute statistics
    n = overlap_matrix.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    off_diag = overlap_matrix[triu_indices]

    stats = {
        "off_diag_mean": float(np.mean(off_diag)),
        "off_diag_std": float(np.std(off_diag)),
        "off_diag_min": float(np.min(off_diag)),
        "off_diag_p10": float(np.percentile(off_diag, 10)),
        "off_diag_p25": float(np.percentile(off_diag, 25)),
        "off_diag_p50": float(np.percentile(off_diag, 50)),
        "off_diag_p75": float(np.percentile(off_diag, 75)),
        "off_diag_p90": float(np.percentile(off_diag, 90)),
        "off_diag_max": float(np.max(off_diag)),
    }

    sparsity_stats = {
        "n_occupations": int(n),
        "n_activities": int(B.shape[1]),
        "mean_activities_per_occupation": float(np.mean(activity_counts)),
        "median_activities_per_occupation": float(np.median(activity_counts)),
        "min_activities": float(np.min(activity_counts)),
        "max_activities": float(np.max(activity_counts)),
        "total_nonzero_entries": int(np.sum(B)),
        "sparsity": float(1 - np.sum(B) / B.size),
    }

    return BinaryOverlapResult(
        overlap_matrix=overlap_matrix,
        occupation_codes=measures.occupation_codes,
        threshold=threshold,
        stats=stats,
        sparsity_stats=sparsity_stats,
    )


def _cluster_se(X: np.ndarray, y: np.ndarray, cluster_ids: np.ndarray) -> tuple:
    """
    Compute OLS coefficient with clustered standard errors.

    Args:
        X: Regressor (1D array)
        y: Outcome (1D array)
        cluster_ids: Cluster identifiers

    Returns:
        Tuple of (beta, clustered_se, r_squared, n_clusters)
    """
    n = len(y)

    # Add constant
    X_mat = np.column_stack([np.ones(n), X])

    # OLS
    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    beta_vec = XtX_inv @ (X_mat.T @ y)
    resid = y - X_mat @ beta_vec

    # R-squared
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Clustered standard errors
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    # Meat of sandwich: sum of outer products of cluster-summed scores
    meat = np.zeros((2, 2))
    for c in unique_clusters:
        mask = cluster_ids == c
        cluster_resid = resid[mask]
        cluster_X = X_mat[mask]
        score = cluster_X.T @ cluster_resid
        meat += np.outer(score, score)

    # Small-sample correction
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - 2))

    # Sandwich variance
    var_beta = correction * XtX_inv @ meat @ XtX_inv
    se_beta = np.sqrt(np.diag(var_beta))

    return beta_vec[1], se_beta[1], r_squared, n_clusters


def run_baseline_regression(
    overlap_result: BinaryOverlapResult,
    comovement_matrix: np.ndarray,
    comovement_codes: list[str],
    crosswalk_map: dict[str, str],
) -> BaselineRegressionResult:
    """
    Run baseline validation regression: WageComovement ~ BinaryOverlap.

    Args:
        overlap_result: BinaryOverlapResult from compute_binary_overlap
        comovement_matrix: Wage comovement matrix (SOC × SOC)
        comovement_codes: SOC codes for comovement matrix
        crosswalk_map: Dict mapping O*NET-SOC codes to SOC codes

    Returns:
        BaselineRegressionResult with regression statistics.
    """
    # Step 1: Map O*NET occupations to SOC codes
    onet_to_soc = {}
    for onet_code in overlap_result.occupation_codes:
        soc = crosswalk_map.get(onet_code)
        if soc:
            onet_to_soc[onet_code] = soc

    # Step 2: Aggregate overlap to SOC level (average overlaps for O*NET→SOC many:1)
    soc_codes = list(set(onet_to_soc.values()))
    soc_codes = [soc for soc in soc_codes if soc in comovement_codes]
    soc_codes = sorted(soc_codes)

    n_soc = len(soc_codes)
    soc_to_idx = {soc: i for i, soc in enumerate(soc_codes)}
    comovement_idx = {soc: i for i, soc in enumerate(comovement_codes)}

    # Build aggregated overlap matrix at SOC level
    # For each SOC pair, average the overlaps of their constituent O*NET occupations
    soc_overlap = np.zeros((n_soc, n_soc))
    soc_counts = np.zeros((n_soc, n_soc))

    onet_to_overlap_idx = {code: i for i, code in enumerate(overlap_result.occupation_codes)}

    for onet_i, soc_i in onet_to_soc.items():
        if soc_i not in soc_to_idx:
            continue
        for onet_j, soc_j in onet_to_soc.items():
            if soc_j not in soc_to_idx:
                continue
            if onet_i >= onet_j:  # Only upper triangle
                continue

            i_idx = onet_to_overlap_idx[onet_i]
            j_idx = onet_to_overlap_idx[onet_j]
            overlap_val = overlap_result.overlap_matrix[i_idx, j_idx]

            si = soc_to_idx[soc_i]
            sj = soc_to_idx[soc_j]
            if si > sj:
                si, sj = sj, si

            soc_overlap[si, sj] += overlap_val
            soc_counts[si, sj] += 1

    # Average
    soc_counts[soc_counts == 0] = 1
    soc_overlap = soc_overlap / soc_counts

    # Step 3: Build pair-level dataset
    pairs = []
    for i in range(n_soc):
        for j in range(i + 1, n_soc):
            soc_i = soc_codes[i]
            soc_j = soc_codes[j]

            # Get comovement value
            ci = comovement_idx[soc_i]
            cj = comovement_idx[soc_j]
            comove_val = comovement_matrix[ci, cj]

            if np.isnan(comove_val):
                continue

            overlap_val = soc_overlap[i, j]

            pairs.append({
                "occ_i": soc_i,
                "occ_j": soc_j,
                "overlap": overlap_val,
                "y": comove_val,
            })

    if len(pairs) == 0:
        raise ValueError("No valid occupation pairs found for regression")

    df = pd.DataFrame(pairs)

    # Step 4: Run regression with clustered SEs
    X = df["overlap"].values
    y = df["y"].values
    clusters = df["occ_i"].values

    unique_clusters, cluster_ids = np.unique(clusters, return_inverse=True)

    beta, se, r_squared, n_clusters = _cluster_se(X, y, cluster_ids)

    # t-stat and p-value
    t_stat = beta / se
    pvalue = 2 * (1 - sp_stats.t.cdf(np.abs(t_stat), df=n_clusters - 1))

    # 95% CI
    t_crit = sp_stats.t.ppf(0.975, df=n_clusters - 1)
    ci_lower = beta - t_crit * se
    ci_upper = beta + t_crit * se

    # Pass criterion: beta > 0 and p < 0.10
    passes = (beta > 0) and (pvalue < 0.10)

    return BaselineRegressionResult(
        beta=beta,
        se=se,
        t_stat=t_stat,
        pvalue=pvalue,
        r_squared=r_squared,
        n_pairs=len(df),
        n_clusters=n_clusters,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        passes=passes,
    )


def save_baseline_results(
    overlap_result: BinaryOverlapResult,
    regression_result: BaselineRegressionResult,
    output_dir: Path,
) -> None:
    """
    Save baseline results to disk.

    Args:
        overlap_result: BinaryOverlapResult
        regression_result: BaselineRegressionResult
        output_dir: Directory to save to
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save overlap matrix
    np.save(output_dir / "binary_overlap.npy", overlap_result.overlap_matrix)

    # Save results JSON
    results = {
        "phase": "A",
        "description": "Raw Binary Overlap Baseline",
        "overlap": {
            "threshold": float(overlap_result.threshold),
            "stats": overlap_result.stats,
            "sparsity": overlap_result.sparsity_stats,
        },
        "regression": {
            "beta": float(regression_result.beta),
            "se": float(regression_result.se),
            "t_stat": float(regression_result.t_stat),
            "pvalue": float(regression_result.pvalue),
            "r_squared": float(regression_result.r_squared),
            "n_pairs": int(regression_result.n_pairs),
            "n_clusters": int(regression_result.n_clusters),
            "ci_lower": float(regression_result.ci_lower),
            "ci_upper": float(regression_result.ci_upper),
            "passes": bool(regression_result.passes),
        },
        "decision_gate": {
            "criterion": "beta > 0 and p < 0.10",
            "passes": bool(regression_result.passes),
            "interpretation": (
                "Baseline established - SAE must beat this R²"
                if regression_result.passes
                else "O*NET structure doesn't predict wages - investigate data quality"
            ),
        },
    }

    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPhase A Baseline Results:")
    print(f"  β = {regression_result.beta:.4f}")
    print(f"  SE = {regression_result.se:.4f}")
    print(f"  t = {regression_result.t_stat:.2f}")
    print(f"  p = {regression_result.pvalue:.4f}")
    print(f"  R² = {regression_result.r_squared:.6f}")
    print(f"  n_pairs = {regression_result.n_pairs}")
    print(f"  Decision: {'PASS' if regression_result.passes else 'FAIL'}")
