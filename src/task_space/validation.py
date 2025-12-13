"""
Phase I external validation module.

Implements Diagnostic B (External Monotonicity Validation) from paper Section 4.4.
Computes occupation-pair overlap and prepares data for validation regressions.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .distances import ActivityDistances, distance_percentiles
from .domain import OccupationMeasures
from .kernel import build_kernel_matrix, compute_overlap
from .crosswalk import OnetOesCrosswalk, WageComovement, aggregate_occupation_measures


# Pre-committed sigma percentiles for validation
SIGMA_PERCENTILES = ["p10", "p25", "p50", "p75", "p90"]


@dataclass
class OverlapResult:
    """
    Occupation-pair overlap computation result for a single sigma.

    Attributes:
        overlap_matrix: Symmetric matrix of pairwise overlaps, shape (n_occ, n_occ)
        occupation_codes: List of O*NET-SOC codes (row/column labels)
        sigma: Kernel bandwidth value used
        sigma_percentile: Which percentile this sigma corresponds to (e.g., "p50")
        stats: Distribution statistics of off-diagonal overlaps
    """
    overlap_matrix: np.ndarray
    occupation_codes: list[str]
    sigma: float
    sigma_percentile: str
    stats: dict[str, float]


@dataclass
class OverlapGrid:
    """
    Overlap results for all sigma values in the pre-committed grid.

    Attributes:
        results: Dict mapping percentile key to OverlapResult
        distance_percentiles: The actual sigma values for each percentile
        n_occupations: Number of occupations
    """
    results: dict[str, OverlapResult]
    distance_percentiles: dict[str, float]
    n_occupations: int


def compute_overlap_stats(overlap_matrix: np.ndarray) -> dict[str, float]:
    """
    Compute distribution statistics for overlap matrix.

    Args:
        overlap_matrix: Symmetric (n_occ, n_occ) overlap matrix

    Returns:
        Dict with statistics for off-diagonal and diagonal entries
    """
    n = overlap_matrix.shape[0]

    # Off-diagonal entries (upper triangle, excluding diagonal)
    triu_indices = np.triu_indices(n, k=1)
    off_diag = overlap_matrix[triu_indices]

    # Diagonal entries
    diag = np.diag(overlap_matrix)

    stats = {
        # Off-diagonal statistics
        "off_diag_mean": float(np.mean(off_diag)),
        "off_diag_std": float(np.std(off_diag)),
        "off_diag_min": float(np.min(off_diag)),
        "off_diag_p10": float(np.percentile(off_diag, 10)),
        "off_diag_p25": float(np.percentile(off_diag, 25)),
        "off_diag_p50": float(np.percentile(off_diag, 50)),
        "off_diag_p75": float(np.percentile(off_diag, 75)),
        "off_diag_p90": float(np.percentile(off_diag, 90)),
        "off_diag_max": float(np.max(off_diag)),
        # Diagonal statistics
        "diag_mean": float(np.mean(diag)),
        "diag_std": float(np.std(diag)),
        "diag_min": float(np.min(diag)),
        "diag_max": float(np.max(diag)),
    }

    return stats


def compute_validation_overlap(
    measures: OccupationMeasures,
    distances: ActivityDistances,
    sigma_percentile: str = "p50",
) -> OverlapResult:
    """
    Compute occupation-pair overlap for validation at a single sigma.

    Uses the same kernel construction as shock propagation:
        Overlap(i, j) = rho_i^T @ K @ rho_j

    Where K is the row-normalized exponential kernel matrix.

    Args:
        measures: OccupationMeasures with occupation probability distributions
        distances: ActivityDistances with pairwise activity distances
        sigma_percentile: Which distance percentile to use for sigma
                         Must be one of: "p10", "p25", "p50", "p75", "p90"

    Returns:
        OverlapResult with overlap matrix and statistics
    """
    if sigma_percentile not in SIGMA_PERCENTILES:
        raise ValueError(
            f"sigma_percentile must be one of {SIGMA_PERCENTILES}, got {sigma_percentile}"
        )

    # Get sigma value from distance percentiles
    dist_pcts = distance_percentiles(distances)
    sigma = dist_pcts[sigma_percentile]

    # Build kernel matrix with this sigma
    kernel = build_kernel_matrix(distances, sigma=sigma, kernel_type="exponential")

    # Compute overlap matrix using existing function
    overlap_matrix = compute_overlap(measures, kernel)

    # Symmetrize: the kernel K is row-normalized (not symmetric), causing small
    # asymmetries in the overlap. Conceptually overlap(i,j) should equal overlap(j,i),
    # so we symmetrize by averaging.
    overlap_matrix = (overlap_matrix + overlap_matrix.T) / 2

    # Compute statistics
    stats = compute_overlap_stats(overlap_matrix)

    return OverlapResult(
        overlap_matrix=overlap_matrix,
        occupation_codes=measures.occupation_codes,
        sigma=sigma,
        sigma_percentile=sigma_percentile,
        stats=stats,
    )


def compute_overlap_grid(
    measures: OccupationMeasures,
    distances: ActivityDistances,
) -> OverlapGrid:
    """
    Compute overlap matrices for all 5 pre-committed sigma values.

    This is the main entry point for Pass 1. Computes overlaps at:
    p10, p25, p50, p75, p90 of pairwise activity distances.

    Args:
        measures: OccupationMeasures with occupation probability distributions
        distances: ActivityDistances with pairwise activity distances

    Returns:
        OverlapGrid with results for all 5 sigma values
    """
    # Get all distance percentiles
    dist_pcts = distance_percentiles(distances)

    results = {}
    for pct in SIGMA_PERCENTILES:
        results[pct] = compute_validation_overlap(measures, distances, pct)

    # Extract just the sigma values we used
    sigma_values = {pct: dist_pcts[pct] for pct in SIGMA_PERCENTILES}

    return OverlapGrid(
        results=results,
        distance_percentiles=sigma_values,
        n_occupations=len(measures.occupation_codes),
    )


def save_overlap_result(result: OverlapResult, path: Path) -> None:
    """
    Save a single OverlapResult to disk.

    Saves as .npz for matrix and .json for metadata.

    Args:
        result: OverlapResult to save
        path: Base path (will create path.npz and path.json)
    """
    path = Path(path)

    # Save matrix as .npz
    np.savez_compressed(
        path.with_suffix(".npz"),
        overlap_matrix=result.overlap_matrix,
    )

    # Save metadata as .json
    metadata = {
        "occupation_codes": result.occupation_codes,
        "sigma": result.sigma,
        "sigma_percentile": result.sigma_percentile,
        "stats": result.stats,
    }
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)


def load_overlap_result(path: Path) -> OverlapResult:
    """
    Load a single OverlapResult from disk.

    Args:
        path: Base path (expects path.npz and path.json)

    Returns:
        OverlapResult
    """
    path = Path(path)

    # Load matrix
    data = np.load(path.with_suffix(".npz"))
    overlap_matrix = data["overlap_matrix"]

    # Load metadata
    with open(path.with_suffix(".json")) as f:
        metadata = json.load(f)

    return OverlapResult(
        overlap_matrix=overlap_matrix,
        occupation_codes=metadata["occupation_codes"],
        sigma=metadata["sigma"],
        sigma_percentile=metadata["sigma_percentile"],
        stats=metadata["stats"],
    )


def save_overlap_grid(grid: OverlapGrid, output_dir: Path) -> None:
    """
    Save all overlap matrices and metadata to disk.

    Creates:
        output_dir/overlap_p10.npz, overlap_p10.json
        output_dir/overlap_p25.npz, overlap_p25.json
        ...
        output_dir/overlap_stats.json (summary)

    Args:
        grid: OverlapGrid to save
        output_dir: Directory to save to (will be created if needed)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each result
    for pct, result in grid.results.items():
        save_overlap_result(result, output_dir / f"overlap_{pct}")

    # Save summary statistics
    summary = {
        "n_occupations": grid.n_occupations,
        "n_pairs": grid.n_occupations * (grid.n_occupations - 1) // 2,
        "sigma_grid": {
            pct: {
                "sigma": grid.distance_percentiles[pct],
                "off_diag_mean": grid.results[pct].stats["off_diag_mean"],
                "off_diag_std": grid.results[pct].stats["off_diag_std"],
            }
            for pct in SIGMA_PERCENTILES
        },
        "headline_p50": {
            "off_diagonal": {
                "mean": grid.results["p50"].stats["off_diag_mean"],
                "std": grid.results["p50"].stats["off_diag_std"],
                "min": grid.results["p50"].stats["off_diag_min"],
                "p10": grid.results["p50"].stats["off_diag_p10"],
                "p25": grid.results["p50"].stats["off_diag_p25"],
                "p50": grid.results["p50"].stats["off_diag_p50"],
                "p75": grid.results["p50"].stats["off_diag_p75"],
                "p90": grid.results["p50"].stats["off_diag_p90"],
                "max": grid.results["p50"].stats["off_diag_max"],
            },
            "diagonal": {
                "mean": grid.results["p50"].stats["diag_mean"],
                "std": grid.results["p50"].stats["diag_std"],
            },
        },
    }

    with open(output_dir / "overlap_stats.json", "w") as f:
        json.dump(summary, f, indent=2)


def load_overlap_grid(output_dir: Path) -> OverlapGrid:
    """
    Load previously computed overlap grid from disk.

    Args:
        output_dir: Directory containing saved overlap files

    Returns:
        OverlapGrid
    """
    output_dir = Path(output_dir)

    # Load each result
    results = {}
    for pct in SIGMA_PERCENTILES:
        results[pct] = load_overlap_result(output_dir / f"overlap_{pct}")

    # Reconstruct distance percentiles from loaded results
    distance_percentiles = {
        pct: results[pct].sigma for pct in SIGMA_PERCENTILES
    }

    n_occupations = len(results["p50"].occupation_codes)

    return OverlapGrid(
        results=results,
        distance_percentiles=distance_percentiles,
        n_occupations=n_occupations,
    )


# =============================================================================
# Pass 3: Validation Regression Implementation
# =============================================================================


@dataclass
class ValidationDataset:
    """
    Occupation-pair dataset for validation regression.

    Attributes:
        pair_data: DataFrame with columns occ_i, occ_j, overlap, y
        n_pairs: Number of occupation pairs
        n_occupations: Number of unique occupations
        y_variable: Name of dependent variable ("wage_comovement")
    """
    pair_data: pd.DataFrame
    n_pairs: int
    n_occupations: int
    y_variable: str


@dataclass
class RegressionResult:
    """
    Validation regression results for a single sigma.

    Attributes:
        sigma_percentile: Which percentile (e.g., "p50")
        sigma: Actual sigma value
        beta: Coefficient on overlap
        se: Clustered standard error
        pvalue: Two-sided p-value
        r_squared: R-squared
        n_pairs: Number of pairs in regression
        n_clusters: Number of clusters
        ci_lower: 95% CI lower bound
        ci_upper: 95% CI upper bound
        passes: True if beta > 0 and p < 0.10
    """
    sigma_percentile: str
    sigma: float
    beta: float
    se: float
    pvalue: float
    r_squared: float
    n_pairs: int
    n_clusters: int
    ci_lower: float
    ci_upper: float
    passes: bool


@dataclass
class ValidationResults:
    """
    Complete validation results across all sigma values.

    Attributes:
        results: Dict mapping percentile to RegressionResult
        headline: The p50 result
        y_variable: Name of dependent variable
        n_passing: How many sigma values pass (beta > 0, p < 0.10)
        overall_decision: "PASS", "FAIL", or "PARTIAL"
    """
    results: dict[str, RegressionResult]
    headline: RegressionResult
    y_variable: str
    n_passing: int
    overall_decision: str


@dataclass
class MonotonicityResult:
    """
    Binned relationship between overlap and outcome.

    Attributes:
        decile_means: Mean Y per overlap decile
        decile_sems: Standard error of mean per decile
        decile_centers: Center of each overlap bin
        decile_edges: Overlap bin edges
        monotonic: True if all decile means are increasing
        spearman_rho: Rank correlation of decile number with mean Y
        spearman_pvalue: P-value for Spearman correlation
    """
    decile_means: np.ndarray
    decile_sems: np.ndarray
    decile_centers: np.ndarray
    decile_edges: np.ndarray
    monotonic: bool
    spearman_rho: float
    spearman_pvalue: float


def build_validation_dataset(
    overlap_result: OverlapResult,
    comovement: WageComovement,
    crosswalk: OnetOesCrosswalk,
    measures: OccupationMeasures,
) -> ValidationDataset:
    """
    Merge overlap with validation target (wage comovement).

    Handles many-to-one O*NET→OES mapping by aggregating occupation measures
    before computing overlap at SOC level.

    Args:
        overlap_result: OverlapResult from compute_validation_overlap
        comovement: WageComovement from compute_wage_comovement
        crosswalk: OnetOesCrosswalk from build_onet_oes_crosswalk
        measures: OccupationMeasures for aggregation

    Returns:
        ValidationDataset with paired overlap and comovement data
    """
    # Step 1: Aggregate O*NET measures to SOC level
    agg_matrix, soc_codes = aggregate_occupation_measures(
        measures.occupation_matrix,
        measures.occupation_codes,
        crosswalk,
    )

    # Step 2: Find common SOC codes between aggregated measures and comovement
    comovement_set = set(comovement.occupation_codes)
    common_socs = [soc for soc in soc_codes if soc in comovement_set]

    if len(common_socs) < 10:
        raise ValueError(
            f"Too few common occupations: {len(common_socs)}. "
            "Check crosswalk and comovement data."
        )

    # Step 3: Build index mappings
    soc_to_agg_idx = {soc: i for i, soc in enumerate(soc_codes)}
    soc_to_com_idx = {soc: i for i, soc in enumerate(comovement.occupation_codes)}

    # Step 4: Recompute overlap for aggregated measures at common SOCs
    # Extract just the common SOC rows from aggregated matrix
    common_indices_agg = [soc_to_agg_idx[soc] for soc in common_socs]
    common_agg_matrix = agg_matrix[common_indices_agg]

    # Build kernel at same sigma
    from .distances import compute_activity_distances
    distances = compute_activity_distances(measures)
    kernel = build_kernel_matrix(distances, sigma=overlap_result.sigma, kernel_type="exponential")

    # Compute overlap for aggregated measures: O = rho @ K @ rho^T
    overlap_agg = common_agg_matrix @ kernel.matrix @ common_agg_matrix.T
    overlap_agg = (overlap_agg + overlap_agg.T) / 2  # Symmetrize

    # Step 5: Extract comovement for common SOCs
    common_indices_com = [soc_to_com_idx[soc] for soc in common_socs]
    comovement_sub = comovement.comovement_matrix[np.ix_(common_indices_com, common_indices_com)]

    # Step 6: Build pair-level dataset (upper triangle, excluding diagonal)
    n = len(common_socs)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            overlap_val = overlap_agg[i, j]
            comove_val = comovement_sub[i, j]

            # Skip if comovement is NaN
            if np.isnan(comove_val):
                continue

            pairs.append({
                "occ_i": common_socs[i],
                "occ_j": common_socs[j],
                "overlap": overlap_val,
                "y": comove_val,
            })

    pair_df = pd.DataFrame(pairs)

    return ValidationDataset(
        pair_data=pair_df,
        n_pairs=len(pair_df),
        n_occupations=len(common_socs),
        y_variable="wage_comovement",
    )


def _cluster_se(X: np.ndarray, y: np.ndarray, cluster_ids: np.ndarray) -> tuple[float, float, float]:
    """
    Compute OLS coefficient with clustered standard errors.

    Args:
        X: Regressor (1D array)
        y: Outcome (1D array)
        cluster_ids: Cluster identifiers

    Returns:
        Tuple of (beta, clustered_se, r_squared)
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
        score = cluster_X.T @ cluster_resid  # Sum of scores in cluster
        meat += np.outer(score, score)

    # Small-sample correction
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - 2))

    # Sandwich variance
    var_beta = correction * XtX_inv @ meat @ XtX_inv
    se_beta = np.sqrt(np.diag(var_beta))

    return beta_vec[1], se_beta[1], r_squared, n_clusters


def run_validation_regression(
    dataset: ValidationDataset,
    sigma_percentile: str,
    sigma: float,
) -> RegressionResult:
    """
    Run Diagnostic B2 regression with clustered SEs.

    Regression: Y_{i,j} = alpha + beta * Overlap_{i,j} + epsilon

    Standard errors clustered by origin occupation (occ_i).

    Args:
        dataset: ValidationDataset from build_validation_dataset
        sigma_percentile: Which percentile this sigma corresponds to
        sigma: Actual sigma value

    Returns:
        RegressionResult with coefficient, SE, p-value, etc.
    """
    df = dataset.pair_data

    X = df["overlap"].values
    y = df["y"].values
    clusters = df["occ_i"].values

    # Get unique cluster IDs
    unique_clusters, cluster_ids = np.unique(clusters, return_inverse=True)

    # Run regression with clustered SEs
    beta, se, r_squared, n_clusters = _cluster_se(X, y, cluster_ids)

    # Compute p-value (t-distribution with n_clusters - 1 df)
    t_stat = beta / se
    pvalue = 2 * (1 - sp_stats.t.cdf(np.abs(t_stat), df=n_clusters - 1))

    # 95% CI
    t_crit = sp_stats.t.ppf(0.975, df=n_clusters - 1)
    ci_lower = beta - t_crit * se
    ci_upper = beta + t_crit * se

    # Pass criterion: beta > 0 and p < 0.10
    passes = (beta > 0) and (pvalue < 0.10)

    return RegressionResult(
        sigma_percentile=sigma_percentile,
        sigma=sigma,
        beta=beta,
        se=se,
        pvalue=pvalue,
        r_squared=r_squared,
        n_pairs=len(df),
        n_clusters=n_clusters,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        passes=passes,
    )


def run_full_validation(
    overlap_grid: OverlapGrid,
    comovement: WageComovement,
    crosswalk: OnetOesCrosswalk,
    measures: OccupationMeasures,
) -> ValidationResults:
    """
    Run validation regression for all 5 sigma values.

    Args:
        overlap_grid: OverlapGrid from compute_overlap_grid
        comovement: WageComovement from compute_wage_comovement
        crosswalk: OnetOesCrosswalk from build_onet_oes_crosswalk
        measures: OccupationMeasures for aggregation

    Returns:
        ValidationResults with complete results and overall decision
    """
    results = {}

    for pct in SIGMA_PERCENTILES:
        overlap_result = overlap_grid.results[pct]

        # Build dataset for this sigma
        dataset = build_validation_dataset(
            overlap_result, comovement, crosswalk, measures
        )

        # Run regression
        reg_result = run_validation_regression(
            dataset,
            sigma_percentile=pct,
            sigma=overlap_result.sigma,
        )

        results[pct] = reg_result

    # Count passing sigmas
    n_passing = sum(1 for r in results.values() if r.passes)

    # Overall decision
    if n_passing == 5:
        overall_decision = "PASS"
    elif n_passing == 0:
        overall_decision = "FAIL"
    else:
        overall_decision = "PARTIAL"

    return ValidationResults(
        results=results,
        headline=results["p50"],
        y_variable="wage_comovement",
        n_passing=n_passing,
        overall_decision=overall_decision,
    )


def check_monotonicity(
    dataset: ValidationDataset,
    n_bins: int = 10,
) -> MonotonicityResult:
    """
    Bin overlap into quantiles and check monotonicity of Y.

    Args:
        dataset: ValidationDataset from build_validation_dataset
        n_bins: Number of bins (default: 10 for deciles)

    Returns:
        MonotonicityResult with binned means and monotonicity check
    """
    df = dataset.pair_data

    # Create overlap bins
    df = df.copy()
    df["bin"], bin_edges = pd.qcut(
        df["overlap"], q=n_bins, labels=False, retbins=True, duplicates="drop"
    )

    # Compute mean and SEM per bin
    grouped = df.groupby("bin")["y"]
    decile_means = grouped.mean().values
    decile_sems = grouped.sem().values
    decile_counts = grouped.count().values

    # Compute bin centers
    decile_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Adjust for dropped bins if duplicates were present
    actual_bins = len(decile_means)
    if actual_bins < n_bins:
        decile_centers = decile_centers[:actual_bins]

    # Check strict monotonicity
    monotonic = all(decile_means[i] <= decile_means[i + 1] for i in range(len(decile_means) - 1))

    # Spearman correlation of bin index with mean
    bin_indices = np.arange(len(decile_means))
    spearman_rho, spearman_pvalue = sp_stats.spearmanr(bin_indices, decile_means)

    return MonotonicityResult(
        decile_means=decile_means,
        decile_sems=decile_sems,
        decile_centers=decile_centers,
        decile_edges=bin_edges,
        monotonic=monotonic,
        spearman_rho=spearman_rho,
        spearman_pvalue=spearman_pvalue,
    )


def plot_monotonicity(
    result: MonotonicityResult,
    output_path: Path,
    title: str = "Overlap vs Wage Comovement",
) -> None:
    """
    Generate binned scatterplot with error bars.

    Args:
        result: MonotonicityResult from check_monotonicity
        output_path: Path to save the plot
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot error bars
    ax.errorbar(
        result.decile_centers,
        result.decile_means,
        yerr=1.96 * result.decile_sems,  # 95% CI
        fmt="o",
        capsize=4,
        capthick=1.5,
        markersize=8,
        color="steelblue",
        ecolor="gray",
    )

    # Fit and plot trend line
    slope, intercept = np.polyfit(result.decile_centers, result.decile_means, 1)
    x_line = np.linspace(result.decile_centers.min(), result.decile_centers.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "--", color="coral", alpha=0.8,
            label=f"Linear fit (slope={slope:.4f})")

    ax.set_xlabel("Overlap (decile bin centers)", fontsize=12)
    ax.set_ylabel("Wage Comovement (mean)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()

    # Add annotation
    mono_str = "Yes" if result.monotonic else "No"
    ax.annotate(
        f"Spearman ρ = {result.spearman_rho:.3f} (p = {result.spearman_pvalue:.3f})\n"
        f"Strictly monotonic: {mono_str}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_validation_results(
    results: ValidationResults,
    monotonicity: MonotonicityResult,
    output_dir: Path,
) -> None:
    """
    Save validation regression results to JSON.

    Args:
        results: ValidationResults from run_full_validation
        monotonicity: MonotonicityResult from check_monotonicity
        output_dir: Directory to save to
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output dict
    output = {
        "y_variable": results.y_variable,
        "overall_decision": results.overall_decision,
        "n_passing": results.n_passing,
        "headline_p50": {
            "beta": float(results.headline.beta),
            "se": float(results.headline.se),
            "pvalue": float(results.headline.pvalue),
            "r_squared": float(results.headline.r_squared),
            "ci_lower": float(results.headline.ci_lower),
            "ci_upper": float(results.headline.ci_upper),
            "n_pairs": int(results.headline.n_pairs),
            "n_clusters": int(results.headline.n_clusters),
            "passes": bool(results.headline.passes),
        },
        "sigma_grid": {
            pct: {
                "sigma": float(r.sigma),
                "beta": float(r.beta),
                "se": float(r.se),
                "pvalue": float(r.pvalue),
                "r_squared": float(r.r_squared),
                "passes": bool(r.passes),
            }
            for pct, r in results.results.items()
        },
        "monotonicity": {
            "strictly_monotonic": bool(monotonicity.monotonic),
            "spearman_rho": float(monotonicity.spearman_rho),
            "spearman_pvalue": float(monotonicity.spearman_pvalue),
            "n_bins": int(len(monotonicity.decile_means)),
            "decile_means": [float(x) for x in monotonicity.decile_means],
        },
    }

    with open(output_dir / "regression_results.json", "w") as f:
        json.dump(output, f, indent=2)
