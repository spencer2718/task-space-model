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

from .distances import ActivityDistances, distance_percentiles
from .domain import OccupationMeasures
from .kernel import build_kernel_matrix, compute_overlap


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
