"""
Wasserstein distance between occupation task distributions.

This module computes exact Earth Mover's Distance (W₁) between occupation
measures using the POT library. Key optimization: exploit 99% sparsity by
extracting ~40×40 subproblems from the 2087×2087 ground metric.

Novel contribution: No prior economics literature applies Wasserstein to
occupation-task distributions. See paper Section 3.6.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class WassersteinDistanceResult:
    """Result container for Wasserstein distance computation."""
    distance_matrix: np.ndarray  # (n_occ, n_occ) pairwise W₁ distances
    n_occupations: int
    n_activities: int
    ground_metric: str  # e.g., "cosine_mpnet", "cosine_jobbert"
    computation_time_seconds: float
    median_support_size: float  # Median |support_union| across pairs
    assumptions: List[str] = field(default_factory=lambda: [
        "W₁ (Earth Mover's Distance) with linear cost",
        "Ground metric: cosine distance from text embeddings",
        "Sparsity exploited: ~40×40 subproblems from 2087×2087",
        "Exact solution via network simplex (POT)",
    ])


def compute_wasserstein_distances(
    occ_measures: np.ndarray,
    ground_distance: np.ndarray,
    n_jobs: int = -1,
    normalize_ground: bool = False,
    verbose: bool = True,
) -> WassersteinDistanceResult:
    """
    Compute pairwise Wasserstein distances between occupation measures.

    Parameters
    ----------
    occ_measures : np.ndarray, shape (n_occ, n_act)
        Occupation probability measures over activities. Rows should sum to 1.
    ground_distance : np.ndarray, shape (n_act, n_act)
        Ground metric (cost matrix) between activities.
    n_jobs : int
        Number of parallel jobs (-1 for all cores).
    normalize_ground : bool
        If True, normalize ground_distance to [0, 1].
    verbose : bool
        Print progress.

    Returns
    -------
    WassersteinDistanceResult
        Contains distance_matrix and metadata.
    """
    import ot
    from joblib import Parallel, delayed
    from itertools import combinations
    import time

    n_occ, n_act = occ_measures.shape
    if ground_distance.shape != (n_act, n_act):
        raise ValueError(
            f"Ground distance shape {ground_distance.shape} doesn't match "
            f"number of activities {n_act}"
        )

    # Normalize ground metric if requested
    M = ground_distance.copy()
    if normalize_ground:
        M = M / M.max()

    # Pre-compute supports for all occupations (indices where mass > 0)
    supports = [np.where(occ_measures[i] > 1e-10)[0] for i in range(n_occ)]

    def sparse_emd(i: int, j: int) -> Tuple[int, int, float, int]:
        """Compute EMD between occupations i and j using sparse supports."""
        supp_i, supp_j = supports[i], supports[j]
        combined = np.union1d(supp_i, supp_j)

        # Extract mass on combined support
        a = occ_measures[i, combined]
        b = occ_measures[j, combined]

        # Renormalize to ensure they sum to exactly 1
        a = a / a.sum()
        b = b / b.sum()

        # Extract submatrix of ground metric
        M_sub = M[np.ix_(combined, combined)]

        # Compute exact EMD using network simplex
        dist = ot.emd2(a, b, M_sub, numThreads=1)

        return i, j, dist, len(combined)

    # Generate all pairs
    pairs = list(combinations(range(n_occ), 2))
    n_pairs = len(pairs)

    if verbose:
        print(f"Computing {n_pairs:,} pairwise Wasserstein distances...")
        print(f"Occupations: {n_occ}, Activities: {n_act}")

    start_time = time.time()

    # Parallel computation
    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10 if verbose else 0)(
        delayed(sparse_emd)(i, j) for i, j in pairs
    )

    elapsed = time.time() - start_time

    # Build symmetric distance matrix
    D = np.zeros((n_occ, n_occ))
    support_sizes = []
    for i, j, dist, supp_size in results:
        D[i, j] = D[j, i] = dist
        support_sizes.append(supp_size)

    if verbose:
        print(f"Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Median support union size: {np.median(support_sizes):.0f}")

    return WassersteinDistanceResult(
        distance_matrix=D,
        n_occupations=n_occ,
        n_activities=n_act,
        ground_metric="cosine",
        computation_time_seconds=elapsed,
        median_support_size=float(np.median(support_sizes)),
    )


def wasserstein_to_similarity(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Convert Wasserstein distance to similarity measure.

    Uses exponential decay: sim(i,j) = exp(-d(i,j) / median(d))

    This matches the kernel overlap similarity structure for comparability.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise Wasserstein distances.

    Returns
    -------
    np.ndarray
        Similarity matrix with values in (0, 1], diagonal = 1.
    """
    # Get median of non-zero distances for scale
    nonzero_dists = distance_matrix[distance_matrix > 0]
    if len(nonzero_dists) == 0:
        # All identical distributions
        return np.ones_like(distance_matrix)

    median_dist = np.median(nonzero_dists)
    return np.exp(-distance_matrix / median_dist)
