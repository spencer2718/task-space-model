"""
Kernel and similarity diagnostics.

Diagnostic checks to detect kernel collapse and verify signal.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DistanceDiagnostics:
    """Diagnostics for distance distribution."""
    n_activities: int
    n_pairs: int
    percentiles: dict[str, float]
    nn_percentiles: dict[str, float]
    degenerate: bool
    degenerate_reason: Optional[str]


@dataclass
class KernelDiagnostics:
    """Diagnostics for kernel matrix."""
    sigma: float
    discrimination_ratio: float
    discrimination_ok: bool
    weight_percentiles: dict[str, float]
    collapsed: bool
    collapsed_reason: Optional[str]


@dataclass
class ValidationDiagnostics:
    """Combined diagnostics report."""
    distance: DistanceDiagnostics
    kernel: KernelDiagnostics
    recommendation: str


def diagnose_distances(dist_matrix: np.ndarray) -> DistanceDiagnostics:
    """
    Diagnose distance matrix for issues.

    Checks for:
    - Zero or near-zero distances (identical embeddings)
    - Uniform distances (no structure)
    - Extreme concentration
    """
    n = dist_matrix.shape[0]
    n_pairs = n * (n - 1) // 2

    # Get pairwise distances
    triu_idx = np.triu_indices(n, k=1)
    pairwise = dist_matrix[triu_idx]

    # Percentiles
    percentiles = {
        'min': float(pairwise.min()),
        'p10': float(np.percentile(pairwise, 10)),
        'p25': float(np.percentile(pairwise, 25)),
        'p50': float(np.percentile(pairwise, 50)),
        'p75': float(np.percentile(pairwise, 75)),
        'p90': float(np.percentile(pairwise, 90)),
        'max': float(pairwise.max()),
        'mean': float(pairwise.mean()),
        'std': float(pairwise.std()),
    }

    # Nearest neighbor distances
    dm_copy = dist_matrix.copy()
    np.fill_diagonal(dm_copy, np.inf)
    nn_dists = dm_copy.min(axis=1)

    nn_percentiles = {
        'min': float(nn_dists.min()),
        'p10': float(np.percentile(nn_dists, 10)),
        'p25': float(np.percentile(nn_dists, 25)),
        'p50': float(np.percentile(nn_dists, 50)),
        'p75': float(np.percentile(nn_dists, 75)),
        'p90': float(np.percentile(nn_dists, 90)),
        'max': float(nn_dists.max()),
    }

    # Check for degeneracy
    degenerate = False
    reason = None

    if percentiles['min'] < 1e-10:
        degenerate = True
        reason = "Near-zero minimum distance (possible duplicates)"
    elif percentiles['std'] / percentiles['mean'] < 0.1:
        degenerate = True
        reason = "Low variance in distances (uniform structure)"

    return DistanceDiagnostics(
        n_activities=n,
        n_pairs=n_pairs,
        percentiles=percentiles,
        nn_percentiles=nn_percentiles,
        degenerate=degenerate,
        degenerate_reason=reason,
    )


def diagnose_kernel(
    dist_matrix: np.ndarray,
    sigma: float,
) -> KernelDiagnostics:
    """
    Diagnose kernel matrix for collapse.

    Checks discrimination ratio and weight distribution.
    """
    n = dist_matrix.shape[0]

    # Build kernel
    K = np.exp(-dist_matrix / sigma)

    # Discrimination ratio
    triu_idx = np.triu_indices(n, k=1)
    pairwise = dist_matrix[triu_idx]
    d_p10 = np.percentile(pairwise, 10)
    d_p90 = np.percentile(pairwise, 90)

    w_p10 = np.exp(-d_p10 / sigma)
    w_p90 = np.exp(-d_p90 / sigma)
    ratio = w_p10 / w_p90 if w_p90 > 0 else np.inf

    # Weight percentiles (off-diagonal)
    K_offdiag = K[triu_idx]
    weight_percentiles = {
        'min': float(K_offdiag.min()),
        'p10': float(np.percentile(K_offdiag, 10)),
        'p25': float(np.percentile(K_offdiag, 25)),
        'p50': float(np.percentile(K_offdiag, 50)),
        'p75': float(np.percentile(K_offdiag, 75)),
        'p90': float(np.percentile(K_offdiag, 90)),
        'max': float(K_offdiag.max()),
    }

    # Check for collapse
    collapsed = False
    reason = None

    if ratio < 2.0:
        collapsed = True
        reason = f"Discrimination ratio ({ratio:.2f}) is below 2.0 - kernel cannot distinguish close from distant"
    elif weight_percentiles['p90'] / weight_percentiles['p10'] < 1.5:
        collapsed = True
        reason = "Weight range is too narrow - all weights are nearly equal"

    return KernelDiagnostics(
        sigma=sigma,
        discrimination_ratio=ratio,
        discrimination_ok=ratio >= 3.0,
        weight_percentiles=weight_percentiles,
        collapsed=collapsed,
        collapsed_reason=reason,
    )


def run_diagnostics(
    dist_matrix: np.ndarray,
    sigma: float = None,
) -> ValidationDiagnostics:
    """
    Run full diagnostic suite.

    Args:
        dist_matrix: (n, n) distance matrix
        sigma: Kernel bandwidth. If None, uses NN median.

    Returns:
        ValidationDiagnostics with distance, kernel, and recommendation.
    """
    from ..similarity.kernel import calibrate_sigma

    # Distance diagnostics
    dist_diag = diagnose_distances(dist_matrix)

    # Auto-calibrate sigma if not provided
    if sigma is None:
        sigma = calibrate_sigma(dist_matrix)

    # Kernel diagnostics
    kernel_diag = diagnose_kernel(dist_matrix, sigma)

    # Recommendation
    if dist_diag.degenerate:
        rec = f"WARNING: Distance matrix is degenerate - {dist_diag.degenerate_reason}"
    elif kernel_diag.collapsed:
        rec = f"WARNING: Kernel is collapsed - {kernel_diag.collapsed_reason}"
    elif not kernel_diag.discrimination_ok:
        rec = f"CAUTION: Low discrimination ratio ({kernel_diag.discrimination_ratio:.2f}). Consider smaller sigma."
    else:
        rec = f"OK: sigma={sigma:.4f}, discrimination={kernel_diag.discrimination_ratio:.2f}x"

    return ValidationDiagnostics(
        distance=dist_diag,
        kernel=kernel_diag,
        recommendation=rec,
    )
