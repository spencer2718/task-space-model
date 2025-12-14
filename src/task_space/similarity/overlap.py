"""
Overlap measures between occupation distributions.

Three measures:
1. Jaccard: Binary set overlap (baseline)
2. Kernel overlap: rho_i^T K rho_j (unnormalized)
3. Normalized overlap: Kernel overlap / sqrt(self-overlaps) (controls concentration)
"""

import numpy as np


def compute_jaccard_overlap(occ_measures: np.ndarray) -> np.ndarray:
    """
    Compute binary Jaccard overlap between occupations.

    Jaccard(i,j) = |A_i intersect A_j| / |A_i union A_j|

    Args:
        occ_measures: (n_occ, n_act) occupation measure matrix

    Returns:
        (n_occ, n_occ) Jaccard similarity matrix
    """
    binary = (occ_measures > 0).astype(float)

    # Intersection: element-wise min
    intersection = binary @ binary.T

    # Union: |A| + |B| - |A intersect B|
    support_sizes = binary.sum(axis=1)
    union = support_sizes[:, None] + support_sizes[None, :] - intersection

    # Avoid division by zero
    union = np.maximum(union, 1e-10)

    return intersection / union


def compute_kernel_overlap(
    occ_measures: np.ndarray,
    kernel_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute unnormalized kernel-weighted overlap.

    Overlap(i,j) = rho_i^T K rho_j

    This is RAW overlap. For concentration-controlled overlap,
    use compute_normalized_overlap().

    Args:
        occ_measures: (n_occ, n_act) occupation probability measures
        kernel_matrix: (n_act, n_act) kernel matrix (NOT row-normalized)

    Returns:
        (n_occ, n_occ) overlap matrix
    """
    return occ_measures @ kernel_matrix @ occ_measures.T


def compute_normalized_overlap(
    occ_measures: np.ndarray,
    kernel_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized (cosine-style) kernel overlap.

    NormOverlap(i,j) = (rho_i^T K rho_j) / sqrt((rho_i^T K rho_i)(rho_j^T K rho_j))

    This controls for concentration effects (specialist vs generalist).
    Per v0.6.2 findings, normalization IMPROVES R^2 by ~57%.

    Args:
        occ_measures: (n_occ, n_act) occupation probability measures
        kernel_matrix: (n_act, n_act) kernel matrix

    Returns:
        (n_occ, n_occ) normalized overlap matrix, values in [0, 1]
    """
    raw = compute_kernel_overlap(occ_measures, kernel_matrix)
    self_overlap = np.diag(raw)

    norm_factor = np.sqrt(np.outer(self_overlap, self_overlap))
    norm_factor = np.maximum(norm_factor, 1e-10)

    result = raw / norm_factor
    # Clip to [0, 1] to handle numerical precision issues
    return np.clip(result, 0.0, 1.0)
