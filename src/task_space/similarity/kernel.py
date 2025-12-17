"""
Kernel construction with correct defaults.

CRITICAL LESSONS FROM PHASE 1:
1. sigma must be calibrated to nearest-neighbor distances, NOT global distribution
2. Do NOT row-normalize the kernel matrix
3. Discrimination ratio must be > 3x to avoid collapse
"""

import warnings
import numpy as np


def calibrate_sigma(
    dist_matrix: np.ndarray,
    method: str | float = 'nn_median',
) -> float:
    """
    Calibrate kernel bandwidth to local distance structure.

    Args:
        dist_matrix: (n, n) pairwise distance matrix
        method: Either 'nn_median' for data-driven calibration,
                or a float for fixed bandwidth value

    Returns:
        Calibrated sigma value

    Raises:
        ValueError: If method is a string other than 'nn_median'
    """
    # Allow fixed numeric values for multiverse analysis
    if isinstance(method, (int, float)):
        return float(method)

    if method != 'nn_median':
        raise ValueError(
            f"Method '{method}' is not supported. Use 'nn_median' or a numeric value. "
            "Global percentile methods cause kernel collapse (see v0.5.0 postmortem). "
            "The NN-median method is the only validated approach."
        )

    dm = dist_matrix.copy()
    np.fill_diagonal(dm, np.inf)
    nn_dists = dm.min(axis=1)
    return float(np.median(nn_dists))


def check_kernel_discrimination(
    dist_matrix: np.ndarray,
    sigma: float,
    min_ratio: float = 3.0,
) -> tuple[float, bool]:
    """
    Check if kernel discriminates between close and distant pairs.

    Args:
        dist_matrix: (n, n) pairwise distance matrix
        sigma: Kernel bandwidth
        min_ratio: Minimum acceptable discrimination ratio

    Returns:
        (discrimination_ratio, is_acceptable)
    """
    d_flat = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    d_p10 = np.percentile(d_flat, 10)
    d_p90 = np.percentile(d_flat, 90)

    w_p10 = np.exp(-d_p10 / sigma)
    w_p90 = np.exp(-d_p90 / sigma)

    ratio = w_p10 / w_p90 if w_p90 > 0 else np.inf
    return ratio, ratio >= min_ratio


def build_kernel_matrix(
    dist_matrix: np.ndarray,
    sigma: float = None,
    kernel_type: str = 'exponential',
    row_normalize: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Build kernel matrix from distance matrix.

    Args:
        dist_matrix: (n, n) pairwise distance matrix
        sigma: Bandwidth. If None, auto-calibrate via NN-median.
        kernel_type: 'exponential' or 'gaussian'
        row_normalize: Whether to row-normalize.
            DEFAULT IS FALSE. This is intentional.

    Returns:
        (kernel_matrix, sigma_used)

    Warnings:
        Emits UserWarning if row_normalize=True
    """
    if sigma is None:
        sigma = calibrate_sigma(dist_matrix)

    # Check discrimination
    ratio, ok = check_kernel_discrimination(dist_matrix, sigma)
    if not ok:
        warnings.warn(
            f"Kernel discrimination ratio ({ratio:.2f}) is below 3.0. "
            "This may indicate kernel collapse. Consider smaller sigma.",
            UserWarning
        )

    # Build kernel
    if kernel_type == 'exponential':
        K = np.exp(-dist_matrix / sigma)
    elif kernel_type == 'gaussian':
        K = np.exp(-dist_matrix**2 / (2 * sigma**2))
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    if row_normalize:
        warnings.warn(
            "Row normalization can destroy signal with large activity counts. "
            "This was the root cause of kernel collapse in v0.5.0. "
            "Only use row_normalize=True if you understand the consequences.",
            UserWarning
        )
        row_sums = K.sum(axis=1, keepdims=True)
        K = K / row_sums

    return K, sigma
