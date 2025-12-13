"""
Kernel matrix, shock propagation, and exposure computation.

Implements the normalized spillover operator (Definition: Spillover Operator)
and occupation-level exposure functionals (Definition: Exposure Functionals).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .distances import ActivityDistances
from .domain import OccupationMeasures


@dataclass
class KernelMatrix:
    """
    Row-normalized kernel matrix for shock propagation.

    Attributes:
        matrix: Row-normalized kernel matrix K, shape (n_activities, n_activities)
                K[a,b] = k(d(a,b); sigma) / sum_c k(d(a,c); sigma)
        sigma: Bandwidth parameter
        kernel_type: 'exponential' or 'gaussian'
        activity_ids: Activity IDs (row/column labels)
    """
    matrix: np.ndarray  # Shape: (n_activities, n_activities)
    sigma: float
    kernel_type: str
    activity_ids: list[str]


def build_kernel_matrix(
    distances: ActivityDistances,
    sigma: float,
    kernel_type: str = "exponential",
) -> KernelMatrix:
    """
    Build row-normalized kernel matrix from activity distances.

    Implements the normalized spillover operator (Definition: Spillover Operator):
        K[a,b] = k(d(a,b); sigma) / sum_c k(d(a,c); sigma)

    Args:
        distances: ActivityDistances object with pairwise distance matrix
        sigma: Bandwidth parameter controlling spillover range
        kernel_type: 'exponential' for k(d) = exp(-d/sigma)
                     'gaussian' for k(d) = exp(-d^2 / (2*sigma^2))

    Returns:
        KernelMatrix with row-normalized kernel.

    Paper reference: Definition (Spillover Operator), Equation for exponential kernel
    """
    D = distances.distance_matrix
    n = D.shape[0]

    # Compute raw kernel values
    if kernel_type == "exponential":
        K_raw = np.exp(-D / sigma)
    elif kernel_type == "gaussian":
        K_raw = np.exp(-D**2 / (2 * sigma**2))
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    # Row-normalize
    row_sums = K_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    K_normalized = K_raw / row_sums

    return KernelMatrix(
        matrix=K_normalized,
        sigma=sigma,
        kernel_type=kernel_type,
        activity_ids=distances.activity_ids,
    )


def propagate_shock(
    kernel: KernelMatrix,
    shock_profile: np.ndarray,
) -> np.ndarray:
    """
    Propagate a shock through the kernel (spillover operator).

    Implements: A = K @ I
    where K is the kernel matrix and I is the shock profile.

    Args:
        kernel: KernelMatrix object
        shock_profile: Shock intensities at each activity, shape (n_activities,)

    Returns:
        Propagated displacement field A, shape (n_activities,)

    Paper reference: Definition (Baseline Exposure Construction)
    """
    if len(shock_profile) != kernel.matrix.shape[0]:
        raise ValueError(
            f"Shock profile length {len(shock_profile)} != "
            f"kernel dimension {kernel.matrix.shape[0]}"
        )

    return kernel.matrix @ shock_profile


def compute_exposure(
    occupation_measures: OccupationMeasures,
    displacement_field: np.ndarray,
) -> np.ndarray:
    """
    Compute occupation-level exposure from displacement field.

    Implements: E_j = rho_j @ A = sum_a rho_j(a) * A(a)
    where rho_j is the occupation measure and A is the displacement field.

    Args:
        occupation_measures: OccupationMeasures with rho_j matrix
        displacement_field: Propagated displacement A, shape (n_activities,)

    Returns:
        Exposure values E_j for each occupation, shape (n_occupations,)

    Paper reference: Definition (Exposure Functionals)
    """
    # rho: (n_occupations, n_activities)
    # A: (n_activities,)
    # E = rho @ A: (n_occupations,)
    return occupation_measures.occupation_matrix @ displacement_field


@dataclass
class ExposureResult:
    """
    Complete exposure computation result.

    Attributes:
        occupation_codes: List of O*NET-SOC codes
        exposures: Exposure values E_j for each occupation
        displacement_field: Propagated displacement field A(a)
        shock_profile: Original shock profile I(a)
        activity_ids: Activity IDs
    """
    occupation_codes: list[str]
    exposures: np.ndarray
    displacement_field: np.ndarray
    shock_profile: np.ndarray
    activity_ids: list[str]


def compute_occupation_exposure(
    occupation_measures: OccupationMeasures,
    kernel: KernelMatrix,
    shock_profile: np.ndarray,
) -> ExposureResult:
    """
    Full exposure computation pipeline: shock → propagation → occupation exposure.

    Args:
        occupation_measures: OccupationMeasures with rho_j matrix
        kernel: KernelMatrix for propagation
        shock_profile: Shock intensities I(a) at each activity

    Returns:
        ExposureResult with exposures and intermediate values.
    """
    # Propagate shock through kernel
    displacement_field = propagate_shock(kernel, shock_profile)

    # Compute occupation exposures
    exposures = compute_exposure(occupation_measures, displacement_field)

    return ExposureResult(
        occupation_codes=occupation_measures.occupation_codes,
        exposures=exposures,
        displacement_field=displacement_field,
        shock_profile=shock_profile,
        activity_ids=kernel.activity_ids,
    )


def create_shock_profile(
    activity_ids: list[str],
    target_activities: dict[str, float],
    default_value: float = 0.0,
) -> np.ndarray:
    """
    Create a shock profile from target activities and intensities.

    Args:
        activity_ids: List of all activity IDs (defines ordering)
        target_activities: Dict mapping activity ID to shock intensity
        default_value: Value for non-targeted activities

    Returns:
        Shock profile array, shape (n_activities,)
    """
    profile = np.full(len(activity_ids), default_value)

    for act_id, intensity in target_activities.items():
        if act_id in activity_ids:
            idx = activity_ids.index(act_id)
            profile[idx] = intensity

    return profile


def compute_overlap(
    occupation_measures: OccupationMeasures,
    kernel: KernelMatrix,
) -> np.ndarray:
    """
    Compute pairwise occupation overlap through kernel.

    Overlap(i,j) = rho_i^T @ K @ rho_j

    Occupation pairs with high overlap should exhibit stronger outcome comovement
    (paper Remark: Covariance Structure).

    Args:
        occupation_measures: OccupationMeasures with rho_j matrix
        kernel: KernelMatrix

    Returns:
        Overlap matrix, shape (n_occupations, n_occupations)
    """
    rho = occupation_measures.occupation_matrix  # (n_occ, n_act)
    K = kernel.matrix  # (n_act, n_act)

    # Overlap = rho @ K @ rho^T
    overlap = rho @ K @ rho.T

    return overlap
