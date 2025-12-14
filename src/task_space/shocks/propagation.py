"""
Shock propagation: I_t → A_t → E_j

Mathematical pipeline (from Theory section):
1. I_t: (n_act,) shock profile over activities
2. A_t = K @ I_t: (n_act,) propagated/displaced field
3. E_j = ρ_j @ A_t: (n_occ,) occupation exposure vector

DESIGN NOTE: This function takes a pre-computed kernel matrix K.
The kernel already encodes σ, so we do not pass σ separately.
If you need different σ, rebuild K first.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PropagationResult:
    """Results from shock propagation."""
    I_t: np.ndarray          # (n_act,) raw shock profile
    A_t: np.ndarray          # (n_act,) propagated field
    E: np.ndarray            # (n_occ,) occupation exposures
    shock_name: str          # Name of shock profile


def propagate_shock(
    I_t: np.ndarray,
    kernel_matrix: np.ndarray,
    occ_measures: np.ndarray,
    shock_name: str = "unknown",
) -> PropagationResult:
    """
    Propagate shock through task space to occupation exposures.

    Pipeline:
        I_t → A_t = K @ I_t → E = ρ @ A_t

    Args:
        I_t: (n_act,) shock profile over activities
        kernel_matrix: (n_act, n_act) kernel matrix (NOT row-normalized).
            Must be pre-computed with desired σ via build_kernel_matrix().
        occ_measures: (n_occ, n_act) occupation probability measures
        shock_name: Name of shock profile (for metadata)

    Returns:
        PropagationResult with I_t, A_t, E, and metadata

    Note:
        σ is NOT a parameter here. The kernel already encodes σ.
        To use different σ, rebuild the kernel matrix first.
    """
    # Propagate shock through kernel
    A_t = kernel_matrix @ I_t

    # Aggregate to occupation-level exposure
    E = occ_measures @ A_t

    return PropagationResult(
        I_t=I_t,
        A_t=A_t,
        E=E,
        shock_name=shock_name,
    )


def compute_exposure_from_shock(
    domain,
    occ_measures: np.ndarray,
    shock_name: str,
    shock_args: dict,
    kernel_matrix: np.ndarray,
) -> PropagationResult:
    """
    Compute occupation exposures from a registered shock profile.

    Convenience function that:
    1. Retrieves shock from registry
    2. Computes I_t
    3. Propagates to E

    Args:
        domain: Activity domain
        occ_measures: (n_occ, n_act) occupation measures
        shock_name: Registry key for shock profile
        shock_args: Arguments to pass to shock function
        kernel_matrix: (n_act, n_act) pre-computed kernel

    Returns:
        PropagationResult
    """
    from .registry import get_shock

    shock = get_shock(shock_name)
    I_t = shock.fn(domain, **shock_args)

    return propagate_shock(I_t, kernel_matrix, occ_measures, shock_name)


def exposure_stats(E: np.ndarray) -> dict:
    """
    Compute summary statistics for exposure vector.

    Args:
        E: (n_occ,) occupation exposure vector

    Returns:
        Dict with min, max, mean, std, p10, p50, p90
    """
    return {
        'min': float(E.min()),
        'max': float(E.max()),
        'mean': float(E.mean()),
        'std': float(E.std()),
        'p10': float(np.percentile(E, 10)),
        'p50': float(np.percentile(E, 50)),
        'p90': float(np.percentile(E, 90)),
    }
