"""
Phase I coherence diagnostics.

Implements diagnostic checks for occupation measures and activity geometry
per paper Section 4.3 (Phase I Validation).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .domain import ActivityDomain, OccupationMeasures
from .distances import ActivityDistances


@dataclass
class MeasureCoherence:
    """
    Diagnostic A: Occupation measure coherence statistics.

    Attributes:
        entropy: Entropy H(rho_j) for each occupation
        effective_support: Number of activities with rho_j(a) > threshold
        median_entropy: Median entropy across occupations
        median_support: Median effective support
        sparse_occupations: List of occupation codes with low support
        coverage: Fraction of occupations with adequate support
    """
    entropy: np.ndarray
    effective_support: np.ndarray
    median_entropy: float
    median_support: float
    sparse_occupations: list[str]
    coverage: float


def compute_entropy(rho: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute entropy H(rho) = -sum(rho * log(rho)) for each row.

    Args:
        rho: Probability matrix, shape (n_occupations, n_activities)
        eps: Small constant to avoid log(0)

    Returns:
        Entropy values, shape (n_occupations,)
    """
    # Clip to avoid log(0)
    rho_safe = np.clip(rho, eps, 1.0)
    # Only sum where rho > 0 to avoid -0 * inf issues
    mask = rho > eps
    log_rho = np.zeros_like(rho)
    log_rho[mask] = np.log(rho_safe[mask])
    entropy = -np.sum(rho * log_rho, axis=1)
    return entropy


def compute_effective_support(rho: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Compute effective support: number of activities with rho_j(a) > threshold.

    Args:
        rho: Probability matrix, shape (n_occupations, n_activities)
        threshold: Minimum weight to count as "in support"

    Returns:
        Effective support counts, shape (n_occupations,)
    """
    return (rho > threshold).sum(axis=1)


def diagnose_measure_coherence(
    occupation_measures: OccupationMeasures,
    support_threshold: float = 0.01,
    min_support: int = 15,
) -> MeasureCoherence:
    """
    Diagnostic A: Check coherence of occupation measures.

    Checks:
    - A1: Sparsity and entropy distribution
    - A2: Effective support distribution

    Args:
        occupation_measures: OccupationMeasures object
        support_threshold: Threshold for counting activity in support
        min_support: Minimum acceptable effective support

    Returns:
        MeasureCoherence with diagnostic statistics.

    Paper reference: Section 4.3 Diagnostic A
    """
    rho = occupation_measures.occupation_matrix
    occupation_codes = occupation_measures.occupation_codes

    entropy = compute_entropy(rho)
    effective_support = compute_effective_support(rho, support_threshold)

    median_entropy = float(np.median(entropy))
    median_support = float(np.median(effective_support))

    # Find sparse occupations
    sparse_mask = effective_support < min_support
    sparse_occupations = [
        occupation_codes[i] for i in range(len(occupation_codes)) if sparse_mask[i]
    ]

    coverage = 1.0 - (sparse_mask.sum() / len(occupation_codes))

    return MeasureCoherence(
        entropy=entropy,
        effective_support=effective_support,
        median_entropy=median_entropy,
        median_support=median_support,
        sparse_occupations=sparse_occupations,
        coverage=coverage,
    )


@dataclass
class FaceValidityCheck:
    """
    Face validity spot check for occupation measures.

    Attributes:
        occupation_code: O*NET-SOC code
        top_activities: List of (activity_id, activity_name, weight) tuples
    """
    occupation_code: str
    top_activities: list[tuple[str, str, float]]


def spot_check_occupation(
    occupation_measures: OccupationMeasures,
    activity_domain: ActivityDomain,
    occupation_code: str,
    top_k: int = 5,
) -> FaceValidityCheck:
    """
    Diagnostic A2: Face validity check for a single occupation.

    Lists top-k activities by weight for manual inspection.

    Args:
        occupation_measures: OccupationMeasures object
        activity_domain: ActivityDomain with activity names
        occupation_code: O*NET-SOC code to check
        top_k: Number of top activities to return

    Returns:
        FaceValidityCheck with top activities.
    """
    if occupation_code not in occupation_measures.occupation_codes:
        raise ValueError(f"Occupation {occupation_code} not found")

    idx = occupation_measures.occupation_codes.index(occupation_code)
    weights = occupation_measures.occupation_matrix[idx]
    activity_ids = occupation_measures.activity_ids

    # Sort by weight descending
    sorted_indices = np.argsort(weights)[::-1][:top_k]

    top_activities = []
    for i in sorted_indices:
        act_id = activity_ids[i]
        act_name = activity_domain.activity_names.get(act_id, act_id)
        weight = weights[i]
        top_activities.append((act_id, act_name, float(weight)))

    return FaceValidityCheck(
        occupation_code=occupation_code,
        top_activities=top_activities,
    )


def spot_check_occupations(
    occupation_measures: OccupationMeasures,
    activity_domain: ActivityDomain,
    occupation_codes: list[str],
    top_k: int = 5,
) -> list[FaceValidityCheck]:
    """
    Run face validity checks for multiple occupations.

    Args:
        occupation_measures: OccupationMeasures object
        activity_domain: ActivityDomain with activity names
        occupation_codes: List of O*NET-SOC codes to check
        top_k: Number of top activities per occupation

    Returns:
        List of FaceValidityCheck results.
    """
    results = []
    for code in occupation_codes:
        if code in occupation_measures.occupation_codes:
            result = spot_check_occupation(
                occupation_measures, activity_domain, code, top_k
            )
            results.append(result)
    return results


@dataclass
class DistanceCoherence:
    """
    Activity distance coherence check.

    Attributes:
        neighbor_stability: Fraction of nearest-neighbor sets unchanged under perturbation
        distance_stats: Dict with distance distribution statistics
    """
    neighbor_stability: float
    distance_stats: dict[str, float]


def check_distance_stability(
    distances: ActivityDistances,
    perturbation_scale: float = 0.1,
    k_neighbors: int = 5,
    n_trials: int = 10,
) -> float:
    """
    Check stability of nearest-neighbor sets under perturbation.

    Perturbs activity profiles and checks how often k-NN sets change.

    Args:
        distances: ActivityDistances object
        perturbation_scale: Scale of Gaussian noise to add (fraction of std)
        k_neighbors: Number of neighbors to check
        n_trials: Number of perturbation trials

    Returns:
        Fraction of (activity, trial) pairs where k-NN set is unchanged.
    """
    profiles = distances.activity_profiles
    n_activities = profiles.shape[0]

    # Get original k-NN sets
    D_original = distances.distance_matrix
    original_neighbors = []
    for i in range(n_activities):
        dists = D_original[i]
        sorted_idx = np.argsort(dists)
        # Exclude self (distance 0)
        neighbors = set(sorted_idx[1:k_neighbors+1])
        original_neighbors.append(neighbors)

    # Run perturbation trials
    stable_count = 0
    total_count = n_activities * n_trials

    profile_std = profiles.std()

    for _ in range(n_trials):
        # Perturb profiles
        noise = np.random.randn(*profiles.shape) * perturbation_scale * profile_std
        perturbed = profiles + noise

        # Recompute distances
        from scipy.spatial.distance import pdist, squareform
        D_perturbed = squareform(pdist(perturbed, metric="euclidean"))

        # Check k-NN stability
        for i in range(n_activities):
            dists = D_perturbed[i]
            sorted_idx = np.argsort(dists)
            new_neighbors = set(sorted_idx[1:k_neighbors+1])
            if new_neighbors == original_neighbors[i]:
                stable_count += 1

    return stable_count / total_count


def diagnose_distances(
    distances: ActivityDistances,
    check_stability: bool = True,
) -> DistanceCoherence:
    """
    Diagnostic for activity distance matrix.

    Args:
        distances: ActivityDistances object
        check_stability: If True, run perturbation stability check (slow)

    Returns:
        DistanceCoherence with statistics.
    """
    from .distances import distance_percentiles

    stats = distance_percentiles(distances)

    if check_stability:
        stability = check_distance_stability(distances)
    else:
        stability = -1.0  # Not computed

    return DistanceCoherence(
        neighbor_stability=stability,
        distance_stats=stats,
    )


def generate_diagnostic_report(
    occupation_measures: OccupationMeasures,
    activity_domain: ActivityDomain,
    distances: ActivityDistances,
    sample_occupations: Optional[list[str]] = None,
) -> str:
    """
    Generate a text diagnostic report for Phase I.

    Args:
        occupation_measures: OccupationMeasures object
        activity_domain: ActivityDomain object
        distances: ActivityDistances object
        sample_occupations: List of occupation codes for face validity check

    Returns:
        Formatted diagnostic report string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PHASE I DIAGNOSTIC REPORT")
    lines.append("=" * 60)

    # Measure coherence
    coherence = diagnose_measure_coherence(occupation_measures)
    lines.append("\n[A] OCCUPATION MEASURE COHERENCE")
    lines.append("-" * 40)
    lines.append(f"  Median entropy: {coherence.median_entropy:.3f}")
    lines.append(f"  Median effective support: {coherence.median_support:.1f}")
    lines.append(f"  Coverage (support >= 15): {coherence.coverage:.1%}")
    lines.append(f"  Sparse occupations: {len(coherence.sparse_occupations)}")

    # Distance statistics
    dist_coherence = diagnose_distances(distances, check_stability=False)
    lines.append("\n[B] ACTIVITY DISTANCE STATISTICS")
    lines.append("-" * 40)
    for key, val in dist_coherence.distance_stats.items():
        lines.append(f"  {key}: {val:.4f}")

    # Face validity spot checks
    if sample_occupations:
        lines.append("\n[C] FACE VALIDITY SPOT CHECKS")
        lines.append("-" * 40)
        checks = spot_check_occupations(
            occupation_measures, activity_domain, sample_occupations, top_k=5
        )
        for check in checks:
            lines.append(f"\n  {check.occupation_code}:")
            for act_id, act_name, weight in check.top_activities:
                lines.append(f"    {weight:.3f}  {act_name[:50]}")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)
