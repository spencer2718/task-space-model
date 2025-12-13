"""
Activity distance computation (Recipe X).

Implements activity-to-activity distances using the rating-cooccurrence geometry
described in paper Section 4.1.1.

Recipe X: Represent each activity by its importance profile across occupations,
apply PCA for dimensionality reduction, compute Euclidean distances.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .domain import OccupationMeasures


@dataclass
class ActivityDistances:
    """
    Pairwise distances between activities (Definition: Admissible Task Distance).

    Attributes:
        distance_matrix: Pairwise Euclidean distances, shape (n_activities, n_activities)
        activity_ids: List of activity IDs (row/column labels)
        activity_profiles: Standardized activity profiles used for distance computation
        pca_variance_explained: Fraction of variance explained by PCA (if applied)
        n_components: Number of PCA components used (None if no PCA)
    """
    distance_matrix: np.ndarray  # Shape: (n_activities, n_activities)
    activity_ids: list[str]
    activity_profiles: np.ndarray  # Shape: (n_activities, n_features)
    pca_variance_explained: Optional[float]
    n_components: Optional[int]


def compute_activity_distances(
    occupation_measures: OccupationMeasures,
    n_components: Optional[int] = None,
    variance_threshold: float = 0.9,
    standardize: bool = True,
) -> ActivityDistances:
    """
    Compute activity-to-activity distances using Recipe X (rating-cooccurrence).

    Each activity is represented by its importance profile across occupations:
        v(a) = [w_1(a), w_2(a), ..., w_J(a)]
    where w_j(a) is occupation j's weight on activity a.

    Activities are close if they co-occur with similar importance across occupations.

    Args:
        occupation_measures: OccupationMeasures from domain.py
        n_components: Number of PCA components. If None, use variance_threshold.
        variance_threshold: Fraction of variance to retain if n_components is None.
        standardize: If True, standardize columns before PCA.

    Returns:
        ActivityDistances with pairwise distance matrix.

    Paper reference: Section 4.1.1 Recipe X
    """
    # Get raw occupation × activity matrix (before normalization to probabilities)
    # We want importance profiles, not probability-normalized values
    raw_matrix = occupation_measures.raw_matrix  # (n_occupations, n_activities)
    activity_ids = occupation_measures.activity_ids

    # Transpose to get activity × occupation matrix
    # Each row is now an activity's profile across occupations
    activity_profiles = raw_matrix.T  # (n_activities, n_occupations)

    n_activities, n_occupations = activity_profiles.shape

    # Standardize columns (occupations) to zero mean, unit variance
    if standardize:
        scaler = StandardScaler()
        activity_profiles_scaled = scaler.fit_transform(activity_profiles)
    else:
        activity_profiles_scaled = activity_profiles

    # Apply PCA if requested
    pca_variance_explained = None
    actual_n_components = None

    if n_components is not None or variance_threshold < 1.0:
        # Determine number of components
        if n_components is not None:
            actual_n_components = min(n_components, min(n_activities, n_occupations))
        else:
            # Find components to explain variance_threshold
            pca_full = PCA()
            pca_full.fit(activity_profiles_scaled)
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            actual_n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
            actual_n_components = min(actual_n_components, min(n_activities, n_occupations))

        pca = PCA(n_components=actual_n_components)
        activity_profiles_reduced = pca.fit_transform(activity_profiles_scaled)
        pca_variance_explained = sum(pca.explained_variance_ratio_)
    else:
        activity_profiles_reduced = activity_profiles_scaled

    # Compute pairwise Euclidean distances
    distances_condensed = pdist(activity_profiles_reduced, metric="euclidean")
    distance_matrix = squareform(distances_condensed)

    return ActivityDistances(
        distance_matrix=distance_matrix,
        activity_ids=activity_ids,
        activity_profiles=activity_profiles_reduced,
        pca_variance_explained=pca_variance_explained,
        n_components=actual_n_components,
    )


def get_nearest_activities(
    distances: ActivityDistances,
    activity_id: str,
    k: int = 5,
) -> list[tuple[str, float]]:
    """
    Get k nearest neighbors for an activity.

    Args:
        distances: ActivityDistances object
        activity_id: Element ID of query activity
        k: Number of neighbors to return

    Returns:
        List of (activity_id, distance) tuples, sorted by distance.
    """
    if activity_id not in distances.activity_ids:
        raise ValueError(f"Activity {activity_id} not found")

    idx = distances.activity_ids.index(activity_id)
    dists = distances.distance_matrix[idx]

    # Get indices sorted by distance (excluding self)
    sorted_indices = np.argsort(dists)
    neighbors = []

    for i in sorted_indices:
        if i != idx:
            neighbors.append((distances.activity_ids[i], dists[i]))
            if len(neighbors) >= k:
                break

    return neighbors


def distance_percentiles(distances: ActivityDistances) -> dict[str, float]:
    """
    Compute percentiles of pairwise activity distances.

    Useful for selecting kernel bandwidth sigma.

    Returns:
        Dict with keys 'p10', 'p25', 'p50', 'p75', 'p90' and distance values.
    """
    # Get upper triangle (excluding diagonal)
    n = len(distances.activity_ids)
    triu_indices = np.triu_indices(n, k=1)
    pairwise_dists = distances.distance_matrix[triu_indices]

    percentiles = {
        "p10": np.percentile(pairwise_dists, 10),
        "p25": np.percentile(pairwise_dists, 25),
        "p50": np.percentile(pairwise_dists, 50),
        "p75": np.percentile(pairwise_dists, 75),
        "p90": np.percentile(pairwise_dists, 90),
        "min": pairwise_dists.min(),
        "max": pairwise_dists.max(),
        "mean": pairwise_dists.mean(),
    }

    return percentiles
