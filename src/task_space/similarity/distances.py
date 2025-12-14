"""
Activity distance computation.

Implements both Recipe X (rating-cooccurrence) and Recipe Y (text embeddings).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class ActivityDistances:
    """
    Pairwise distances between activities.

    Attributes:
        distance_matrix: Pairwise distances, shape (n_activities, n_activities)
        activity_ids: List of activity IDs (row/column labels)
        activity_profiles: Profiles used for distance computation
        method: 'recipe_x' (rating-cooccurrence) or 'recipe_y' (embeddings)
        pca_variance_explained: Fraction of variance explained by PCA (if applied)
        n_components: Number of PCA components used (None if no PCA)
    """
    distance_matrix: np.ndarray
    activity_ids: list[str]
    activity_profiles: np.ndarray
    method: str
    pca_variance_explained: Optional[float] = None
    n_components: Optional[int] = None


def compute_recipe_x_distances(
    raw_matrix: np.ndarray,
    activity_ids: list[str],
    n_components: Optional[int] = None,
    variance_threshold: float = 0.9,
    standardize: bool = True,
) -> ActivityDistances:
    """
    Compute activity distances using Recipe X (rating-cooccurrence).

    Each activity is represented by its importance profile across occupations:
        v(a) = [w_1(a), w_2(a), ..., w_J(a)]
    where w_j(a) is occupation j's weight on activity a.

    Activities are close if they co-occur with similar importance across occupations.

    Args:
        raw_matrix: (n_occupations, n_activities) raw weight matrix
        activity_ids: Activity ID labels
        n_components: Number of PCA components. If None, use variance_threshold.
        variance_threshold: Fraction of variance to retain if n_components is None.
        standardize: If True, standardize columns before PCA.

    Returns:
        ActivityDistances with pairwise distance matrix.
    """
    # Transpose to get activity x occupation matrix
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
        if n_components is not None:
            actual_n_components = min(n_components, min(n_activities, n_occupations))
        else:
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
        method='recipe_x',
        pca_variance_explained=pca_variance_explained,
        n_components=actual_n_components,
    )


def compute_recipe_y_distances(
    embeddings: np.ndarray,
    activity_ids: list[str],
    metric: str = "cosine",
) -> ActivityDistances:
    """
    Compute activity distances using Recipe Y (text embeddings).

    Args:
        embeddings: (n_activities, embedding_dim) embedding matrix
        activity_ids: Activity ID labels
        metric: 'cosine' (recommended) or 'euclidean'

    Returns:
        ActivityDistances with embedding-based distance matrix.
    """
    if len(embeddings) != len(activity_ids):
        raise ValueError(
            f"Length mismatch: {len(embeddings)} embeddings vs {len(activity_ids)} IDs"
        )

    # Compute pairwise distances
    distances_condensed = pdist(embeddings, metric=metric)
    distance_matrix = squareform(distances_condensed)

    return ActivityDistances(
        distance_matrix=distance_matrix,
        activity_ids=activity_ids,
        activity_profiles=embeddings,
        method='recipe_y',
        pca_variance_explained=None,
        n_components=None,
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

    Useful for understanding distance distribution.

    Returns:
        Dict with keys 'p10', 'p25', 'p50', 'p75', 'p90' and distance values.
    """
    n = len(distances.activity_ids)
    triu_indices = np.triu_indices(n, k=1)
    pairwise_dists = distances.distance_matrix[triu_indices]

    return {
        "p10": float(np.percentile(pairwise_dists, 10)),
        "p25": float(np.percentile(pairwise_dists, 25)),
        "p50": float(np.percentile(pairwise_dists, 50)),
        "p75": float(np.percentile(pairwise_dists, 75)),
        "p90": float(np.percentile(pairwise_dists, 90)),
        "min": float(pairwise_dists.min()),
        "max": float(pairwise_dists.max()),
        "mean": float(pairwise_dists.mean()),
    }
