"""
Text embedding computation for activities.

Uses sentence-transformers to embed activity titles/descriptions.
All embeddings are cached via the artifacts module.
"""

import numpy as np

from ..data.artifacts import get_embeddings, get_distance_matrix


def compute_activity_embeddings(
    activity_titles: list[str],
    model: str = "all-mpnet-base-v2",
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Compute embeddings for activity titles.

    This is a convenience wrapper around artifacts.get_embeddings().

    Args:
        activity_titles: List of activity description strings
        model: Sentence transformer model name
            Default: "all-mpnet-base-v2" (768-dim, best quality)
            Alternative: "all-MiniLM-L6-v2" (384-dim, 5x faster)
        force_recompute: Bypass cache

    Returns:
        (n_activities, embedding_dim) array
    """
    return get_embeddings(activity_titles, model=model, force_recompute=force_recompute)


def compute_embedding_distances(
    embeddings: np.ndarray,
    metric: str = "cosine",
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Compute pairwise distances from embeddings.

    This is a convenience wrapper around artifacts.get_distance_matrix().

    Args:
        embeddings: (n, d) embedding matrix
        metric: 'cosine' (recommended) or 'euclidean'
        force_recompute: Bypass cache

    Returns:
        (n, n) distance matrix
    """
    return get_distance_matrix(embeddings, metric=metric, force_recompute=force_recompute)


def embeddings_to_similarity(
    embeddings: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Convert embeddings to similarity matrix.

    Args:
        embeddings: (n, d) embedding matrix
        metric: 'cosine' or 'euclidean'

    Returns:
        (n, n) similarity matrix (higher = more similar)
    """
    distances = compute_embedding_distances(embeddings, metric=metric)

    if metric == "cosine":
        # Cosine distance is in [0, 2], similarity = 1 - distance
        return 1 - distances
    else:
        # For Euclidean, use exp(-d) as similarity
        return np.exp(-distances)
