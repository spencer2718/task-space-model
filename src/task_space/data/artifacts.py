"""
Unified artifact store for expensive computations.

IMPORTANT: This is THE canonical location for cached embeddings and distances.
Do not create .npy files in outputs/ or elsewhere.

Cache location: .cache/artifacts/v1/
"""

from pathlib import Path
from typing import Optional
import hashlib

import numpy as np


CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".cache" / "artifacts"
CACHE_VERSION = "v1"


def _hash_texts(texts: list[str]) -> str:
    """Create deterministic hash of text list."""
    h = hashlib.sha256()
    for t in sorted(texts):  # Sort for determinism
        h.update(t.encode('utf-8'))
    return h.hexdigest()[:16]


def _get_cache_path(artifact_type: str, identifier: str) -> Path:
    """Get canonical path for an artifact."""
    cache_dir = CACHE_DIR / CACHE_VERSION / artifact_type
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{identifier}.npz"


def _compute_embeddings_impl(texts: list[str], model: str) -> np.ndarray:
    """
    Internal function to compute embeddings.

    Separated for testability (allows mocking to verify cache behavior).
    """
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(model)
    return encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)


def get_embeddings(
    texts: list[str],
    model: str = "all-mpnet-base-v2",
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Get text embeddings, using cache if available.

    This is THE canonical way to get embeddings. Do not compute elsewhere.

    Args:
        texts: List of texts to embed
        model: Sentence transformer model name
        force_recompute: Bypass cache

    Returns:
        (n_texts, embedding_dim) array
    """
    text_hash = _hash_texts(texts)
    cache_path = _get_cache_path("embeddings", f"{model}_{text_hash}")

    if not force_recompute and cache_path.exists():
        data = np.load(cache_path)
        return data['embeddings']

    # Compute using internal function (testable)
    embeddings = _compute_embeddings_impl(texts, model)

    # Cache
    np.savez_compressed(cache_path, embeddings=embeddings, model=model, n_texts=len(texts))

    return embeddings


def get_distance_matrix(
    embeddings: np.ndarray,
    metric: str = "cosine",
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Get distance matrix, using cache if available.

    Args:
        embeddings: (n, d) embedding matrix
        metric: Distance metric ('cosine' or 'euclidean')
        force_recompute: Bypass cache

    Returns:
        (n, n) distance matrix
    """
    # Hash based on embedding shape and first/last values (fast proxy)
    emb_id = f"{embeddings.shape}_{embeddings[0,0]:.6f}_{embeddings[-1,-1]:.6f}"
    emb_hash = hashlib.sha256(emb_id.encode()).hexdigest()[:16]
    cache_path = _get_cache_path("distances", f"{metric}_{emb_hash}")

    if not force_recompute and cache_path.exists():
        data = np.load(cache_path)
        return data['distances']

    # Compute
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
    if metric == "cosine":
        distances = cosine_distances(embeddings)
    elif metric == "euclidean":
        distances = euclidean_distances(embeddings)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Cache
    np.savez_compressed(cache_path, distances=distances, metric=metric)

    return distances


def clear_cache(artifact_type: Optional[str] = None) -> int:
    """
    Clear cached artifacts.

    Args:
        artifact_type: 'embeddings', 'distances', or None for all

    Returns:
        Number of files deleted
    """
    if artifact_type:
        target = CACHE_DIR / CACHE_VERSION / artifact_type
    else:
        target = CACHE_DIR / CACHE_VERSION

    count = 0
    if target.exists():
        for f in target.rglob("*.npz"):
            f.unlink()
            count += 1
    return count
