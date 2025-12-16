"""
Similarity computation for task-space model.

Submodules:
    kernel: Kernel matrix construction with correct defaults
    overlap: Jaccard, kernel, and normalized overlap measures
    embeddings: Text embedding computation
    distances: Distance matrix computation
"""

from .kernel import (
    calibrate_sigma,
    check_kernel_discrimination,
    build_kernel_matrix,
)

from .overlap import (
    compute_jaccard_overlap,
    compute_kernel_overlap,
    compute_normalized_overlap,
)

from .embeddings import (
    compute_activity_embeddings,
    compute_embedding_distances,
    embeddings_to_similarity,
)

from .distances import (
    ActivityDistances,
    compute_recipe_x_distances,
    compute_recipe_y_distances,
    get_nearest_activities,
    distance_percentiles,
)

from .wasserstein import (
    WassersteinDistanceResult,
    compute_wasserstein_distances,
    wasserstein_to_similarity,
)

__all__ = [
    # Kernel
    'calibrate_sigma',
    'check_kernel_discrimination',
    'build_kernel_matrix',
    # Overlap
    'compute_jaccard_overlap',
    'compute_kernel_overlap',
    'compute_normalized_overlap',
    # Embeddings
    'compute_activity_embeddings',
    'compute_embedding_distances',
    'embeddings_to_similarity',
    # Distances
    'ActivityDistances',
    'compute_recipe_x_distances',
    'compute_recipe_y_distances',
    'get_nearest_activities',
    'distance_percentiles',
    # Wasserstein
    'WassersteinDistanceResult',
    'compute_wasserstein_distances',
    'wasserstein_to_similarity',
]
