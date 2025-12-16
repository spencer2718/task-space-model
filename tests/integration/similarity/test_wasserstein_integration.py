"""Integration test: Wasserstein on actual O*NET data."""
import pytest
import numpy as np


@pytest.mark.slow
def test_wasserstein_on_onet_sample():
    """Test Wasserstein computation on a sample of real occupations.

    Uses 50 occupations to keep runtime reasonable (~10-30 seconds).
    Full 894×894 matrix would take 30-60 minutes.
    """
    from task_space.domain import build_dwa_occupation_measures
    from task_space.data import get_dwa_titles
    from task_space.data.artifacts import get_embeddings, get_distance_matrix
    from task_space.similarity.wasserstein import compute_wasserstein_distances

    # Load real data
    measures = build_dwa_occupation_measures()

    # Sample first 50 occupations
    occ_matrix = measures.occupation_matrix[:50]

    # Get activity titles for embeddings
    dwa_titles = get_dwa_titles()
    activity_titles = [dwa_titles.get(aid, aid) for aid in measures.activity_ids]

    # Get activity embeddings and ground distance matrix
    embeddings = get_embeddings(activity_titles, model="all-mpnet-base-v2")
    ground = get_distance_matrix(embeddings, metric="cosine")

    # Compute Wasserstein distances
    result = compute_wasserstein_distances(occ_matrix, ground, verbose=True)

    # Basic sanity checks
    assert result.distance_matrix.shape == (50, 50), \
        f"Expected (50, 50), got {result.distance_matrix.shape}"

    # Self-distance should be zero
    np.testing.assert_allclose(
        np.diag(result.distance_matrix), 0, atol=1e-10,
        err_msg="Self-distances should be zero"
    )

    # Matrix should be symmetric
    np.testing.assert_allclose(
        result.distance_matrix, result.distance_matrix.T, atol=1e-10,
        err_msg="Distance matrix should be symmetric"
    )

    # Median support size should be reasonable (expect ~15-40 based on DWA sparsity)
    assert 10 < result.median_support_size < 100, \
        f"Unexpected median support size: {result.median_support_size}"

    # All distances should be non-negative
    assert np.all(result.distance_matrix >= 0), "Distances should be non-negative"

    # Print diagnostics for the report
    print(f"\n--- Integration Test Results ---")
    print(f"Matrix shape: {result.distance_matrix.shape}")
    print(f"Computation time: {result.computation_time_seconds:.1f}s")
    print(f"Median support union size: {result.median_support_size:.1f}")
    print(f"Distance range: [{result.distance_matrix.min():.4f}, {result.distance_matrix.max():.4f}]")
    print(f"Mean distance: {result.distance_matrix.mean():.4f}")

    # Check triangle inequality on a sample of triplets
    n_violations = 0
    for i in range(min(10, 50)):
        for j in range(i+1, min(20, 50)):
            for k in range(j+1, min(30, 50)):
                D = result.distance_matrix
                if D[i, k] > D[i, j] + D[j, k] + 1e-8:
                    n_violations += 1

    assert n_violations == 0, f"Triangle inequality violations: {n_violations}"
    print(f"Triangle inequality: PASSED (sampled)")


@pytest.mark.slow
def test_wasserstein_vs_kernel_direction():
    """Verify that Wasserstein and kernel overlap rank occupations similarly.

    This is a weak test: we just check that the correlation is positive,
    not that one dominates the other (that's for Path A experiment).
    """
    from task_space.domain import build_dwa_occupation_measures
    from task_space.data import get_dwa_titles
    from task_space.data.artifacts import get_embeddings, get_distance_matrix
    from task_space.similarity.wasserstein import compute_wasserstein_distances
    from task_space.similarity import build_kernel_matrix, compute_normalized_overlap
    from scipy.stats import spearmanr

    # Load data (smaller sample for speed)
    measures = build_dwa_occupation_measures()
    occ_matrix = measures.occupation_matrix[:30]

    dwa_titles = get_dwa_titles()
    activity_titles = [dwa_titles.get(aid, aid) for aid in measures.activity_ids]

    embeddings = get_embeddings(activity_titles, model="all-mpnet-base-v2")
    ground = get_distance_matrix(embeddings, metric="cosine")

    # Compute Wasserstein
    wass_result = compute_wasserstein_distances(occ_matrix, ground, verbose=False)
    wass_dist = wass_result.distance_matrix

    # Compute kernel overlap
    K, sigma = build_kernel_matrix(ground)
    overlap = compute_normalized_overlap(occ_matrix, K)
    kernel_dist = 1 - overlap  # Convert similarity to distance

    # Extract upper triangle for correlation
    triu_idx = np.triu_indices(30, k=1)
    wass_flat = wass_dist[triu_idx]
    kernel_flat = kernel_dist[triu_idx]

    # Check correlation
    corr, pval = spearmanr(wass_flat, kernel_flat)

    print(f"\n--- Wasserstein vs Kernel Correlation ---")
    print(f"Spearman rho: {corr:.3f}")
    print(f"p-value: {pval:.2e}")

    # We expect positive correlation (both measure task dissimilarity)
    # but not necessarily high correlation (different geometry)
    assert corr > 0, f"Expected positive correlation, got {corr:.3f}"
    assert pval < 0.05, f"Correlation not significant: p={pval:.3f}"
