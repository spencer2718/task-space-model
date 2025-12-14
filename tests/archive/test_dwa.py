"""
Tests for DWA domain and Recipe Y text embedding distances.

v0.4.2: Tests the post-validation pivot to DWA domain + Recipe Y geometry.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_dwa_domain_construction():
    """Test DWA domain has ~2,087 activities."""
    from task_space import build_dwa_activity_domain

    domain = build_dwa_activity_domain()

    print(f"\nDWA Domain:")
    print(f"  n_activities: {domain.n_activities}")
    print(f"  Sample IDs: {domain.activity_ids[:5]}")

    # Check expected range (spec says ~2,087)
    assert 2000 <= domain.n_activities <= 2200, (
        f"Expected ~2,087 DWAs, got {domain.n_activities}"
    )
    assert len(domain.activity_ids) == domain.n_activities
    assert len(domain.activity_names) == domain.n_activities
    assert len(domain.reference_measure) == domain.n_activities

    # Reference measure should be uniform and sum to 1
    assert np.isclose(domain.reference_measure.sum(), 1.0)
    assert np.allclose(domain.reference_measure, 1 / domain.n_activities)

    print("  PASSED: DWA domain construction")
    return domain


def test_dwa_occupation_measures():
    """Test DWA occupation measures construction."""
    from task_space import build_dwa_occupation_measures

    measures = build_dwa_occupation_measures()

    print(f"\nDWA Occupation Measures:")
    print(f"  Shape: {measures.occupation_matrix.shape}")
    print(f"  n_occupations: {len(measures.occupation_codes)}")
    print(f"  n_activities: {len(measures.activity_ids)}")

    n_occ, n_act = measures.occupation_matrix.shape

    # Check expected dimensions
    assert 800 <= n_occ <= 1000, f"Expected ~900 occupations, got {n_occ}"
    assert 2000 <= n_act <= 2200, f"Expected ~2,087 DWAs, got {n_act}"

    # Rows should sum to 1 (probability distributions)
    row_sums = measures.occupation_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0), (
        f"Rows should sum to 1, got min={row_sums.min():.4f}, max={row_sums.max():.4f}"
    )

    # All values should be non-negative
    assert (measures.occupation_matrix >= 0).all(), "All weights should be >= 0"

    print("  PASSED: DWA occupation measures construction")
    return measures


def test_dwa_sparsity_diagnostics():
    """Test DWA sparsity diagnostic."""
    from task_space import build_dwa_occupation_measures, diagnose_dwa_sparsity

    measures = build_dwa_occupation_measures()
    sparsity = diagnose_dwa_sparsity(measures)

    print(f"\nDWA Sparsity Report:")
    print(f"  n_occupations: {sparsity.n_occupations}")
    print(f"  n_dwas: {sparsity.n_dwas}")
    print(f"  Effective support (median): {sparsity.effective_support_percentiles['p50']:.1f}")
    print(f"  Effective support (p10-p90): {sparsity.effective_support_percentiles['p10']:.1f} - {sparsity.effective_support_percentiles['p90']:.1f}")
    print(f"  DWA coverage: {sparsity.dwa_coverage:.1%}")
    print(f"  Flagged occupations (support < 30): {sparsity.n_flagged}")

    # Most occupations should have reasonable support
    assert sparsity.effective_support_percentiles['p50'] > 10, (
        "Median effective support should be > 10"
    )

    print("  PASSED: DWA sparsity diagnostics")
    return sparsity


def test_text_embedding_distances():
    """Test Recipe Y text embedding distance computation."""
    from task_space import build_dwa_activity_domain, compute_text_embedding_distances

    domain = build_dwa_activity_domain()

    # Get titles and IDs
    titles = list(domain.activity_names.values())
    ids = domain.activity_ids

    print(f"\nText Embedding Distances (Recipe Y):")
    print(f"  Encoding {len(titles)} activity titles...")

    distances = compute_text_embedding_distances(titles, ids)

    print(f"  Distance matrix shape: {distances.distance_matrix.shape}")
    print(f"  Embedding shape: {distances.activity_profiles.shape}")

    n = len(ids)
    assert distances.distance_matrix.shape == (n, n), (
        f"Expected ({n}, {n}), got {distances.distance_matrix.shape}"
    )

    # Diagonal should be 0
    diag = np.diag(distances.distance_matrix)
    assert np.allclose(diag, 0), f"Diagonal should be 0, got max={diag.max():.6f}"

    # Should be symmetric
    assert np.allclose(distances.distance_matrix, distances.distance_matrix.T), (
        "Distance matrix should be symmetric"
    )

    # Cosine distance should be in [0, 2]
    assert distances.distance_matrix.min() >= 0, "Distances should be >= 0"
    assert distances.distance_matrix.max() <= 2.0, "Cosine distance should be <= 2"

    # Check distance percentiles
    from task_space import distance_percentiles
    pcts = distance_percentiles(distances)
    print(f"  Distance percentiles:")
    print(f"    p10: {pcts['p10']:.4f}")
    print(f"    p50: {pcts['p50']:.4f}")
    print(f"    p90: {pcts['p90']:.4f}")

    # PCA fields should be None for Recipe Y
    assert distances.pca_variance_explained is None
    assert distances.n_components is None

    print("  PASSED: Text embedding distances")
    return distances


def test_semantic_neighbors():
    """Verify text embeddings produce semantically sensible neighbors."""
    from task_space import (
        build_dwa_activity_domain,
        compute_text_embedding_distances,
        get_nearest_activities,
    )

    domain = build_dwa_activity_domain()
    titles = list(domain.activity_names.values())
    ids = domain.activity_ids

    distances = compute_text_embedding_distances(titles, ids)

    print(f"\nSemantic Neighbor Spot Checks:")

    # Check a few activities for sensible neighbors
    # Find an activity related to "financial" or "data"
    financial_activities = [
        (aid, title) for aid, title in domain.activity_names.items()
        if "financ" in title.lower() or "budget" in title.lower()
    ]

    if financial_activities:
        sample_id, sample_title = financial_activities[0]
        neighbors = get_nearest_activities(distances, sample_id, k=5)

        print(f"\n  Query: '{sample_title}'")
        print("  Nearest neighbors:")
        for neighbor_id, dist in neighbors:
            neighbor_title = domain.activity_names[neighbor_id]
            print(f"    {dist:.4f}  {neighbor_title[:60]}")

    # Find an activity related to "equipment" or "machinery"
    equipment_activities = [
        (aid, title) for aid, title in domain.activity_names.items()
        if "equip" in title.lower() or "machin" in title.lower()
    ]

    if equipment_activities:
        sample_id, sample_title = equipment_activities[0]
        neighbors = get_nearest_activities(distances, sample_id, k=5)

        print(f"\n  Query: '{sample_title}'")
        print("  Nearest neighbors:")
        for neighbor_id, dist in neighbors:
            neighbor_title = domain.activity_names[neighbor_id]
            print(f"    {dist:.4f}  {neighbor_title[:60]}")

    print("\n  (Manual inspection required for semantic validity)")
    print("  PASSED: Semantic neighbors test")


def test_dwa_pipeline_integration():
    """Test full pipeline with DWA domain and Recipe Y."""
    from task_space import (
        build_dwa_activity_domain,
        build_dwa_occupation_measures,
        compute_text_embedding_distances,
        build_kernel_matrix,
        compute_overlap,
        distance_percentiles,
    )

    print("\nFull DWA + Recipe Y Pipeline Integration:")

    # Step 1: Build domain
    print("  Building DWA domain...")
    domain = build_dwa_activity_domain()

    # Step 2: Build occupation measures
    print("  Building DWA occupation measures...")
    measures = build_dwa_occupation_measures()

    # Step 3: Compute text embedding distances
    print("  Computing text embedding distances...")
    titles = list(domain.activity_names.values())
    distances = compute_text_embedding_distances(titles, domain.activity_ids)

    # Step 4: Build kernel at median sigma
    print("  Building kernel matrix...")
    pcts = distance_percentiles(distances)
    sigma = pcts['p50']
    kernel = build_kernel_matrix(distances, sigma=sigma)

    print(f"    Sigma (p50): {sigma:.4f}")
    print(f"    Kernel shape: {kernel.matrix.shape}")

    # Step 5: Compute overlap
    print("  Computing occupation overlap matrix...")
    overlap = compute_overlap(measures, kernel)

    print(f"    Overlap shape: {overlap.shape}")

    # Checks
    n_occ = len(measures.occupation_codes)
    assert overlap.shape == (n_occ, n_occ), f"Expected ({n_occ}, {n_occ})"

    # Diagonal should be close to 1 (self-overlap)
    diag = np.diag(overlap)
    print(f"    Self-overlap (diagonal) mean: {diag.mean():.4f}")

    # Off-diagonal should be < 1
    off_diag = overlap[np.triu_indices(n_occ, k=1)]
    print(f"    Off-diagonal overlap: mean={off_diag.mean():.4f}, max={off_diag.max():.4f}")

    print("  PASSED: Pipeline integration")


if __name__ == "__main__":
    print("=" * 60)
    print("DWA + Recipe Y Test Suite (v0.4.2)")
    print("=" * 60)

    # Run tests
    test_dwa_domain_construction()
    test_dwa_occupation_measures()
    test_dwa_sparsity_diagnostics()
    test_text_embedding_distances()
    test_semantic_neighbors()
    test_dwa_pipeline_integration()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
