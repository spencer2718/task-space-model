"""
Test the full v0.4 pipeline with O*NET data.

Verifies:
1. Data loading from O*NET files
2. Activity domain construction (41 GWAs)
3. Occupation measure construction (~923 occupations)
4. Activity distance computation (Recipe X)
5. Kernel matrix and exposure computation
6. Phase I diagnostics

Usage:
    cd /home/spencer/Research/task-space-model
    . .venv/bin/activate
    PYTHONPATH=src python tests/test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from task_space import (
    # Data
    load_work_activities,
    get_gwa_ids,
    get_occupation_codes,
    # Domain
    build_activity_domain,
    build_occupation_measures,
    # Distances
    compute_activity_distances,
    get_nearest_activities,
    distance_percentiles,
    # Kernel
    build_kernel_matrix,
    compute_occupation_exposure,
    create_shock_profile,
    # Diagnostics
    diagnose_measure_coherence,
    generate_diagnostic_report,
)


def test_data_loading():
    """Test O*NET data loading."""
    print("\n[1] Testing data loading...")

    df = load_work_activities()
    print(f"  Work Activities rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    gwa_ids = get_gwa_ids()
    print(f"  GWA IDs: {len(gwa_ids)}")

    occ_codes = get_occupation_codes()
    print(f"  Occupation codes: {len(occ_codes)}")

    assert len(gwa_ids) == 41, f"Expected 41 GWAs, got {len(gwa_ids)}"
    assert len(occ_codes) > 800, f"Expected >800 occupations, got {len(occ_codes)}"

    print("  PASS: Data loading works correctly")
    return True


def test_activity_domain():
    """Test activity domain construction."""
    print("\n[2] Testing activity domain construction...")

    domain = build_activity_domain()
    print(f"  Activity IDs: {domain.n_activities}")
    print(f"  Sample activities: {list(domain.activity_names.values())[:3]}")
    print(f"  Reference measure sum: {domain.reference_measure.sum():.4f}")

    assert domain.n_activities == 41, f"Expected 41 activities, got {domain.n_activities}"
    assert abs(domain.reference_measure.sum() - 1.0) < 1e-6, "Reference measure should sum to 1"

    print("  PASS: Activity domain construction works correctly")
    return domain


def test_occupation_measures(domain):
    """Test occupation measure construction."""
    print("\n[3] Testing occupation measure construction...")

    measures = build_occupation_measures()
    print(f"  Occupations: {len(measures.occupation_codes)}")
    print(f"  Activities: {len(measures.activity_ids)}")
    print(f"  Matrix shape: {measures.occupation_matrix.shape}")

    # Check that rows sum to 1 (probability measures)
    row_sums = measures.occupation_matrix.sum(axis=1)
    print(f"  Row sums (min, max): ({row_sums.min():.4f}, {row_sums.max():.4f})")

    assert measures.occupation_matrix.shape[1] == 41, "Expected 41 activities"
    assert np.allclose(row_sums, 1.0), "Rows should sum to 1"

    print("  PASS: Occupation measures constructed correctly")
    return measures


def test_activity_distances(measures):
    """Test Recipe X activity distances."""
    print("\n[4] Testing activity distance computation (Recipe X)...")

    distances = compute_activity_distances(measures, variance_threshold=0.9)
    print(f"  Distance matrix shape: {distances.distance_matrix.shape}")
    print(f"  PCA components: {distances.n_components}")
    print(f"  Variance explained: {distances.pca_variance_explained:.2%}")

    # Check distance matrix properties
    D = distances.distance_matrix
    assert D.shape[0] == D.shape[1] == 41, "Distance matrix should be 41x41"
    assert np.allclose(np.diag(D), 0), "Diagonal should be zero"
    assert np.allclose(D, D.T), "Distance matrix should be symmetric"

    # Get distance percentiles
    percentiles = distance_percentiles(distances)
    print(f"  Distance percentiles: p25={percentiles['p25']:.3f}, p50={percentiles['p50']:.3f}, p75={percentiles['p75']:.3f}")

    # Check nearest neighbors for a sample activity
    sample_id = distances.activity_ids[0]
    neighbors = get_nearest_activities(distances, sample_id, k=3)
    print(f"  Nearest neighbors of {sample_id}: {[n[0] for n in neighbors]}")

    print("  PASS: Activity distances computed correctly")
    return distances


def test_kernel_and_exposure(measures, distances, domain):
    """Test kernel matrix and exposure computation."""
    print("\n[5] Testing kernel matrix and exposure computation...")

    # Use median distance as sigma
    percentiles = distance_percentiles(distances)
    sigma = percentiles['p50']
    print(f"  Using sigma = {sigma:.3f} (p50 of distances)")

    kernel = build_kernel_matrix(distances, sigma=sigma, kernel_type="exponential")
    print(f"  Kernel matrix shape: {kernel.matrix.shape}")
    print(f"  Kernel row sums (should be 1): {kernel.matrix.sum(axis=1)[:5]}")

    # Check kernel properties
    K = kernel.matrix
    assert K.shape == (41, 41), "Kernel should be 41x41"
    assert np.allclose(K.sum(axis=1), 1.0), "Kernel rows should sum to 1"

    # Create a simple shock profile (high on first few activities)
    shock = create_shock_profile(
        activity_ids=distances.activity_ids,
        target_activities={distances.activity_ids[0]: 1.0, distances.activity_ids[1]: 0.5},
    )
    print(f"  Shock profile sum: {shock.sum():.2f}")

    # Compute exposure
    result = compute_occupation_exposure(measures, kernel, shock)
    print(f"  Exposure range: [{result.exposures.min():.4f}, {result.exposures.max():.4f}]")

    # Check that exposures vary across occupations
    exposure_std = result.exposures.std()
    print(f"  Exposure std: {exposure_std:.4f}")
    assert exposure_std > 0, "Exposures should vary across occupations"

    # Show top 5 most exposed occupations
    top_indices = np.argsort(result.exposures)[::-1][:5]
    print("  Top 5 exposed occupations:")
    for idx in top_indices:
        print(f"    {result.occupation_codes[idx]}: {result.exposures[idx]:.4f}")

    print("  PASS: Kernel and exposure computation works correctly")
    return kernel, result


def test_diagnostics(measures, domain, distances):
    """Test Phase I diagnostics."""
    print("\n[6] Testing Phase I diagnostics...")

    # Measure coherence
    coherence = diagnose_measure_coherence(measures)
    print(f"  Median entropy: {coherence.median_entropy:.3f}")
    print(f"  Median effective support: {coherence.median_support:.1f}")
    print(f"  Coverage (support >= 15): {coherence.coverage:.1%}")
    print(f"  Sparse occupations: {len(coherence.sparse_occupations)}")

    # Sample occupations for spot check
    sample_codes = measures.occupation_codes[:5]

    # Generate diagnostic report
    report = generate_diagnostic_report(
        measures, domain, distances, sample_occupations=sample_codes
    )
    print("\n  Diagnostic report preview:")
    for line in report.split('\n')[:20]:
        print(f"    {line}")
    print("    ...")

    print("  PASS: Diagnostics generated correctly")
    return coherence


def main():
    print("=" * 60)
    print("TASK SPACE v0.4 PIPELINE TEST")
    print("=" * 60)

    try:
        # Run tests in sequence
        test_data_loading()
        domain = test_activity_domain()
        measures = test_occupation_measures(domain)
        distances = test_activity_distances(measures)
        kernel, result = test_kernel_and_exposure(measures, distances, domain)
        coherence = test_diagnostics(measures, domain, distances)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)

        # Summary
        print("\nPipeline Summary:")
        print(f"  Activity domain: {domain.n_activities} GWAs")
        print(f"  Occupations: {len(measures.occupation_codes)}")
        print(f"  Distance matrix: {distances.distance_matrix.shape}")
        print(f"  Kernel sigma: {kernel.sigma:.3f}")
        print(f"  Measure coverage: {coherence.coverage:.1%}")

        return True

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
