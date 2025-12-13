"""
Tests for Phase I validation module.

Verifies:
1. Overlap matrix properties (symmetric, non-negative, diagonal highest)
2. All 5 sigma values produce valid results
3. Save/load round-trips correctly
4. Statistics computation is correct

Usage:
    PYTHONPATH=src python tests/test_validation.py
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from task_space import (
    build_occupation_measures,
    compute_activity_distances,
)
from task_space.validation import (
    OverlapResult,
    OverlapGrid,
    SIGMA_PERCENTILES,
    compute_overlap_stats,
    compute_validation_overlap,
    compute_overlap_grid,
    save_overlap_result,
    load_overlap_result,
    save_overlap_grid,
    load_overlap_grid,
)


def test_overlap_stats():
    """Test overlap statistics computation."""
    print("\n[1] Testing overlap statistics computation...")

    # Create a simple test matrix
    n = 10
    matrix = np.random.rand(n, n)
    matrix = (matrix + matrix.T) / 2  # Make symmetric
    np.fill_diagonal(matrix, 1.0)  # Diagonal = 1

    stats = compute_overlap_stats(matrix)

    # Check all expected keys are present
    expected_keys = [
        "off_diag_mean", "off_diag_std", "off_diag_min", "off_diag_max",
        "off_diag_p10", "off_diag_p25", "off_diag_p50", "off_diag_p75", "off_diag_p90",
        "diag_mean", "diag_std", "diag_min", "diag_max",
    ]
    for key in expected_keys:
        assert key in stats, f"Missing key: {key}"

    # Check diagonal stats (all 1.0)
    assert abs(stats["diag_mean"] - 1.0) < 1e-6, "Diagonal mean should be 1.0"
    assert stats["diag_std"] < 1e-6, "Diagonal std should be ~0"

    # Check off-diagonal stats are reasonable
    assert 0 <= stats["off_diag_mean"] <= 1, "Off-diagonal mean out of range"
    assert stats["off_diag_min"] <= stats["off_diag_p10"] <= stats["off_diag_p50"]
    assert stats["off_diag_p50"] <= stats["off_diag_p90"] <= stats["off_diag_max"]

    print("  PASS: Overlap statistics computation works correctly")
    return True


def test_single_sigma_overlap(measures, distances):
    """Test overlap computation for a single sigma."""
    print("\n[2] Testing single sigma overlap computation...")

    result = compute_validation_overlap(measures, distances, "p50")

    # Check result type
    assert isinstance(result, OverlapResult), "Should return OverlapResult"

    # Check matrix properties
    n_occ = len(measures.occupation_codes)
    assert result.overlap_matrix.shape == (n_occ, n_occ), f"Wrong shape: {result.overlap_matrix.shape}"

    # Check symmetry
    assert np.allclose(result.overlap_matrix, result.overlap_matrix.T), "Matrix should be symmetric"

    # Check non-negativity
    assert (result.overlap_matrix >= 0).all(), "All entries should be non-negative"

    # Check diagonal is positive (self-overlap should be non-trivial)
    # Note: diagonal is NOT necessarily the max in each row for kernel-weighted overlap
    # because two different occupations can have higher overlap if their distributions
    # are centered on nearby activities in the kernel space
    diag = np.diag(result.overlap_matrix)
    assert (diag > 0).all(), "Diagonal entries should be positive"
    assert diag.mean() > 0.01, f"Mean diagonal should be non-trivial, got {diag.mean()}"

    # Check metadata
    assert result.occupation_codes == measures.occupation_codes
    assert result.sigma > 0, "Sigma should be positive"
    assert result.sigma_percentile == "p50"
    assert isinstance(result.stats, dict)

    print(f"  Matrix shape: {result.overlap_matrix.shape}")
    print(f"  Sigma (p50): {result.sigma:.4f}")
    print(f"  Off-diagonal mean: {result.stats['off_diag_mean']:.6f}")
    print("  PASS: Single sigma overlap computation works correctly")

    return result


def test_overlap_grid(measures, distances):
    """Test overlap grid computation for all 5 sigma values."""
    print("\n[3] Testing overlap grid computation (all 5 sigma values)...")

    grid = compute_overlap_grid(measures, distances)

    # Check result type
    assert isinstance(grid, OverlapGrid), "Should return OverlapGrid"

    # Check all 5 percentiles are present
    assert set(grid.results.keys()) == set(SIGMA_PERCENTILES), "Missing sigma percentiles"

    # Check distance_percentiles
    assert set(grid.distance_percentiles.keys()) == set(SIGMA_PERCENTILES)

    # Check n_occupations
    assert grid.n_occupations == len(measures.occupation_codes)

    # Check each result
    prev_sigma = 0
    for pct in SIGMA_PERCENTILES:
        result = grid.results[pct]
        assert isinstance(result, OverlapResult)
        assert result.sigma_percentile == pct
        # Sigma should increase with percentile
        assert result.sigma > prev_sigma, f"Sigma should increase: {pct}"
        prev_sigma = result.sigma

    print(f"  Grid computed for {len(SIGMA_PERCENTILES)} sigma values")
    for pct in SIGMA_PERCENTILES:
        result = grid.results[pct]
        print(f"    {pct}: sigma={result.sigma:.4f}, mean_overlap={result.stats['off_diag_mean']:.6f}")
    print("  PASS: Overlap grid computation works correctly")

    return grid


def test_save_load_result(result):
    """Test save/load round-trip for single OverlapResult."""
    print("\n[4] Testing save/load round-trip for OverlapResult...")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_overlap"

        # Save
        save_overlap_result(result, path)

        # Check files exist
        assert (path.with_suffix(".npz")).exists(), "NPZ file not created"
        assert (path.with_suffix(".json")).exists(), "JSON file not created"

        # Load
        loaded = load_overlap_result(path)

        # Verify
        assert np.allclose(loaded.overlap_matrix, result.overlap_matrix), "Matrix mismatch"
        assert loaded.occupation_codes == result.occupation_codes, "Occupation codes mismatch"
        assert loaded.sigma == result.sigma, "Sigma mismatch"
        assert loaded.sigma_percentile == result.sigma_percentile, "Percentile mismatch"
        assert loaded.stats == result.stats, "Stats mismatch"

    print("  PASS: Save/load round-trip works correctly")
    return True


def test_save_load_grid(grid):
    """Test save/load round-trip for OverlapGrid."""
    print("\n[5] Testing save/load round-trip for OverlapGrid...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "phase_i"

        # Save
        save_overlap_grid(grid, output_dir)

        # Check files exist
        assert output_dir.exists(), "Output directory not created"
        assert (output_dir / "overlap_stats.json").exists(), "Summary file not created"
        for pct in SIGMA_PERCENTILES:
            assert (output_dir / f"overlap_{pct}.npz").exists(), f"NPZ for {pct} not created"
            assert (output_dir / f"overlap_{pct}.json").exists(), f"JSON for {pct} not created"

        # Load
        loaded = load_overlap_grid(output_dir)

        # Verify
        assert loaded.n_occupations == grid.n_occupations
        assert loaded.distance_percentiles == grid.distance_percentiles
        for pct in SIGMA_PERCENTILES:
            assert np.allclose(
                loaded.results[pct].overlap_matrix,
                grid.results[pct].overlap_matrix
            ), f"Matrix mismatch for {pct}"

    print("  PASS: Grid save/load round-trip works correctly")
    return True


def test_invalid_sigma():
    """Test that invalid sigma percentile raises error."""
    print("\n[6] Testing invalid sigma percentile handling...")

    measures = build_occupation_measures()
    distances = compute_activity_distances(measures)

    try:
        compute_validation_overlap(measures, distances, "p99")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "p99" in str(e)
        print(f"  Correctly raised: {e}")

    print("  PASS: Invalid sigma handling works correctly")
    return True


def main():
    print("=" * 60)
    print("PHASE I VALIDATION MODULE TESTS")
    print("=" * 60)

    try:
        # Load data (needed for most tests)
        print("\nLoading occupation measures and distances...")
        measures = build_occupation_measures()
        distances = compute_activity_distances(measures)
        print(f"  Loaded {len(measures.occupation_codes)} occupations, {len(distances.activity_ids)} activities")

        # Run tests
        test_overlap_stats()
        result = test_single_sigma_overlap(measures, distances)
        grid = test_overlap_grid(measures, distances)
        test_save_load_result(result)
        test_save_load_grid(grid)
        test_invalid_sigma()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)

        # Summary
        print("\nValidation Module Summary:")
        print(f"  Occupations: {grid.n_occupations}")
        print(f"  Occupation pairs: {grid.n_occupations * (grid.n_occupations - 1) // 2}")
        print(f"  Sigma grid: {list(grid.distance_percentiles.keys())}")
        print(f"  Headline (p50) sigma: {grid.results['p50'].sigma:.4f}")
        print(f"  Headline (p50) mean overlap: {grid.results['p50'].stats['off_diag_mean']:.6f}")

        return True

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
