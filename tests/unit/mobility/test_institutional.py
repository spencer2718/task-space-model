"""
Unit tests for institutional distance computation.

Tests:
- Distance formula with known values
- Imputation behavior for missing certification
- Gamma parameter effect
- Symmetry and non-negativity
"""

import numpy as np
import pandas as pd
import pytest

from task_space.mobility.institutional import (
    InstitutionalDistanceResult,
    build_institutional_distance_matrix,
    compute_institutional_distance,
    get_zone_difference,
    # v0.6.6.0: Asymmetric
    AsymmetricInstitutionalDistanceResult,
    build_asymmetric_institutional_distance,
    verify_asymmetric_decomposition,
)


class TestInstitutionalDistanceFormula:
    """Test the distance formula implementation."""

    def test_distance_symmetry(self):
        """d_inst(i,j) should equal d_inst(j,i)."""
        result = build_institutional_distance_matrix()
        matrix = result.matrix

        # Check symmetry
        assert np.allclose(matrix, matrix.T), "Distance matrix should be symmetric"

    def test_distance_non_negative(self):
        """All distances should be >= 0."""
        result = build_institutional_distance_matrix()
        assert np.all(result.matrix >= 0), "All distances should be non-negative"

    def test_diagonal_zero(self):
        """Distance from occupation to itself should be 0."""
        result = build_institutional_distance_matrix()
        diagonal = np.diag(result.matrix)
        assert np.allclose(diagonal, 0), "Diagonal should be zero"

    def test_zone_component_range(self):
        """Zone difference component should be in [0, 4]."""
        result = build_institutional_distance_matrix()
        # Zone range is 1-5, so max difference is 4
        zone_diff = np.abs(result.zone_vector[:, None] - result.zone_vector[None, :])
        assert zone_diff.max() <= 4, "Max zone difference should be <= 4"
        assert zone_diff.min() >= 0, "Min zone difference should be >= 0"


class TestGammaParameter:
    """Test gamma (certification weight) parameter."""

    def test_gamma_zero_equals_zone_only(self):
        """With gamma=0, distance should equal zone difference only."""
        result = build_institutional_distance_matrix(gamma=0.0)
        zone_diff = np.abs(result.zone_vector[:, None] - result.zone_vector[None, :])
        assert np.allclose(result.matrix, zone_diff), \
            "With gamma=0, distance should equal zone difference"

    def test_gamma_increases_distance(self):
        """Higher gamma should increase average distance (when certs differ)."""
        result_low = build_institutional_distance_matrix(gamma=0.5)
        result_high = build_institutional_distance_matrix(gamma=2.0)

        mean_low = result_low.matrix[np.triu_indices_from(result_low.matrix, k=1)].mean()
        mean_high = result_high.matrix[np.triu_indices_from(result_high.matrix, k=1)].mean()

        assert mean_high > mean_low, "Higher gamma should increase average distance"

    def test_gamma_stored_in_result(self):
        """Gamma value should be stored in result."""
        result = build_institutional_distance_matrix(gamma=1.5)
        assert result.gamma == 1.5


class TestCertificationImputation:
    """Test certification imputation behavior."""

    def test_imputation_count_tracked(self):
        """Number of imputed certifications should be tracked."""
        result = build_institutional_distance_matrix()
        # Should have some imputation (not all occupations have cert data)
        assert result.n_imputed_cert >= 0
        assert result.cert_coverage <= 1.0

    def test_imputation_in_assumptions(self):
        """Imputation assumption should be documented."""
        result = build_institutional_distance_matrix()
        assumption_texts = " ".join(result.assumptions)
        assert "imputed" in assumption_texts.lower() or "missing" in assumption_texts.lower()


class TestInstitutionalDistanceResult:
    """Test the result dataclass."""

    def test_result_has_all_fields(self):
        """Result should have all expected fields."""
        result = build_institutional_distance_matrix()

        assert hasattr(result, "matrix")
        assert hasattr(result, "occupations")
        assert hasattr(result, "zone_vector")
        assert hasattr(result, "cert_vector")
        assert hasattr(result, "cert_normalized")
        assert hasattr(result, "n_occupations")
        assert hasattr(result, "n_imputed_cert")
        assert hasattr(result, "cert_coverage")
        assert hasattr(result, "gamma")
        assert hasattr(result, "assumptions")

    def test_matrix_occupation_alignment(self):
        """Matrix dimensions should match occupation count."""
        result = build_institutional_distance_matrix()
        n = result.n_occupations

        assert result.matrix.shape == (n, n)
        assert len(result.occupations) == n
        assert len(result.zone_vector) == n
        assert len(result.cert_vector) == n

    def test_assumptions_list_populated(self):
        """Assumptions list should have content."""
        result = build_institutional_distance_matrix()
        assert len(result.assumptions) > 0
        assert all(isinstance(a, str) for a in result.assumptions)


class TestComputeInstitutionalDistance:
    """Test lookup function."""

    def test_lookup_returns_correct_value(self):
        """Lookup should return correct matrix value."""
        result = build_institutional_distance_matrix()

        if len(result.occupations) >= 2:
            occ_i = result.occupations[0]
            occ_j = result.occupations[1]
            expected = result.matrix[0, 1]
            actual = compute_institutional_distance(occ_i, occ_j, result)
            assert actual == expected

    def test_lookup_invalid_code_raises(self):
        """Invalid occupation code should raise ValueError."""
        result = build_institutional_distance_matrix()

        with pytest.raises(ValueError):
            compute_institutional_distance("INVALID-CODE", result.occupations[0], result)


class TestGetZoneDifference:
    """Test zone difference lookup."""

    def test_zone_difference_signs(self):
        """Zone difference should have correct sign for up/down mobility."""
        result = build_institutional_distance_matrix()

        # Find occupations in different zones
        zone_1_idx = np.where(result.zone_vector == 1)[0]
        zone_5_idx = np.where(result.zone_vector == 5)[0]

        if len(zone_1_idx) > 0 and len(zone_5_idx) > 0:
            occ_low = result.occupations[zone_1_idx[0]]
            occ_high = result.occupations[zone_5_idx[0]]

            # Moving up: low -> high should be positive
            diff_up = get_zone_difference(occ_low, occ_high, result)
            assert diff_up > 0, "Upward mobility should have positive zone difference"

            # Moving down: high -> low should be negative
            diff_down = get_zone_difference(occ_high, occ_low, result)
            assert diff_down < 0, "Downward mobility should have negative zone difference"


# ============================================================================
# Asymmetric Institutional Distance Tests (v0.6.6.0)
# ============================================================================


class TestAsymmetricInstitutionalDistance:
    """Test asymmetric (directional) institutional distance computation."""

    def test_decomposition_valid(self):
        """d_up + d_down should equal d_symmetric for all pairs."""
        result = build_asymmetric_institutional_distance()
        verification = verify_asymmetric_decomposition(result)

        assert verification["decomposition_valid"], \
            f"Decomposition invalid, max violation: {verification['decomposition_max_violation']}"

    def test_transpose_relationship(self):
        """d_up[i,j] should equal d_down[j,i]."""
        result = build_asymmetric_institutional_distance()
        verification = verify_asymmetric_decomposition(result)

        assert verification["transpose_valid"], \
            f"Transpose invalid, max violation: {verification['transpose_max_violation']}"

    def test_diagonal_zero(self):
        """All diagonals should be zero."""
        result = build_asymmetric_institutional_distance()
        verification = verify_asymmetric_decomposition(result)

        assert verification["diagonal_zero"], "Diagonal elements should be zero"

    def test_all_properties_pass(self):
        """All verification properties should pass."""
        result = build_asymmetric_institutional_distance()
        verification = verify_asymmetric_decomposition(result)

        assert verification["all_properties_pass"], \
            f"Some properties failed: {verification}"

    def test_upward_barrier_correct_direction(self):
        """d_up should be positive when Zone_j > Zone_i, zero otherwise."""
        result = build_asymmetric_institutional_distance()

        # Find indices with different zones
        zone_1_idx = np.where(result.zone_vector == 1)[0]
        zone_5_idx = np.where(result.zone_vector == 5)[0]

        if len(zone_1_idx) > 0 and len(zone_5_idx) > 0:
            i = zone_1_idx[0]  # Zone 1 origin
            j = zone_5_idx[0]  # Zone 5 destination

            # Upward: Zone 1 -> Zone 5, d_up should be ~4 (zone diff)
            assert result.d_up[i, j] >= 4.0, \
                f"Upward barrier Zone 1->5 should be >= 4, got {result.d_up[i, j]}"

            # Downward: Zone 1 -> Zone 5, d_down should be 0
            # (no barrier going "up" from 1 to 5 in the downward sense)
            assert result.d_down[i, j] == 0.0, \
                f"Downward barrier Zone 1->5 should be 0, got {result.d_down[i, j]}"

    def test_downward_barrier_correct_direction(self):
        """d_down should be positive when Zone_j < Zone_i, zero otherwise."""
        result = build_asymmetric_institutional_distance()

        # Find indices with different zones
        zone_1_idx = np.where(result.zone_vector == 1)[0]
        zone_5_idx = np.where(result.zone_vector == 5)[0]

        if len(zone_1_idx) > 0 and len(zone_5_idx) > 0:
            i = zone_5_idx[0]  # Zone 5 origin
            j = zone_1_idx[0]  # Zone 1 destination

            # Downward: Zone 5 -> Zone 1, d_down should be ~4 (zone diff)
            assert result.d_down[i, j] >= 4.0, \
                f"Downward barrier Zone 5->1 should be >= 4, got {result.d_down[i, j]}"

            # Upward: Zone 5 -> Zone 1, d_up should be 0
            assert result.d_up[i, j] == 0.0, \
                f"Upward barrier Zone 5->1 should be 0, got {result.d_up[i, j]}"

    def test_specific_zone_pair(self):
        """Spot check: Zone 2 -> Zone 4 should have d_up = 2, d_down = 0."""
        result = build_asymmetric_institutional_distance(gamma=0.0)  # Zone only

        # Find zones
        zone_2_idx = np.where(result.zone_vector == 2)[0]
        zone_4_idx = np.where(result.zone_vector == 4)[0]

        if len(zone_2_idx) > 0 and len(zone_4_idx) > 0:
            i = zone_2_idx[0]
            j = zone_4_idx[0]

            assert result.d_up[i, j] == 2.0, \
                f"Zone 2->4 d_up should be 2, got {result.d_up[i, j]}"
            assert result.d_down[i, j] == 0.0, \
                f"Zone 2->4 d_down should be 0, got {result.d_down[i, j]}"

    def test_non_negative(self):
        """All distance values should be >= 0."""
        result = build_asymmetric_institutional_distance()

        assert np.all(result.d_up >= 0), "d_up should be non-negative"
        assert np.all(result.d_down >= 0), "d_down should be non-negative"
        assert np.all(result.d_symmetric >= 0), "d_symmetric should be non-negative"

    def test_result_has_all_fields(self):
        """Result should have all expected fields."""
        result = build_asymmetric_institutional_distance()

        assert hasattr(result, "d_up")
        assert hasattr(result, "d_down")
        assert hasattr(result, "d_symmetric")
        assert hasattr(result, "occupations")
        assert hasattr(result, "zone_vector")
        assert hasattr(result, "cert_normalized")
        assert hasattr(result, "n_occupations")
        assert hasattr(result, "gamma")
        assert hasattr(result, "cert_coverage")
        assert hasattr(result, "assumptions")

    def test_matrix_dimensions_match(self):
        """All matrices should have same dimensions."""
        result = build_asymmetric_institutional_distance()
        n = result.n_occupations

        assert result.d_up.shape == (n, n)
        assert result.d_down.shape == (n, n)
        assert result.d_symmetric.shape == (n, n)

    def test_gamma_zero_equals_zone_only(self):
        """With gamma=0, only zone component should contribute."""
        result = build_asymmetric_institutional_distance(gamma=0.0)

        # Compute expected zone-only distances
        zones = result.zone_vector
        zone_diff = zones[None, :] - zones[:, None]
        expected_up = np.maximum(0, zone_diff)
        expected_down = np.maximum(0, -zone_diff)

        assert np.allclose(result.d_up, expected_up), \
            "With gamma=0, d_up should equal zone-only upward"
        assert np.allclose(result.d_down, expected_down), \
            "With gamma=0, d_down should equal zone-only downward"
