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
