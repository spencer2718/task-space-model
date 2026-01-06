"""
Unit tests for task_space.validation.spearman module.

Tests verify:
- Spearman correlation functions work with synthetic data
- Perfect monotone data yields Spearman ≈ 1
- Anti-monotone data yields Spearman ≈ -1
- Result dataclasses serialize correctly
- Bootstrap confidence intervals are reasonable
"""

import numpy as np
import pandas as pd
import pytest

from task_space.validation.spearman import (
    # Data classes
    SpearmanResult,
    PerOriginSpearmanResult,
    # Model probability methods
    aggregate_spearman_model_prob,
    per_origin_spearman_model_prob,
    # Inverse distance methods
    aggregate_spearman_inv_distance,
    per_origin_spearman_inv_distance,
    # Utilities
    spearman_with_bootstrap,
    compute_model_probability_matrix,
)


class TestSpearmanResult:
    """Tests for SpearmanResult dataclass."""

    def test_to_dict(self):
        """to_dict should produce serializable dictionary."""
        result = SpearmanResult(
            spearman=0.8567,
            p_value=0.001234,
            n_destinations=447,
            n_origins=100,
            method="test_method",
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["spearman"] == 0.8567
        assert d["p_value"] == 0.0012
        assert d["n_destinations"] == 447
        assert d["n_origins"] == 100
        assert d["method"] == "test_method"

    def test_to_dict_handles_none(self):
        """to_dict should handle None values."""
        result = SpearmanResult(
            spearman=np.nan,
            p_value=None,
            n_destinations=0,
            n_origins=0,
            method="empty",
        )

        d = result.to_dict()
        assert d["p_value"] is None


class TestPerOriginSpearmanResult:
    """Tests for PerOriginSpearmanResult dataclass."""

    def test_to_dict(self):
        """to_dict should produce serializable dictionary."""
        result = PerOriginSpearmanResult(
            mean_spearman=0.15,
            median_spearman=0.12,
            std_spearman=0.08,
            n_origins_evaluated=50,
            correlations=[0.1, 0.2, 0.15],
            min_destinations_filter=5,
            method="test",
        )

        d = result.to_dict()

        assert d["mean_spearman"] == 0.15
        assert d["median_spearman"] == 0.12
        assert d["n_origins_evaluated"] == 50
        # Correlations list is not included in to_dict (summary only)


class TestSpearmanWithBootstrap:
    """Tests for spearman_with_bootstrap function."""

    def test_perfect_monotone_correlation(self):
        """Perfectly monotone data should have Spearman ≈ 1."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        rho, (lo, hi) = spearman_with_bootstrap(x, y, n_bootstrap=100, random_state=42)

        assert abs(rho - 1.0) < 0.0001
        assert lo > 0.9
        assert hi <= 1.0

    def test_perfect_anti_monotone(self):
        """Perfectly anti-monotone data should have Spearman ≈ -1."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

        rho, (lo, hi) = spearman_with_bootstrap(x, y, n_bootstrap=100, random_state=42)

        assert abs(rho + 1.0) < 0.0001
        assert lo >= -1.0
        assert hi < -0.9

    def test_random_correlation_near_zero(self):
        """Uncorrelated random data should have Spearman near 0."""
        rng = np.random.default_rng(42)
        x = rng.random(100)
        y = rng.random(100)

        rho, (lo, hi) = spearman_with_bootstrap(x, y, n_bootstrap=100, random_state=42)

        assert abs(rho) < 0.3  # Should be close to 0
        assert lo < hi  # CI should be valid

    def test_confidence_interval_contains_point(self):
        """Point estimate should generally be within confidence interval."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1.1, 2.2, 2.9, 4.1, 5.2])

        rho, (lo, hi) = spearman_with_bootstrap(x, y, n_bootstrap=100, random_state=42)

        # Point estimate should be within or near CI
        # (Bootstrap CIs don't always contain point estimate, but should be close)
        assert lo - 0.1 <= rho <= hi + 0.1

    def test_random_state_reproducibility(self):
        """Same random_state should produce same results."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 1, 4, 3, 5])

        rho1, ci1 = spearman_with_bootstrap(x, y, n_bootstrap=50, random_state=123)
        rho2, ci2 = spearman_with_bootstrap(x, y, n_bootstrap=50, random_state=123)

        assert rho1 == rho2
        assert ci1 == ci2


class TestComputeModelProbabilityMatrix:
    """Tests for compute_model_probability_matrix function."""

    def test_rows_sum_to_one(self):
        """Each row of probability matrix should sum to 1."""
        n = 5
        d_sem = np.random.rand(n, n)
        d_inst = np.random.rand(n, n)

        P = compute_model_probability_matrix(d_sem, d_inst, gamma_sem=1.0, gamma_inst=0.1)

        row_sums = P.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)

    def test_diagonal_is_zero(self):
        """Self-transition probabilities should be 0."""
        n = 5
        d_sem = np.random.rand(n, n)
        d_inst = np.random.rand(n, n)

        P = compute_model_probability_matrix(d_sem, d_inst, gamma_sem=1.0, gamma_inst=0.1)

        np.testing.assert_array_equal(np.diag(P), 0)

    def test_closer_destinations_higher_prob(self):
        """Closer destinations should have higher probabilities."""
        n = 3
        # Origin 0 is closer to destination 1 than destination 2
        d_sem = np.array([
            [0.0, 0.1, 0.5],
            [0.1, 0.0, 0.3],
            [0.5, 0.3, 0.0],
        ])
        d_inst = np.zeros((n, n))

        P = compute_model_probability_matrix(d_sem, d_inst, gamma_sem=5.0, gamma_inst=0.0)

        # From origin 0, P(1) should be > P(2) since d(0,1) < d(0,2)
        assert P[0, 1] > P[0, 2]

    def test_all_probabilities_positive(self):
        """All non-diagonal probabilities should be positive."""
        n = 5
        d_sem = np.random.rand(n, n) + 0.1  # Ensure all positive
        d_inst = np.random.rand(n, n) + 0.1

        P = compute_model_probability_matrix(d_sem, d_inst, gamma_sem=1.0, gamma_inst=0.1)

        # Off-diagonal should be positive
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert P[i, j] > 0


class TestAggregateSpearmanModelProb:
    """Tests for aggregate_spearman_model_prob function."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        # 3 occupations
        census_codes = [100, 200, 300]

        # Probability matrix where destination 200 is most likely
        prob_matrix = np.array([
            [0.0, 0.7, 0.3],  # From 100: prefer 200
            [0.4, 0.0, 0.6],  # From 200: prefer 300
            [0.5, 0.5, 0.0],  # From 300: equal
        ])

        # Holdout data matching the probabilities
        holdout_df = pd.DataFrame({
            "origin_occ": [100, 100, 100, 200, 300, 300],
            "dest_occ": [200, 200, 200, 300, 100, 200],
        })

        return holdout_df, prob_matrix, census_codes

    def test_returns_spearman_result(self, simple_data):
        """Should return SpearmanResult instance."""
        holdout_df, prob_matrix, census_codes = simple_data

        result = aggregate_spearman_model_prob(
            holdout_df, prob_matrix, census_codes
        )

        assert isinstance(result, SpearmanResult)
        assert result.method.startswith("model_probability")

    def test_high_correlation_when_matched(self, simple_data):
        """When predictions match observations, correlation should be positive."""
        holdout_df, prob_matrix, census_codes = simple_data

        result = aggregate_spearman_model_prob(
            holdout_df, prob_matrix, census_codes
        )

        # Should have positive correlation
        assert result.spearman > 0

    def test_counts_all_destinations(self, simple_data):
        """n_destinations should equal total occupations."""
        holdout_df, prob_matrix, census_codes = simple_data

        result = aggregate_spearman_model_prob(
            holdout_df, prob_matrix, census_codes
        )

        assert result.n_destinations == len(census_codes)


class TestPerOriginSpearmanModelProb:
    """Tests for per_origin_spearman_model_prob function."""

    @pytest.fixture
    def multi_origin_data(self):
        """Create data with multiple origins having many destinations."""
        census_codes = list(range(10))

        # Random probability matrix
        rng = np.random.default_rng(42)
        prob_matrix = rng.random((10, 10))
        np.fill_diagonal(prob_matrix, 0)
        prob_matrix /= prob_matrix.sum(axis=1, keepdims=True)

        # Create holdout with several transitions per origin
        origins = []
        dests = []
        for origin in [0, 1, 2]:
            for _ in range(8):  # 8 transitions per origin
                dest = rng.choice([d for d in range(10) if d != origin])
                origins.append(origin)
                dests.append(dest)

        holdout_df = pd.DataFrame({
            "origin_occ": origins,
            "dest_occ": dests,
        })

        return holdout_df, prob_matrix, census_codes

    def test_returns_per_origin_result(self, multi_origin_data):
        """Should return PerOriginSpearmanResult instance."""
        holdout_df, prob_matrix, census_codes = multi_origin_data

        result = per_origin_spearman_model_prob(
            holdout_df, prob_matrix, census_codes, min_destinations=3
        )

        assert isinstance(result, PerOriginSpearmanResult)

    def test_respects_min_destinations(self, multi_origin_data):
        """Should only include origins meeting min_destinations threshold."""
        holdout_df, prob_matrix, census_codes = multi_origin_data

        # With high threshold, fewer origins qualify
        result_high = per_origin_spearman_model_prob(
            holdout_df, prob_matrix, census_codes, min_destinations=6
        )
        result_low = per_origin_spearman_model_prob(
            holdout_df, prob_matrix, census_codes, min_destinations=2
        )

        assert result_high.n_origins_evaluated <= result_low.n_origins_evaluated


class TestAggregateSpearmanInvDistance:
    """Tests for aggregate_spearman_inv_distance function."""

    @pytest.fixture
    def distance_data(self):
        """Create test data with distance matrix."""
        census_codes = [100, 200, 300]

        # Distance matrix (closer = smaller)
        distance_matrix = np.array([
            [0.0, 0.1, 0.5],  # 100 is close to 200
            [0.1, 0.0, 0.3],
            [0.5, 0.3, 0.0],
        ])

        # Observations should match inverse distance pattern
        holdout_df = pd.DataFrame({
            "origin_occ": [100, 100, 100, 200],
            "dest_occ": [200, 200, 300, 300],
        })

        return holdout_df, distance_matrix, census_codes

    def test_returns_spearman_result(self, distance_data):
        """Should return SpearmanResult instance."""
        holdout_df, distance_matrix, census_codes = distance_data

        result = aggregate_spearman_inv_distance(
            holdout_df, distance_matrix, census_codes
        )

        assert isinstance(result, SpearmanResult)
        assert "inverse_distance" in result.method

    def test_common_destinations_only(self, distance_data):
        """n_destinations should be common destinations count."""
        holdout_df, distance_matrix, census_codes = distance_data

        result = aggregate_spearman_inv_distance(
            holdout_df, distance_matrix, census_codes
        )

        # Only destinations with both predictions and observations
        assert result.n_destinations <= len(census_codes)


class TestIntegration:
    """Integration tests with realistic synthetic data."""

    @pytest.fixture
    def realistic_data(self):
        """Create more realistic test scenario."""
        rng = np.random.default_rng(42)
        n_occ = 20

        census_codes = list(range(1000, 1000 + n_occ))

        # Random distance matrix (symmetric)
        d = rng.random((n_occ, n_occ))
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0)
        d_sem = d * 0.5  # Scale down

        d_inst = rng.random((n_occ, n_occ)) * 0.1
        d_inst = (d_inst + d_inst.T) / 2
        np.fill_diagonal(d_inst, 0)

        # Build probability matrix
        P = compute_model_probability_matrix(d_sem, d_inst, gamma_sem=5.0, gamma_inst=0.5)

        # Sample transitions according to model
        origins = []
        dests = []
        for _ in range(500):
            origin_idx = rng.integers(0, n_occ)
            dest_idx = rng.choice(n_occ, p=P[origin_idx])
            origins.append(census_codes[origin_idx])
            dests.append(census_codes[dest_idx])

        holdout_df = pd.DataFrame({
            "origin_occ": origins,
            "dest_occ": dests,
        })

        return holdout_df, P, d_sem, census_codes

    def test_model_sampled_data_high_correlation(self, realistic_data):
        """Data sampled from model should show positive correlation."""
        holdout_df, prob_matrix, d_sem, census_codes = realistic_data

        result = aggregate_spearman_model_prob(
            holdout_df, prob_matrix, census_codes
        )

        # Should have strong positive correlation since data matches model
        assert result.spearman > 0.5

    def test_per_origin_positive_correlation(self, realistic_data):
        """Per-origin correlations should generally be positive."""
        holdout_df, prob_matrix, d_sem, census_codes = realistic_data

        result = per_origin_spearman_model_prob(
            holdout_df, prob_matrix, census_codes, min_destinations=3
        )

        # Mean should be positive (data matches model)
        assert result.mean_spearman > 0
