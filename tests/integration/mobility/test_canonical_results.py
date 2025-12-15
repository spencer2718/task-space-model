"""
Integration tests verifying canonical results from the paper.

These tests ensure the codebase reproduces the primary findings from
paper Section 4.4 (CPS Mobility Validation).

Canonical Results:
- n_transitions: 89,329 verified occupation transitions
- α (semantic): 2.994 (t = 98.5)
- β (institutional): 0.215 (t = 63.4)
- Log-likelihood: -205,528.9
"""

import json
from pathlib import Path

import pytest

from task_space.mobility import (
    load_canonical_results,
    ChoiceModelResult,
)


@pytest.mark.slow
class TestCanonicalClogitResults:
    """Verify we reproduce the paper's main conditional logit results."""

    @pytest.fixture
    def canonical_results(self):
        """Load canonical results from data/processed/mobility/."""
        return load_canonical_results()

    def test_sample_size(self, canonical_results):
        """Verify sample size matches paper."""
        assert canonical_results.n_transitions == 89329, \
            f"Expected 89,329 transitions, got {canonical_results.n_transitions}"

    def test_alpha_coefficient(self, canonical_results):
        """Verify semantic distance coefficient matches paper."""
        expected_alpha = 2.994
        tolerance = 0.001

        assert abs(canonical_results.alpha - expected_alpha) < tolerance, \
            f"Expected α ≈ {expected_alpha}, got {canonical_results.alpha}"

    def test_beta_coefficient(self, canonical_results):
        """Verify institutional distance coefficient matches paper."""
        expected_beta = 0.215
        tolerance = 0.001

        assert abs(canonical_results.beta - expected_beta) < tolerance, \
            f"Expected β ≈ {expected_beta}, got {canonical_results.beta}"

    def test_alpha_t_statistic(self, canonical_results):
        """Verify semantic t-statistic is in expected range."""
        # t ≈ 98.5, allow some tolerance for rounding
        assert canonical_results.alpha_t > 95, \
            f"Expected α t-stat > 95, got {canonical_results.alpha_t}"
        assert canonical_results.alpha_t < 102, \
            f"Expected α t-stat < 102, got {canonical_results.alpha_t}"

    def test_beta_t_statistic(self, canonical_results):
        """Verify institutional t-statistic is in expected range."""
        # t ≈ 63.4
        assert canonical_results.beta_t > 60, \
            f"Expected β t-stat > 60, got {canonical_results.beta_t}"
        assert canonical_results.beta_t < 67, \
            f"Expected β t-stat < 67, got {canonical_results.beta_t}"

    def test_log_likelihood(self, canonical_results):
        """Verify log-likelihood is in expected range."""
        expected_ll = -205528.9
        tolerance = 1.0

        assert abs(canonical_results.log_likelihood - expected_ll) < tolerance, \
            f"Expected LL ≈ {expected_ll}, got {canonical_results.log_likelihood}"

    def test_positive_coefficients(self, canonical_results):
        """Both coefficients should be positive (prefer lower distances)."""
        assert canonical_results.alpha > 0, "α should be positive"
        assert canonical_results.beta > 0, "β should be positive"

    def test_alpha_dominates_beta_raw(self, canonical_results):
        """α should be much larger than β in raw scale."""
        ratio = canonical_results.alpha / canonical_results.beta
        assert ratio > 10, f"α/β ratio should be > 10, got {ratio:.1f}"

    def test_both_significant(self, canonical_results):
        """Both coefficients should be highly significant."""
        # p-values should be essentially zero
        assert canonical_results.alpha_p < 1e-10, "α should be significant"
        assert canonical_results.beta_p < 1e-10, "β should be significant"

    def test_assumptions_documented(self, canonical_results):
        """Result should document modeling assumptions."""
        assert len(canonical_results.assumptions) > 0, "Should have assumptions"
        assert any("IIA" in a for a in canonical_results.assumptions), \
            "Should document IIA assumption"


@pytest.mark.slow
class TestCanonicalDataFiles:
    """Verify canonical data files exist and have expected properties."""

    def test_verified_transitions_exists(self):
        """Verified transitions parquet should exist."""
        path = Path("data/processed/mobility/verified_transitions.parquet")
        assert path.exists(), f"Missing {path}"

    def test_conditional_logit_results_exists(self):
        """Conditional logit results JSON should exist."""
        path = Path("data/processed/mobility/conditional_logit_results.json")
        assert path.exists(), f"Missing {path}"

    def test_results_json_valid(self):
        """Results JSON should be valid and contain expected keys."""
        path = Path("data/processed/mobility/conditional_logit_results.json")
        with open(path) as f:
            data = json.load(f)

        required_keys = ["n_transitions", "alpha_coef", "beta_coef", "log_likelihood"]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"


@pytest.mark.slow
class TestDistanceMatrixCaches:
    """Verify cached distance matrices exist."""

    def test_semantic_distance_cache_exists(self):
        """Semantic distance matrix cache should exist."""
        path = Path(".cache/artifacts/v1/mobility/d_sem_census.npz")
        assert path.exists(), f"Missing {path}"

    def test_institutional_distance_cache_exists(self):
        """Institutional distance matrix cache should exist."""
        path = Path(".cache/artifacts/v1/mobility/d_inst_census.npz")
        assert path.exists(), f"Missing {path}"

    def test_crosswalk_cache_exists(self):
        """Crosswalk CSV cache should exist."""
        path = Path(".cache/artifacts/v1/mobility/onet_to_census_improved.csv")
        assert path.exists(), f"Missing {path}"
