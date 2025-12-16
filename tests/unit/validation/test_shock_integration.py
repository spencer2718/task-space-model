"""
Unit tests for shock integration module.

Tests:
- AIOE coverage computation
- Partition produces non-empty exposed set
- Historical baseline rows sum to 1
"""

import numpy as np
import pandas as pd
import pytest

from task_space.validation.shock_integration import (
    get_aioe_by_soc_dataframe,
    map_aioe_to_census,
    compute_aioe_geometry_correlations,
    partition_transitions_by_exposure,
    compute_historical_baseline,
    compute_uniform_baseline,
    evaluate_model_on_holdout,
    compute_verdict,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_aioe_census_df():
    """Mock AIOE scores at Census level."""
    return pd.DataFrame({
        "census_code": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "aioe_score": [0.5, 1.2, 0.8, 1.5, 0.3, 0.9, 1.1, 0.6, 1.3, 0.4],
        "n_soc_codes": [1, 2, 1, 1, 1, 2, 1, 1, 1, 1],
    })


@pytest.fixture
def mock_transitions_df():
    """Mock CPS transition data."""
    return pd.DataFrame({
        "CPSIDP": list(range(100)),
        "YEARMONTH": [202401] * 50 + [202301] * 50,
        "origin_occ": [10, 20, 30, 40, 50] * 20,
        "dest_occ": [20, 30, 40, 50, 60] * 20,
        "AGE": [35] * 100,
        "SEX": [1] * 100,
    })


@pytest.fixture
def mock_wasserstein_matrix():
    """Mock 10x10 Wasserstein distance matrix."""
    np.random.seed(42)
    # Create symmetric distance matrix
    n = 10
    d = np.random.rand(n, n)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0)
    return d


@pytest.fixture
def mock_census_codes():
    """Census codes matching the mock matrices."""
    return [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


# =============================================================================
# Test: AIOE Coverage
# =============================================================================


def test_aioe_coverage_computation(mock_aioe_census_df, mock_wasserstein_matrix, mock_census_codes):
    """Test that AIOE coverage is computed correctly."""
    result = compute_aioe_geometry_correlations(
        mock_aioe_census_df, mock_wasserstein_matrix, mock_census_codes
    )

    # All 10 codes should match
    assert result["aioe_coverage"] == 1.0
    assert result["n_matched_occupations"] == 10


def test_aioe_coverage_partial_match(mock_wasserstein_matrix):
    """Test coverage with partial AIOE data."""
    # AIOE only has 5 of 10 codes
    partial_aioe = pd.DataFrame({
        "census_code": [10, 20, 30, 40, 50],
        "aioe_score": [0.5, 1.2, 0.8, 1.5, 0.3],
    })
    census_codes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    result = compute_aioe_geometry_correlations(
        partial_aioe, mock_wasserstein_matrix, census_codes
    )

    assert result["aioe_coverage"] == 0.5
    assert result["n_matched_occupations"] == 5


def test_aioe_wasserstein_correlation_bounds(mock_aioe_census_df, mock_wasserstein_matrix, mock_census_codes):
    """Test that correlation is within valid bounds."""
    result = compute_aioe_geometry_correlations(
        mock_aioe_census_df, mock_wasserstein_matrix, mock_census_codes
    )

    corr = result["aioe_wasserstein_corr"]
    assert corr is not None
    assert -1.0 <= corr <= 1.0


# =============================================================================
# Test: Partition by Exposure
# =============================================================================


def test_partition_produces_nonempty_exposed(mock_transitions_df, mock_aioe_census_df):
    """Test that partitioning produces non-empty exposed set."""
    exposed, unexposed = partition_transitions_by_exposure(
        mock_transitions_df, mock_aioe_census_df, quartile=0.75
    )

    assert len(exposed) > 0, "Exposed set should not be empty"
    assert len(unexposed) > 0, "Unexposed set should not be empty"


def test_partition_disjoint_sets(mock_transitions_df, mock_aioe_census_df):
    """Test that exposed and unexposed sets are disjoint and cover all with AIOE."""
    exposed, unexposed = partition_transitions_by_exposure(
        mock_transitions_df, mock_aioe_census_df, quartile=0.75
    )

    # Check disjoint
    exposed_ids = set(exposed.index)
    unexposed_ids = set(unexposed.index)
    assert len(exposed_ids & unexposed_ids) == 0, "Sets should be disjoint"


def test_partition_quartile_threshold(mock_transitions_df, mock_aioe_census_df):
    """Test that exposed contains top quartile of AIOE."""
    exposed, unexposed = partition_transitions_by_exposure(
        mock_transitions_df, mock_aioe_census_df, quartile=0.75
    )

    # Exposed should have higher mean AIOE than unexposed
    if len(exposed) > 0 and len(unexposed) > 0:
        mean_exposed = exposed["origin_aioe"].mean()
        mean_unexposed = unexposed["origin_aioe"].mean()
        assert mean_exposed >= mean_unexposed, "Exposed should have higher AIOE"


# =============================================================================
# Test: Historical Baseline
# =============================================================================


def test_historical_baseline_rows_sum_to_one(mock_transitions_df, mock_census_codes):
    """Test that historical baseline probability rows sum to 1."""
    prob_matrix = compute_historical_baseline(mock_transitions_df, mock_census_codes)

    row_sums = prob_matrix.sum(axis=1)
    np.testing.assert_array_almost_equal(
        row_sums, np.ones(len(mock_census_codes)),
        decimal=5,
        err_msg="All rows should sum to 1"
    )


def test_historical_baseline_nonnegative(mock_transitions_df, mock_census_codes):
    """Test that all probabilities are non-negative."""
    prob_matrix = compute_historical_baseline(mock_transitions_df, mock_census_codes)

    assert np.all(prob_matrix >= 0), "All probabilities should be non-negative"


def test_historical_baseline_correct_shape(mock_transitions_df, mock_census_codes):
    """Test baseline matrix has correct shape."""
    prob_matrix = compute_historical_baseline(mock_transitions_df, mock_census_codes)

    n = len(mock_census_codes)
    assert prob_matrix.shape == (n, n), f"Expected ({n}, {n}), got {prob_matrix.shape}"


def test_uniform_baseline_rows_sum_to_one(mock_census_codes):
    """Test that uniform baseline rows sum to 1."""
    n = len(mock_census_codes)
    prob_matrix = compute_uniform_baseline(n)

    row_sums = prob_matrix.sum(axis=1)
    np.testing.assert_array_almost_equal(
        row_sums, np.ones(n),
        decimal=10,
        err_msg="Uniform baseline rows should sum to 1"
    )


def test_uniform_baseline_all_equal(mock_census_codes):
    """Test that uniform baseline has equal probabilities."""
    n = len(mock_census_codes)
    prob_matrix = compute_uniform_baseline(n)

    expected = 1.0 / n
    np.testing.assert_array_almost_equal(
        prob_matrix, np.full((n, n), expected),
        decimal=10,
    )


# =============================================================================
# Test: Verdict Logic
# =============================================================================


def test_verdict_proceed_strong():
    """Test verdict when geometry beats historical by >100."""
    verdict = compute_verdict(
        geometry_ll=-1000,
        historical_ll=-1200,
        uniform_ll=-5000,
    )
    assert verdict == "proceed_strong"


def test_verdict_proceed_validated():
    """Test verdict when geometry and historical are close."""
    verdict = compute_verdict(
        geometry_ll=-1000,
        historical_ll=-1050,  # Within 100
        uniform_ll=-5000,
    )
    assert verdict == "proceed_validated"


def test_verdict_proceed_cautious():
    """Test verdict when historical beats geometry but geometry beats uniform."""
    verdict = compute_verdict(
        geometry_ll=-1500,
        historical_ll=-1000,  # Historical better by >100
        uniform_ll=-5000,     # But geometry still beats uniform
    )
    assert verdict == "proceed_cautious"


def test_verdict_stop():
    """Test verdict when geometry is close to uniform."""
    verdict = compute_verdict(
        geometry_ll=-5000,
        historical_ll=-1000,
        uniform_ll=-5050,  # Geometry close to uniform
    )
    assert verdict == "stop"


# =============================================================================
# Test: Evaluation Metrics
# =============================================================================


def test_evaluate_model_returns_required_keys(mock_census_codes):
    """Test that evaluation returns all required metric keys."""
    n = len(mock_census_codes)
    model_probs = compute_uniform_baseline(n)
    historical_probs = compute_uniform_baseline(n)

    # Create simple holdout
    holdout = pd.DataFrame({
        "origin_occ": [10, 20, 30],
        "dest_occ": [20, 30, 40],
    })

    metrics = evaluate_model_on_holdout(
        model_probs, historical_probs, holdout, mock_census_codes
    )

    required_keys = [
        "geometry_ll",
        "baseline_historical_ll",
        "baseline_uniform_ll",
        "geometry_top5_acc",
        "baseline_historical_top5_acc",
        "n_evaluated",
    ]

    for key in required_keys:
        assert key in metrics, f"Missing required key: {key}"


def test_evaluate_model_top5_accuracy_bounds(mock_census_codes):
    """Test that top-5 accuracy is within [0, 1]."""
    n = len(mock_census_codes)
    model_probs = compute_uniform_baseline(n)
    historical_probs = compute_uniform_baseline(n)

    holdout = pd.DataFrame({
        "origin_occ": [10, 20, 30, 40, 50],
        "dest_occ": [20, 30, 40, 50, 60],
    })

    metrics = evaluate_model_on_holdout(
        model_probs, historical_probs, holdout, mock_census_codes
    )

    assert 0.0 <= metrics["geometry_top5_acc"] <= 1.0
    assert 0.0 <= metrics["baseline_historical_top5_acc"] <= 1.0


# =============================================================================
# Integration Test: Real AIOE Data (if available)
# =============================================================================


@pytest.mark.slow
def test_real_aioe_data_loads():
    """Test that real AIOE data loads correctly."""
    try:
        aioe_df = get_aioe_by_soc_dataframe(use_lm=True)
        assert len(aioe_df) > 0, "AIOE data should not be empty"
        assert "soc_code" in aioe_df.columns
        assert "aioe_score" in aioe_df.columns
    except FileNotFoundError:
        pytest.skip("AIOE data not available")


@pytest.mark.slow
def test_real_aioe_to_census_mapping():
    """Test that AIOE maps to Census codes correctly."""
    try:
        aioe_df = get_aioe_by_soc_dataframe(use_lm=True)
        census_df = map_aioe_to_census(aioe_df)

        assert len(census_df) > 0, "Census mapping should produce results"
        assert "census_code" in census_df.columns
        assert "aioe_score" in census_df.columns
        assert "n_soc_codes" in census_df.columns

        # Check coverage is reasonable
        assert len(census_df) >= 100, "Should have at least 100 Census codes with AIOE"
    except FileNotFoundError:
        pytest.skip("AIOE data or crosswalk not available")
