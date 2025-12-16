"""
Unit tests for reallocation module.

Tests:
- Employment loading and mapping
- Exposed occupation identification
- Destination probability computation (softmax correctness)
- Reallocation flow aggregation
- Absorption ranking
- Capacity/credential constraint flagging
- Validation metrics
"""

import numpy as np
import pandas as pd
import pytest

from task_space.validation.reallocation import (
    get_exposed_occupations,
    compute_destination_probabilities,
    aggregate_reallocation_flows,
    compute_absorption_ranking,
    validate_against_holdout,
    flag_capacity_constraints,
    split_feasible_constrained,
    compute_validation_verdict,
    CREDENTIAL_GATED_OCCUPATIONS,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_aioe_df():
    """Mock AIOE scores at Census level."""
    return pd.DataFrame({
        "census_code": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "aioe_score": [0.3, 0.5, 0.7, 0.9, 0.95, 0.4, 0.6, 0.8, 0.85, 0.92],
    })


@pytest.fixture
def mock_employment_df():
    """Mock employment data at Census level."""
    return pd.DataFrame({
        "census_code": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "tot_emp": [100000, 80000, 60000, 40000, 30000, 90000, 70000, 50000, 45000, 35000],
    })


@pytest.fixture
def mock_wasserstein_matrix():
    """Mock 10x10 Wasserstein distance matrix."""
    np.random.seed(42)
    n = 10
    d = np.random.rand(n, n) * 0.5
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0)
    return d


@pytest.fixture
def mock_inst_matrix():
    """Mock 10x10 institutional distance matrix."""
    np.random.seed(43)
    n = 10
    d = np.random.rand(n, n) * 2
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0)
    return d


@pytest.fixture
def mock_census_codes():
    """Census codes matching the mock matrices."""
    return [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


# =============================================================================
# Test: Exposed Occupation Identification
# =============================================================================


def test_get_exposed_occupations_quartile(mock_aioe_df, mock_employment_df):
    """Test that exposed occupations are top quartile by AIOE."""
    exposed_df, threshold = get_exposed_occupations(
        mock_aioe_df, mock_employment_df, quartile=0.75
    )

    # Should return top 25% (approximately 2-3 occupations)
    assert len(exposed_df) > 0
    assert len(exposed_df) <= len(mock_aioe_df) * 0.30  # Allow some margin

    # All returned should be above threshold
    assert (exposed_df["aioe_score"] >= threshold).all()


def test_get_exposed_occupations_has_employment(mock_aioe_df, mock_employment_df):
    """Test that exposed occupations have employment data."""
    exposed_df, _ = get_exposed_occupations(
        mock_aioe_df, mock_employment_df, quartile=0.75
    )

    # All should have employment
    assert "tot_emp" in exposed_df.columns
    assert (exposed_df["tot_emp"] > 0).all()


def test_get_exposed_occupations_empty_employment():
    """Test behavior with no matching employment."""
    aioe_df = pd.DataFrame({
        "census_code": [10, 20, 30],
        "aioe_score": [0.9, 0.95, 0.98],
    })
    employment_df = pd.DataFrame({
        "census_code": [100, 200, 300],  # No overlap
        "tot_emp": [10000, 20000, 30000],
    })

    exposed_df, _ = get_exposed_occupations(aioe_df, employment_df, quartile=0.75)

    assert len(exposed_df) == 0


# =============================================================================
# Test: Destination Probability Computation
# =============================================================================


def test_destination_probabilities_sum_to_one(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes
):
    """Test that probabilities sum to 1 for each origin."""
    prob_matrix = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=5.0, gamma_inst=0.5,
        exclude_self=True,
    )

    # Each row should sum to 1
    row_sums = prob_matrix.sum(axis=1)
    np.testing.assert_array_almost_equal(row_sums, np.ones(10), decimal=6)


def test_destination_probabilities_self_excluded(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes
):
    """Test that self-transitions have zero probability when excluded."""
    prob_matrix = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=5.0, gamma_inst=0.5,
        exclude_self=True,
    )

    # Diagonal should be zero
    np.testing.assert_array_almost_equal(np.diag(prob_matrix), np.zeros(10))


def test_destination_probabilities_self_included(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes
):
    """Test that self-transitions have positive probability when included."""
    prob_matrix = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=5.0, gamma_inst=0.5,
        exclude_self=False,
    )

    # Diagonal should be positive (self-distance is 0, so highest utility)
    assert (np.diag(prob_matrix) > 0).all()


def test_destination_probabilities_higher_gamma_concentrates(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes
):
    """Test that higher gamma leads to more concentrated probabilities."""
    prob_low = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=1.0, gamma_inst=0.1,
        exclude_self=True,
    )

    prob_high = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=10.0, gamma_inst=1.0,
        exclude_self=True,
    )

    # Higher gamma should have more concentrated probabilities (higher max)
    assert prob_high.max() > prob_low.max()


def test_destination_probabilities_all_positive(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes
):
    """Test that all off-diagonal probabilities are positive."""
    prob_matrix = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=5.0, gamma_inst=0.5,
        exclude_self=True,
    )

    # Off-diagonal should be positive
    off_diag = prob_matrix[~np.eye(10, dtype=bool)]
    assert (off_diag > 0).all()


# =============================================================================
# Test: Reallocation Flow Aggregation
# =============================================================================


def test_aggregate_flows_total_equals_employment(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
    mock_aioe_df, mock_employment_df
):
    """Test that total flows equal displaced employment."""
    exposed_df, _ = get_exposed_occupations(
        mock_aioe_df, mock_employment_df, quartile=0.75
    )

    prob_matrix = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=5.0, gamma_inst=0.5,
        exclude_self=True,
    )

    flows_df = aggregate_reallocation_flows(
        prob_matrix, mock_census_codes, exposed_df,
        displacement_rate=1.0,
    )

    # Total flows should equal exposed employment
    expected_total = exposed_df["tot_emp"].sum()
    actual_total = flows_df["flow"].sum()

    np.testing.assert_almost_equal(actual_total, expected_total, decimal=0)


def test_aggregate_flows_displacement_rate(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
    mock_aioe_df, mock_employment_df
):
    """Test that displacement rate scales flows correctly."""
    exposed_df, _ = get_exposed_occupations(
        mock_aioe_df, mock_employment_df, quartile=0.75
    )

    prob_matrix = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=5.0, gamma_inst=0.5,
        exclude_self=True,
    )

    flows_full = aggregate_reallocation_flows(
        prob_matrix, mock_census_codes, exposed_df,
        displacement_rate=1.0,
    )

    flows_half = aggregate_reallocation_flows(
        prob_matrix, mock_census_codes, exposed_df,
        displacement_rate=0.5,
    )

    # Half displacement rate should give half flows
    np.testing.assert_almost_equal(
        flows_half["flow"].sum(),
        flows_full["flow"].sum() * 0.5,
        decimal=0,
    )


def test_aggregate_flows_has_required_columns(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
    mock_aioe_df, mock_employment_df
):
    """Test that flows DataFrame has required columns."""
    exposed_df, _ = get_exposed_occupations(
        mock_aioe_df, mock_employment_df, quartile=0.75
    )

    prob_matrix = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=5.0, gamma_inst=0.5,
        exclude_self=True,
    )

    flows_df = aggregate_reallocation_flows(
        prob_matrix, mock_census_codes, exposed_df,
        displacement_rate=1.0,
    )

    required_cols = ["origin", "dest", "flow", "origin_emp", "prob"]
    for col in required_cols:
        assert col in flows_df.columns, f"Missing column: {col}"


# =============================================================================
# Test: Absorption Ranking
# =============================================================================


def test_absorption_ranking_sorted(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
    mock_aioe_df, mock_employment_df
):
    """Test that absorption ranking is sorted descending."""
    exposed_df, _ = get_exposed_occupations(
        mock_aioe_df, mock_employment_df, quartile=0.75
    )

    prob_matrix = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=5.0, gamma_inst=0.5,
        exclude_self=True,
    )

    flows_df = aggregate_reallocation_flows(
        prob_matrix, mock_census_codes, exposed_df,
        displacement_rate=1.0,
    )

    absorption_df = compute_absorption_ranking(
        flows_df, mock_employment_df, mock_census_codes
    )

    # Should be sorted by total_absorption descending
    absorptions = absorption_df["total_absorption"].values
    assert (absorptions[:-1] >= absorptions[1:]).all()


def test_absorption_ranking_has_absorption_rate(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
    mock_aioe_df, mock_employment_df
):
    """Test that absorption rate is computed correctly."""
    exposed_df, _ = get_exposed_occupations(
        mock_aioe_df, mock_employment_df, quartile=0.75
    )

    prob_matrix = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=5.0, gamma_inst=0.5,
        exclude_self=True,
    )

    flows_df = aggregate_reallocation_flows(
        prob_matrix, mock_census_codes, exposed_df,
        displacement_rate=1.0,
    )

    absorption_df = compute_absorption_ranking(
        flows_df, mock_employment_df, mock_census_codes
    )

    # Absorption rate should be total_absorption / current_emp
    for _, row in absorption_df.iterrows():
        if pd.notna(row["current_emp"]) and row["current_emp"] > 0:
            expected_rate = row["total_absorption"] / row["current_emp"]
            np.testing.assert_almost_equal(row["absorption_rate"], expected_rate)


def test_absorption_ranking_with_occupation_names(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
    mock_aioe_df, mock_employment_df
):
    """Test that occupation names are included when provided."""
    exposed_df, _ = get_exposed_occupations(
        mock_aioe_df, mock_employment_df, quartile=0.75
    )

    prob_matrix = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=5.0, gamma_inst=0.5,
        exclude_self=True,
    )

    flows_df = aggregate_reallocation_flows(
        prob_matrix, mock_census_codes, exposed_df,
        displacement_rate=1.0,
    )

    occupation_names = {10: "Test Occupation A", 20: "Test Occupation B"}

    absorption_df = compute_absorption_ranking(
        flows_df, mock_employment_df, mock_census_codes,
        occupation_names=occupation_names,
    )

    # Should have occupation_name column
    assert "occupation_name" in absorption_df.columns


# =============================================================================
# Test: Holdout Validation
# =============================================================================


def test_validate_against_holdout_returns_metrics(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
    mock_aioe_df, mock_employment_df
):
    """Test that holdout validation returns required metrics."""
    exposed_df, _ = get_exposed_occupations(
        mock_aioe_df, mock_employment_df, quartile=0.75
    )

    prob_matrix = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=5.0, gamma_inst=0.5,
        exclude_self=True,
    )

    flows_df = aggregate_reallocation_flows(
        prob_matrix, mock_census_codes, exposed_df,
        displacement_rate=1.0,
    )

    exposed_codes = list(exposed_df["census_code"])

    # Mock holdout data with sufficient transitions
    holdout_df = pd.DataFrame({
        "origin_occ": [40, 40, 50, 50, 90, 90, 100, 100] * 10,
        "dest_occ": [10, 20, 30, 60, 70, 80, 10, 20] * 10,
    })

    result = validate_against_holdout(
        flows_df=flows_df,
        holdout_df=holdout_df,
        census_codes=mock_census_codes,
        exposed_codes=exposed_codes,
        prob_matrix=prob_matrix,
    )

    assert "n_observed_transitions" in result
    assert "spearman_correlation" in result or "warning" in result
    assert "top5_overlap" in result or "warning" in result


def test_validate_against_holdout_insufficient_data(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes
):
    """Test that insufficient exposed transitions returns warning."""
    flows_df = pd.DataFrame({
        "origin": [10, 10],
        "dest": [20, 30],
        "flow": [100, 200],
    })

    prob_matrix = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=5.0, gamma_inst=0.5,
        exclude_self=True,
    )

    # Holdout has transitions but not from exposed occupations
    holdout_df = pd.DataFrame({
        "origin_occ": [10, 10, 20, 20],  # Not in exposed_codes
        "dest_occ": [30, 40, 50, 60],
    })

    exposed_codes = [100, 200]  # None in holdout

    result = validate_against_holdout(
        flows_df=flows_df,
        holdout_df=holdout_df,
        census_codes=mock_census_codes,
        exposed_codes=exposed_codes,
        prob_matrix=prob_matrix,
    )

    # Should have low n_observed_transitions
    assert result["n_observed_transitions"] < 20


# =============================================================================
# Test: Capacity and Credential Constraints
# =============================================================================


def test_flag_capacity_constraints(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
    mock_aioe_df, mock_employment_df
):
    """Test that capacity constraints are flagged correctly."""
    exposed_df, _ = get_exposed_occupations(
        mock_aioe_df, mock_employment_df, quartile=0.75
    )

    prob_matrix = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=5.0, gamma_inst=0.5,
        exclude_self=True,
    )

    flows_df = aggregate_reallocation_flows(
        prob_matrix, mock_census_codes, exposed_df,
        displacement_rate=1.0,
    )

    absorption_df = compute_absorption_ranking(
        flows_df, mock_employment_df, mock_census_codes
    )

    # Flag with 50% threshold
    flagged_df = flag_capacity_constraints(absorption_df, capacity_threshold=0.5)

    # Should have new columns
    assert "capacity_constrained" in flagged_df.columns
    assert "credential_gated" in flagged_df.columns
    assert "constraint_type" in flagged_df.columns
    assert "is_constrained" in flagged_df.columns

    # High absorption rates should be flagged
    high_rate = flagged_df[flagged_df["absorption_rate"] > 0.5]
    if len(high_rate) > 0:
        assert high_rate["capacity_constrained"].all()


def test_split_feasible_constrained(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
    mock_aioe_df, mock_employment_df
):
    """Test splitting into feasible vs constrained."""
    exposed_df, _ = get_exposed_occupations(
        mock_aioe_df, mock_employment_df, quartile=0.75
    )

    prob_matrix = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=5.0, gamma_inst=0.5,
        exclude_self=True,
    )

    flows_df = aggregate_reallocation_flows(
        prob_matrix, mock_census_codes, exposed_df,
        displacement_rate=1.0,
    )

    absorption_df = compute_absorption_ranking(
        flows_df, mock_employment_df, mock_census_codes
    )

    feasible, constrained = split_feasible_constrained(absorption_df, 0.5)

    # All feasible should have is_constrained = False
    if len(feasible) > 0:
        assert not feasible["is_constrained"].any()

    # All constrained should have is_constrained = True
    if len(constrained) > 0:
        assert constrained["is_constrained"].all()


def test_validation_verdict_logic():
    """Test verdict computation logic."""
    # Validated: spearman > 0.3 AND top5 >= 0.4
    assert compute_validation_verdict(0.4, 0.5) == "validated"

    # Partial: spearman > 0.1 OR top5 >= 0.2
    assert compute_validation_verdict(0.2, 0.1) == "partial"
    assert compute_validation_verdict(0.05, 0.3) == "partial"

    # Failed: below thresholds
    assert compute_validation_verdict(0.05, 0.1) == "failed"

    # Insufficient data
    assert compute_validation_verdict(None, 0.5) == "insufficient_data"
    assert compute_validation_verdict(0.3, None) == "insufficient_data"


def test_credential_gated_occupations_exist():
    """Test that credential-gated occupations are defined."""
    assert len(CREDENTIAL_GATED_OCCUPATIONS) > 0
    # Should include teachers
    assert any("Teacher" in v for v in CREDENTIAL_GATED_OCCUPATIONS.values())
    # Should include nurses
    assert any("Nurse" in v for v in CREDENTIAL_GATED_OCCUPATIONS.values())


# =============================================================================
# Integration Test: Full Pipeline
# =============================================================================


def test_full_pipeline(
    mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
    mock_aioe_df, mock_employment_df
):
    """Test full reallocation pipeline runs without error."""
    # 1. Get exposed occupations
    exposed_df, threshold = get_exposed_occupations(
        mock_aioe_df, mock_employment_df, quartile=0.75
    )
    assert len(exposed_df) > 0

    # 2. Compute probabilities
    prob_matrix = compute_destination_probabilities(
        mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes,
        gamma_sem=5.0, gamma_inst=0.5,
        exclude_self=True,
    )
    assert prob_matrix.shape == (10, 10)

    # 3. Aggregate flows
    flows_df = aggregate_reallocation_flows(
        prob_matrix, mock_census_codes, exposed_df,
        displacement_rate=1.0,
    )
    assert len(flows_df) > 0

    # 4. Compute ranking
    absorption_df = compute_absorption_ranking(
        flows_df, mock_employment_df, mock_census_codes
    )
    assert len(absorption_df) > 0

    # 5. Verify conservation
    total_flows = flows_df["flow"].sum()
    total_exposed = exposed_df["tot_emp"].sum()
    np.testing.assert_almost_equal(total_flows, total_exposed, decimal=0)
