"""
Unit tests for scaled costs module.

Tests:
- OES wage coverage of Census occupations
- β_wage has correct sign (positive)
- Switching cost computation is correct given mock coefficients
- All three model variants estimate without error
"""

import numpy as np
import pandas as pd
import pytest

from task_space.validation.scaled_costs import (
    load_oes_wages_by_census,
    get_wage_coverage,
    build_choice_dataset_with_wages,
    estimate_scaled_model,
    compute_switching_costs,
    compute_median_distances,
    compute_verdict,
    ScaledModelResult,
)


# =============================================================================
# Fixtures (shared fixtures from tests/conftest.py)
# =============================================================================
# mock_wages_df, mock_transitions_df, mock_wasserstein_matrix, mock_inst_matrix,
# and mock_census_codes are defined in tests/conftest.py and auto-discovered.


# =============================================================================
# Test: OES Wage Coverage
# =============================================================================


def test_wage_coverage_full_match(mock_wages_df, mock_census_codes):
    """Test coverage when all codes have wages."""
    coverage = get_wage_coverage(mock_wages_df, mock_census_codes)
    assert coverage == 1.0


def test_wage_coverage_partial_match(mock_census_codes):
    """Test coverage with partial wage data."""
    partial_wages = pd.DataFrame({
        "census_code": [10, 20, 30, 40, 50],  # Only 5 of 10
        "mean_annual_wage": [50000, 60000, 70000, 80000, 55000],
    })
    coverage = get_wage_coverage(partial_wages, mock_census_codes)
    assert coverage == 0.5


def test_wage_coverage_empty():
    """Test coverage with empty wage data."""
    empty_wages = pd.DataFrame({"census_code": [], "mean_annual_wage": []})
    coverage = get_wage_coverage(empty_wages, [10, 20, 30])
    assert coverage == 0.0


# =============================================================================
# Test: Choice Dataset Building
# =============================================================================


def test_choice_dataset_has_required_columns(
    mock_transitions_df, mock_wasserstein_matrix, mock_inst_matrix,
    mock_wages_df, mock_census_codes
):
    """Test that choice dataset has all required columns."""
    choice_df = build_choice_dataset_with_wages(
        mock_transitions_df,
        mock_wasserstein_matrix,
        mock_inst_matrix,
        mock_wages_df,
        mock_census_codes,
        n_alternatives=5,
    )

    required_cols = [
        "case_id", "alt_id", "chosen",
        "d_wasserstein", "d_inst",
        "log_wage_dest", "log_wage_ratio", "wage_dest"
    ]
    for col in required_cols:
        assert col in choice_df.columns, f"Missing column: {col}"


def test_choice_dataset_chosen_structure(
    mock_transitions_df, mock_wasserstein_matrix, mock_inst_matrix,
    mock_wages_df, mock_census_codes
):
    """Test that each case has exactly one chosen alternative."""
    choice_df = build_choice_dataset_with_wages(
        mock_transitions_df,
        mock_wasserstein_matrix,
        mock_inst_matrix,
        mock_wages_df,
        mock_census_codes,
        n_alternatives=5,
    )

    # Each case should have exactly one chosen
    chosen_counts = choice_df.groupby("case_id")["chosen"].sum()
    assert (chosen_counts == 1).all(), "Each case should have exactly one chosen"


def test_choice_dataset_alternatives_count(
    mock_transitions_df, mock_wasserstein_matrix, mock_inst_matrix,
    mock_wages_df, mock_census_codes
):
    """Test that each case has correct number of alternatives."""
    n_alts = 5
    choice_df = build_choice_dataset_with_wages(
        mock_transitions_df,
        mock_wasserstein_matrix,
        mock_inst_matrix,
        mock_wages_df,
        mock_census_codes,
        n_alternatives=n_alts,
    )

    # Each case should have n_alts + 1 rows (chosen + alternatives)
    rows_per_case = choice_df.groupby("case_id").size()
    assert (rows_per_case == n_alts + 1).all()


# =============================================================================
# Test: Model Estimation
# =============================================================================


def test_estimate_model_m1_runs(
    mock_transitions_df, mock_wasserstein_matrix, mock_inst_matrix,
    mock_wages_df, mock_census_codes
):
    """Test that M1 model estimates without error."""
    choice_df = build_choice_dataset_with_wages(
        mock_transitions_df,
        mock_wasserstein_matrix,
        mock_inst_matrix,
        mock_wages_df,
        mock_census_codes,
        n_alternatives=5,
    )

    result = estimate_scaled_model(choice_df, model_variant="M1")

    assert result.model_variant == "M1"
    assert result.n_cases > 0
    assert result.n_obs > 0
    assert not np.isnan(result.log_likelihood)


def test_estimate_model_m2_runs(
    mock_transitions_df, mock_wasserstein_matrix, mock_inst_matrix,
    mock_wages_df, mock_census_codes
):
    """Test that M2 model estimates without error."""
    choice_df = build_choice_dataset_with_wages(
        mock_transitions_df,
        mock_wasserstein_matrix,
        mock_inst_matrix,
        mock_wages_df,
        mock_census_codes,
        n_alternatives=5,
    )

    result = estimate_scaled_model(choice_df, model_variant="M2")

    assert result.model_variant == "M2"
    assert not np.isnan(result.log_likelihood)


def test_estimate_model_m3_runs(
    mock_transitions_df, mock_wasserstein_matrix, mock_inst_matrix,
    mock_wages_df, mock_census_codes
):
    """Test that M3 model estimates without error."""
    choice_df = build_choice_dataset_with_wages(
        mock_transitions_df,
        mock_wasserstein_matrix,
        mock_inst_matrix,
        mock_wages_df,
        mock_census_codes,
        n_alternatives=5,
    )

    result = estimate_scaled_model(choice_df, model_variant="M3")

    assert result.model_variant == "M3"
    assert not np.isnan(result.log_likelihood)


def test_estimate_model_invalid_variant(
    mock_transitions_df, mock_wasserstein_matrix, mock_inst_matrix,
    mock_wages_df, mock_census_codes
):
    """Test that invalid model variant raises error."""
    choice_df = build_choice_dataset_with_wages(
        mock_transitions_df,
        mock_wasserstein_matrix,
        mock_inst_matrix,
        mock_wages_df,
        mock_census_codes,
        n_alternatives=5,
    )

    with pytest.raises(ValueError, match="Unknown model variant"):
        estimate_scaled_model(choice_df, model_variant="M4")


# =============================================================================
# Test: Switching Cost Computation
# =============================================================================


def test_switching_cost_computation_m1():
    """Test switching cost computation with known coefficients."""
    # Create mock result with known values
    mock_result = ScaledModelResult(
        gamma_sem=5.0,      # Workers prefer closer (positive coef on neg distance)
        gamma_sem_se=0.1,
        gamma_sem_t=50.0,
        gamma_sem_p=0.0,
        gamma_inst=0.5,
        gamma_inst_se=0.05,
        gamma_inst_t=10.0,
        gamma_inst_p=0.0,
        beta_wage=2.5,      # Workers prefer higher wages
        beta_wage_se=0.1,
        beta_wage_t=25.0,
        beta_wage_p=0.0,
        log_likelihood=-10000,
        n_cases=1000,
        n_obs=11000,
        converged=True,
        model_variant="M1",
    )

    costs = compute_switching_costs(
        mock_result,
        d_wasserstein_median=0.2,
        d_inst_median=1.0,
        mean_annual_wage=60000,
    )

    # MRS_sem = -gamma_sem / beta_wage = -5.0 / 2.5 = -2.0
    # But since gamma is on negated distance, the actual MRS is 5.0 / 2.5 = 2.0
    expected_mrs_sem = 5.0 / 2.5  # = 2.0
    expected_mrs_inst = 0.5 / 2.5  # = 0.2

    assert abs(costs["sc_sem_per_unit"] - expected_mrs_sem) < 0.01
    assert abs(costs["sc_inst_per_unit"] - expected_mrs_inst) < 0.01

    # Typical cost = mrs_sem * d_sem_median + mrs_inst * d_inst_median
    expected_typical = 2.0 * 0.2 + 0.2 * 1.0  # = 0.4 + 0.2 = 0.6
    assert abs(costs["sc_typical_wage_years"] - expected_typical) < 0.01


def test_switching_cost_positive():
    """Test that switching costs are positive when signs are correct."""
    mock_result = ScaledModelResult(
        gamma_sem=8.0,
        gamma_sem_se=0.1,
        gamma_sem_t=80.0,
        gamma_sem_p=0.0,
        gamma_inst=0.2,
        gamma_inst_se=0.05,
        gamma_inst_t=4.0,
        gamma_inst_p=0.0,
        beta_wage=3.0,
        beta_wage_se=0.1,
        beta_wage_t=30.0,
        beta_wage_p=0.0,
        log_likelihood=-10000,
        n_cases=1000,
        n_obs=11000,
        converged=True,
        model_variant="M1",
    )

    costs = compute_switching_costs(
        mock_result,
        d_wasserstein_median=0.15,
        d_inst_median=0.8,
        mean_annual_wage=70000,
    )

    assert costs["sc_typical_wage_years"] > 0


# =============================================================================
# Test: Median Distance Computation
# =============================================================================


def test_median_distances(mock_transitions_df, mock_wasserstein_matrix, mock_inst_matrix, mock_census_codes):
    """Test median distance computation from transitions."""
    d_sem_med, d_inst_med = compute_median_distances(
        mock_transitions_df,
        mock_wasserstein_matrix,
        mock_inst_matrix,
        mock_census_codes,
    )

    # Should be positive
    assert d_sem_med >= 0
    assert d_inst_med >= 0


# =============================================================================
# Test: Verdict Logic
# =============================================================================


def test_verdict_validated():
    """Test verdict when all conditions met."""
    verdict = compute_verdict(
        beta_wage=2.5,       # Positive
        gamma_sem=8.0,       # Positive
        gamma_inst=0.2,      # Positive
        sc_typical_wage_years=1.8,  # In range [0.5, 5.0]
    )
    assert verdict == "validated"


def test_verdict_investigate_out_of_range():
    """Test verdict when cost outside range but signs correct."""
    verdict = compute_verdict(
        beta_wage=2.5,       # Positive
        gamma_sem=8.0,       # Positive
        gamma_inst=0.2,      # Positive
        sc_typical_wage_years=0.2,  # Too low
    )
    assert verdict == "investigate"


def test_verdict_misidentified_negative_wage():
    """Test verdict when wage coefficient is negative."""
    verdict = compute_verdict(
        beta_wage=-1.0,      # Wrong sign!
        gamma_sem=8.0,
        gamma_inst=0.2,
        sc_typical_wage_years=1.8,
    )
    assert verdict == "misidentified"


def test_verdict_investigate_negative_gamma():
    """Test verdict when distance coefficient has wrong sign."""
    verdict = compute_verdict(
        beta_wage=2.5,
        gamma_sem=-8.0,      # Wrong sign
        gamma_inst=0.2,
        sc_typical_wage_years=1.8,
    )
    assert verdict == "investigate"


# =============================================================================
# Integration Test: Real OES Data (if available)
# =============================================================================


@pytest.mark.slow
def test_real_oes_data_loads():
    """Test that real OES data loads correctly."""
    try:
        wages_df = load_oes_wages_by_census(year=2023)
        assert len(wages_df) > 0, "OES data should not be empty"
        assert "census_code" in wages_df.columns
        assert "mean_annual_wage" in wages_df.columns
        assert "median_annual_wage" in wages_df.columns
    except FileNotFoundError:
        pytest.skip("OES data not available")


@pytest.mark.slow
def test_real_oes_wage_coverage():
    """Test wage coverage with real data."""
    try:
        from task_space.mobility.census_crosswalk import load_census_onet_crosswalk

        wages_df = load_oes_wages_by_census(year=2023)
        xwalk = load_census_onet_crosswalk()
        census_codes = list(xwalk.census_to_onet.keys())

        coverage = get_wage_coverage(wages_df, census_codes)

        # Should have at least 70% coverage
        assert coverage >= 0.70, f"Coverage too low: {coverage:.1%}"
    except FileNotFoundError:
        pytest.skip("Required data not available")
