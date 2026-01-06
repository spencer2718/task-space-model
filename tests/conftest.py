"""
Shared test fixtures for task-space-model tests.

This module centralizes commonly used mock data fixtures to avoid duplication
across test files. All fixtures use consistent shapes and conventions.

Fixture naming conventions:
- mock_*_matrix: NumPy array distance matrices (symmetric, zero diagonal)
- mock_*_df: Pandas DataFrames with test data
- mock_*_codes: List of occupation/census codes
"""

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Distance Matrix Fixtures
# =============================================================================


@pytest.fixture
def mock_wasserstein_matrix():
    """
    Mock 10x10 Wasserstein distance matrix.

    Symmetric with zero diagonal, values in [0, 0.5] range.
    Uses fixed seed for reproducibility.
    """
    np.random.seed(42)
    n = 10
    d = np.random.rand(n, n) * 0.5
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0)
    return d


@pytest.fixture
def mock_inst_matrix():
    """
    Mock 10x10 institutional distance matrix.

    Symmetric with zero diagonal, values in [0, 2] range.
    Uses fixed seed for reproducibility.
    """
    np.random.seed(43)
    n = 10
    d = np.random.rand(n, n) * 2
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0)
    return d


@pytest.fixture
def mock_distance_matrix_3x3():
    """
    Small 3x3 distance matrix for simple tests.

    Origin 0 is closer to destination 1 than destination 2.
    """
    return np.array([
        [0.0, 0.1, 0.5],
        [0.1, 0.0, 0.3],
        [0.5, 0.3, 0.0],
    ])


@pytest.fixture
def mock_distance_matrix_20x20():
    """
    Larger 20x20 distance matrix for integration tests.
    """
    rng = np.random.default_rng(42)
    n = 20
    d = rng.random((n, n))
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0)
    return d * 0.5


# =============================================================================
# Census/Occupation Code Fixtures
# =============================================================================


@pytest.fixture
def mock_census_codes():
    """Census codes matching the mock 10x10 matrices."""
    return [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


@pytest.fixture
def mock_census_codes_3():
    """Census codes matching the 3x3 matrix."""
    return [100, 200, 300]


@pytest.fixture
def mock_census_codes_20():
    """Census codes for 20x20 matrix."""
    return list(range(1000, 1020))


# =============================================================================
# DataFrame Fixtures
# =============================================================================


@pytest.fixture
def mock_transitions_df():
    """
    Mock CPS transition data with 100 transitions.

    Uses 5 occupations (10, 20, 30, 40, 50) as origins/destinations.
    """
    return pd.DataFrame({
        "CPSIDP": list(range(100)),
        "YEARMONTH": [202401] * 50 + [202301] * 50,
        "origin_occ": [10, 20, 30, 40, 50] * 20,
        "dest_occ": [20, 30, 40, 50, 60] * 20,
        "AGE": [35] * 100,
        "SEX": [1] * 100,
    })


@pytest.fixture
def mock_aioe_df():
    """Mock AIOE scores at Census level for 10 occupations."""
    return pd.DataFrame({
        "census_code": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "aioe_score": [0.3, 0.5, 0.7, 0.9, 0.95, 0.4, 0.6, 0.8, 0.85, 0.92],
    })


@pytest.fixture
def mock_aioe_census_df():
    """Mock AIOE scores at Census level with n_soc_codes."""
    return pd.DataFrame({
        "census_code": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "aioe_score": [0.5, 1.2, 0.8, 1.5, 0.3, 0.9, 1.1, 0.6, 1.3, 0.4],
        "n_soc_codes": [1, 2, 1, 1, 1, 2, 1, 1, 1, 1],
    })


@pytest.fixture
def mock_employment_df():
    """Mock employment data at Census level for 10 occupations."""
    return pd.DataFrame({
        "census_code": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "tot_emp": [100000, 80000, 60000, 40000, 30000, 90000, 70000, 50000, 45000, 35000],
    })


@pytest.fixture
def mock_wages_df():
    """Mock OES wages at Census level for 10 occupations."""
    return pd.DataFrame({
        "census_code": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "mean_annual_wage": [50000, 60000, 70000, 80000, 55000, 65000, 75000, 85000, 58000, 68000],
        "median_annual_wage": [48000, 58000, 68000, 78000, 53000, 63000, 73000, 83000, 56000, 66000],
        "soc_code": ["11-1011", "11-1021", "11-2011", "11-2021", "11-2022",
                     "11-2033", "11-3012", "11-3013", "11-3021", "11-3031"],
        "n_soc_codes": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    })


@pytest.fixture
def mock_holdout_df():
    """
    Mock holdout data for validation tests.

    Transitions from high-AIOE occupations (40, 50, 90, 100) to various destinations.
    """
    return pd.DataFrame({
        "origin_occ": [40, 40, 50, 50, 90, 90, 100, 100] * 10,
        "dest_occ": [10, 20, 30, 60, 70, 80, 10, 20] * 10,
    })


# =============================================================================
# Probability Matrix Fixtures
# =============================================================================


@pytest.fixture
def mock_probability_matrix():
    """
    Mock 3x3 probability matrix.

    Rows sum to 1, diagonal is 0 (no self-transitions).
    """
    return np.array([
        [0.0, 0.7, 0.3],  # From 100: prefer 200
        [0.4, 0.0, 0.6],  # From 200: prefer 300
        [0.5, 0.5, 0.0],  # From 300: equal
    ])


@pytest.fixture
def mock_probability_matrix_10x10(mock_wasserstein_matrix, mock_inst_matrix):
    """
    Generate a 10x10 probability matrix from mock distance matrices.

    Uses softmax with gamma_sem=5.0 and gamma_inst=0.5.
    """
    d_sem = mock_wasserstein_matrix
    d_inst = mock_inst_matrix
    n = d_sem.shape[0]

    gamma_sem = 5.0
    gamma_inst = 0.5

    # Compute utilities
    utility = -gamma_sem * d_sem - gamma_inst * d_inst

    # Zero diagonal before softmax
    np.fill_diagonal(utility, -np.inf)

    # Softmax per row
    exp_util = np.exp(utility - utility.max(axis=1, keepdims=True))
    P = exp_util / exp_util.sum(axis=1, keepdims=True)

    # Ensure diagonal is exactly 0
    np.fill_diagonal(P, 0)

    # Renormalize
    P = P / P.sum(axis=1, keepdims=True)

    return P


# =============================================================================
# Census Crosswalk Fixtures
# =============================================================================


@pytest.fixture
def mock_census_crosswalk():
    """
    Mock O*NET to Census crosswalk DataFrame.

    Maps 3 O*NET codes to 2 Census codes.
    """
    return pd.DataFrame({
        "onetsoc_code": ["11-1011.00", "11-1011.00", "11-1021.00"],
        "census_code": [10, 10, 20],
    })
