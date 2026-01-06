"""
Unit tests for task_space.mobility.io module.

Tests verify:
- Functions exist and are callable
- Return types are correct (DataFrame, ndarray)
- Default constants are defined
- Error handling for missing files

Note: These are primarily smoke tests. Full integration tests
require actual artifact files.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from task_space.mobility.io import (
    # Constants
    DEFAULT_TRAIN_YEARS,
    DEFAULT_HOLDOUT_YEAR,
    # Functions
    load_transitions,
    get_holdout_transitions,
    get_training_transitions,
    load_wasserstein_census,
    load_institutional_census,
    load_distance_matrix,
    aggregate_institutional_distances,
)


class TestConstants:
    """Tests for module constants."""

    def test_train_years_defined(self):
        """DEFAULT_TRAIN_YEARS should be a list of years."""
        assert isinstance(DEFAULT_TRAIN_YEARS, list)
        assert len(DEFAULT_TRAIN_YEARS) > 0
        assert all(isinstance(y, int) for y in DEFAULT_TRAIN_YEARS)

    def test_train_years_excludes_covid(self):
        """Training years should exclude 2020-2021 (COVID)."""
        assert 2020 not in DEFAULT_TRAIN_YEARS
        assert 2021 not in DEFAULT_TRAIN_YEARS

    def test_train_years_includes_expected(self):
        """Training years should include 2015-2019, 2022-2023."""
        assert 2015 in DEFAULT_TRAIN_YEARS
        assert 2019 in DEFAULT_TRAIN_YEARS
        assert 2022 in DEFAULT_TRAIN_YEARS
        assert 2023 in DEFAULT_TRAIN_YEARS

    def test_holdout_year_defined(self):
        """DEFAULT_HOLDOUT_YEAR should be an integer."""
        assert isinstance(DEFAULT_HOLDOUT_YEAR, int)
        assert DEFAULT_HOLDOUT_YEAR == 2024


class TestFunctionSignatures:
    """Tests that functions exist and have expected signatures."""

    def test_load_transitions_callable(self):
        """load_transitions should be callable."""
        assert callable(load_transitions)

    def test_get_holdout_transitions_callable(self):
        """get_holdout_transitions should be callable."""
        assert callable(get_holdout_transitions)

    def test_get_training_transitions_callable(self):
        """get_training_transitions should be callable."""
        assert callable(get_training_transitions)

    def test_load_wasserstein_census_callable(self):
        """load_wasserstein_census should be callable."""
        assert callable(load_wasserstein_census)

    def test_load_institutional_census_callable(self):
        """load_institutional_census should be callable."""
        assert callable(load_institutional_census)

    def test_load_distance_matrix_callable(self):
        """load_distance_matrix should be callable."""
        assert callable(load_distance_matrix)

    def test_aggregate_institutional_distances_callable(self):
        """aggregate_institutional_distances should be callable."""
        assert callable(aggregate_institutional_distances)


class TestLoadDistanceMatrixValidation:
    """Tests for load_distance_matrix input validation."""

    def test_invalid_kind_raises(self):
        """Invalid distance kind should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown distance kind"):
            load_distance_matrix("invalid_kind")

    def test_valid_kinds_accepted(self):
        """Valid kinds should not raise ValueError on signature check."""
        valid_kinds = [
            "wasserstein",
            "institutional",
            "cosine_onet",
            "cosine_embed",
            "euclidean_dwa",
            "wasserstein_identity",
        ]
        # Just verify no ValueError is raised for valid kinds
        # (FileNotFoundError is expected if files don't exist)
        for kind in valid_kinds:
            try:
                load_distance_matrix(kind)
            except FileNotFoundError:
                pass  # Expected if test run without artifacts
            except ValueError as e:
                pytest.fail(f"Valid kind '{kind}' raised ValueError: {e}")


class TestGetTrainingTransitions:
    """Tests for get_training_transitions function."""

    def test_filters_to_train_years(self):
        """Should filter DataFrame to specified training years."""
        # Create mock data
        df = pd.DataFrame({
            "origin_occ": [1, 2, 3, 4, 5],
            "dest_occ": [2, 3, 4, 5, 6],
            "YEARMONTH": [201501, 202001, 202201, 202401, 201901],
            "year": [2015, 2020, 2022, 2024, 2019],
        })

        result = get_training_transitions(df, train_years=[2015, 2019, 2022])

        assert len(result) == 3
        assert set(result["year"].unique()) == {2015, 2019, 2022}

    def test_uses_default_train_years(self):
        """Should use DEFAULT_TRAIN_YEARS when not specified."""
        df = pd.DataFrame({
            "origin_occ": [1, 2, 3],
            "dest_occ": [2, 3, 4],
            "YEARMONTH": [201501, 202001, 202301],
            "year": [2015, 2020, 2023],
        })

        result = get_training_transitions(df)

        # 2020 should be excluded, 2015 and 2023 included
        assert 2020 not in result["year"].values
        assert 2015 in result["year"].values
        assert 2023 in result["year"].values

    def test_returns_copy(self):
        """Should return a copy, not a view."""
        df = pd.DataFrame({
            "origin_occ": [1],
            "dest_occ": [2],
            "YEARMONTH": [201501],
            "year": [2015],
        })

        result = get_training_transitions(df)
        result["new_col"] = 1

        assert "new_col" not in df.columns


class TestWithMockArtifacts:
    """Tests using mock artifacts (if available)."""

    @pytest.fixture
    def mock_transitions(self, tmp_path):
        """Create a mock transitions parquet file."""
        df = pd.DataFrame({
            "origin_occ": [1000, 1020, 1040, 1060, 1080],
            "dest_occ": [1020, 1040, 1060, 1080, 1100],
            "YEARMONTH": [201501, 201912, 202201, 202401, 202412],
        })
        path = tmp_path / "transitions.parquet"
        df.to_parquet(path)
        return str(path)

    def test_load_transitions_with_mock(self, mock_transitions):
        """load_transitions should work with mock data."""
        df = load_transitions(path=mock_transitions)

        assert isinstance(df, pd.DataFrame)
        assert "year" in df.columns
        assert len(df) == 5

    def test_load_transitions_adds_year(self, mock_transitions):
        """load_transitions should add year column from YEARMONTH."""
        df = load_transitions(path=mock_transitions)

        assert df["year"].iloc[0] == 2015
        assert df["year"].iloc[3] == 2024

    def test_load_transitions_holdout_filter(self, mock_transitions):
        """load_transitions(holdout=True) should filter to 2024+."""
        df = load_transitions(path=mock_transitions, holdout=True)

        assert len(df) == 2  # Only 2024 rows
        assert (df["year"] >= 2024).all()

    def test_get_holdout_transitions_with_mock(self, mock_transitions):
        """get_holdout_transitions should return only holdout year."""
        df = get_holdout_transitions(path=mock_transitions)

        assert len(df) == 2
        assert (df["year"] >= 2024).all()

    def test_get_holdout_custom_year(self, mock_transitions):
        """get_holdout_transitions should respect year parameter."""
        df = get_holdout_transitions(year=2022, path=mock_transitions)

        assert len(df) == 3  # 2022, 2024, 2024
        assert (df["year"] >= 2022).all()


class TestErrorHandling:
    """Tests for error handling."""

    def test_load_transitions_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_transitions(path="/nonexistent/path/to/file.parquet")

    def test_load_wasserstein_missing_file(self, tmp_path, monkeypatch):
        """Should raise FileNotFoundError when artifacts missing."""
        # Monkeypatch CACHE_DIR to empty temp directory
        from task_space.mobility import io
        monkeypatch.setattr(io, "CACHE_DIR", tmp_path)

        with pytest.raises(FileNotFoundError, match="Wasserstein distances not found"):
            load_wasserstein_census()

    def test_load_institutional_missing_file(self, tmp_path, monkeypatch):
        """Should raise FileNotFoundError when artifacts missing."""
        from task_space.mobility import io
        monkeypatch.setattr(io, "CACHE_DIR", tmp_path)

        with pytest.raises(FileNotFoundError, match="Institutional distances not found"):
            load_institutional_census()
