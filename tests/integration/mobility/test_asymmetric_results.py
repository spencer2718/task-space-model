"""
Integration tests for asymmetric institutional barriers (v0.6.6.0).

Tests:
- Model runs without errors
- LR test is computed correctly
- Asymmetry ratio is finite
- Output file is produced correctly
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from task_space.mobility import (
    build_asymmetric_institutional_distance,
    verify_asymmetric_decomposition,
    build_asymmetric_choice_dataset,
    fit_asymmetric_conditional_logit,
    AsymmetricChoiceModelResult,
)


@pytest.mark.slow
class TestAsymmetricModelRuns:
    """Test that asymmetric model runs without errors."""

    @pytest.fixture
    def asymmetric_distances(self):
        """Build asymmetric institutional distances."""
        return build_asymmetric_institutional_distance()

    def test_asymmetric_distances_build(self, asymmetric_distances):
        """Asymmetric distances should build successfully."""
        assert asymmetric_distances.n_occupations > 0
        assert asymmetric_distances.d_up.shape[0] == asymmetric_distances.n_occupations
        assert asymmetric_distances.d_down.shape[0] == asymmetric_distances.n_occupations

    def test_decomposition_passes(self, asymmetric_distances):
        """Decomposition verification should pass."""
        verification = verify_asymmetric_decomposition(asymmetric_distances)
        assert verification["all_properties_pass"]


@pytest.mark.slow
class TestAsymmetricChoiceDataset:
    """Test asymmetric choice dataset construction."""

    @pytest.fixture
    def mock_transitions(self):
        """Create mock transition data for testing."""
        # Create synthetic transitions for testing
        np.random.seed(42)
        n_transitions = 100
        n_occs = 50

        # Generate random transitions
        origins = np.random.randint(0, n_occs, n_transitions)
        dests = np.random.randint(0, n_occs, n_transitions)

        # Ensure origin != destination
        dests = np.where(dests == origins, (dests + 1) % n_occs, dests)

        return pd.DataFrame({
            "origin_occ": origins,
            "dest_occ": dests,
        })

    @pytest.fixture
    def mock_distances(self):
        """Create mock distance matrices for testing."""
        np.random.seed(42)
        n_occs = 50

        # Generate symmetric semantic distance
        d_sem = np.random.uniform(0.5, 1.0, (n_occs, n_occs))
        d_sem = (d_sem + d_sem.T) / 2
        np.fill_diagonal(d_sem, 0)

        # Generate asymmetric institutional distances
        zones = np.random.randint(1, 6, n_occs)
        zone_diff = zones[None, :] - zones[:, None]
        d_up = np.maximum(0, zone_diff).astype(float)
        d_down = np.maximum(0, -zone_diff).astype(float)

        return d_sem, d_up, d_down, list(range(n_occs))

    def test_asymmetric_choice_dataset_builds(self, mock_transitions, mock_distances):
        """Asymmetric choice dataset should build without errors."""
        d_sem, d_up, d_down, occ_codes = mock_distances

        choice_df = build_asymmetric_choice_dataset(
            mock_transitions,
            d_sem,
            d_up,
            d_down,
            occ_codes,
            n_alternatives=5,
            random_seed=42,
        )

        assert len(choice_df) > 0
        assert "neg_d_sem" in choice_df.columns
        assert "neg_d_inst_up" in choice_df.columns
        assert "neg_d_inst_down" in choice_df.columns
        assert "chosen" in choice_df.columns
        assert "transition_id" in choice_df.columns

    def test_asymmetric_clogit_fits(self, mock_transitions, mock_distances):
        """Asymmetric conditional logit should fit without errors."""
        d_sem, d_up, d_down, occ_codes = mock_distances

        choice_df = build_asymmetric_choice_dataset(
            mock_transitions,
            d_sem,
            d_up,
            d_down,
            occ_codes,
            n_alternatives=5,
            random_seed=42,
        )

        # Skip if too few transitions made it through
        if choice_df["transition_id"].nunique() < 10:
            pytest.skip("Too few transitions for model fitting")

        result = fit_asymmetric_conditional_logit(choice_df)

        # Basic checks
        assert isinstance(result, AsymmetricChoiceModelResult)
        assert np.isfinite(result.alpha)
        assert np.isfinite(result.beta_up)
        assert np.isfinite(result.beta_down)
        assert np.isfinite(result.log_likelihood)

    def test_lr_test_computed(self, mock_transitions, mock_distances):
        """LR test should be computed with finite values."""
        d_sem, d_up, d_down, occ_codes = mock_distances

        choice_df = build_asymmetric_choice_dataset(
            mock_transitions,
            d_sem,
            d_up,
            d_down,
            occ_codes,
            n_alternatives=5,
            random_seed=42,
        )

        if choice_df["transition_id"].nunique() < 10:
            pytest.skip("Too few transitions for model fitting")

        result = fit_asymmetric_conditional_logit(choice_df)

        assert np.isfinite(result.lr_test_statistic)
        assert np.isfinite(result.lr_test_pvalue)
        assert 0 <= result.lr_test_pvalue <= 1
        assert result.lr_test_statistic >= 0  # LR stat should be non-negative

    def test_asymmetry_ratio_finite(self, mock_transitions, mock_distances):
        """Asymmetry ratio should be finite and positive."""
        d_sem, d_up, d_down, occ_codes = mock_distances

        choice_df = build_asymmetric_choice_dataset(
            mock_transitions,
            d_sem,
            d_up,
            d_down,
            occ_codes,
            n_alternatives=5,
            random_seed=42,
        )

        if choice_df["transition_id"].nunique() < 10:
            pytest.skip("Too few transitions for model fitting")

        result = fit_asymmetric_conditional_logit(choice_df)

        # Asymmetry ratio should be positive (or inf if beta_down ≈ 0)
        assert result.asymmetry_ratio > 0 or result.asymmetry_ratio == float('inf')


@pytest.mark.slow
class TestAsymmetricOutputFile:
    """Test asymmetric results output file format."""

    def test_output_json_schema(self):
        """If output file exists, verify it has expected schema."""
        output_path = Path("outputs/experiments/asymmetric_mobility_v0660.json")

        if not output_path.exists():
            pytest.skip("Output file not yet generated - run experiment first")

        with open(output_path) as f:
            data = json.load(f)

        # Check required top-level keys
        required_keys = [
            "version",
            "hypothesis",
            "sample",
            "models",
            "hypothesis_tests",
            "interpretation",
        ]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"

        # Check model structure
        assert "symmetric" in data["models"]
        assert "asymmetric" in data["models"]

        # Check asymmetric model has expected fields
        asym = data["models"]["asymmetric"]
        assert "alpha" in asym
        assert "beta_up" in asym
        assert "beta_down" in asym
        assert "lr_test_statistic" in asym
        assert "lr_test_pvalue" in asym
        assert "asymmetry_ratio" in asym

        # Check hypothesis tests
        tests = data["hypothesis_tests"]
        assert "lr_test_statistic" in tests
        assert "lr_test_pvalue" in tests
        assert "asymmetry_ratio" in tests

        # Check interpretation
        interp = data["interpretation"]
        assert "beta_up_significant" in interp
        assert "beta_down_significant" in interp
        assert "asymmetry_significant" in interp

    def test_output_values_reasonable(self):
        """If output file exists, verify values are in reasonable ranges."""
        output_path = Path("outputs/experiments/asymmetric_mobility_v0660.json")

        if not output_path.exists():
            pytest.skip("Output file not yet generated - run experiment first")

        with open(output_path) as f:
            data = json.load(f)

        asym = data["models"]["asymmetric"]

        # Coefficients should be finite
        assert np.isfinite(asym["alpha"])
        assert np.isfinite(asym["beta_up"])
        assert np.isfinite(asym["beta_down"])

        # Alpha should be similar to symmetric model (~3)
        assert 1 < asym["alpha"] < 10, f"Alpha {asym['alpha']} seems unusual"

        # Beta_up should be positive if credentials matter
        # (but we don't require it - that's what we're testing)

        # LR test should be non-negative
        assert data["hypothesis_tests"]["lr_test_statistic"] >= 0

        # P-value should be in [0, 1]
        assert 0 <= data["hypothesis_tests"]["lr_test_pvalue"] <= 1
