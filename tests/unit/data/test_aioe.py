"""
Unit tests for AIOE data loader.
"""

import pytest
from pathlib import Path

from task_space.data.aioe import load_aioe, get_aioe_by_soc, AIOEData


class TestLoadAIOE:
    """Tests for load_aioe function."""

    def test_load_returns_aioe_data(self):
        """load_aioe returns AIOEData dataclass."""
        data = load_aioe()
        assert isinstance(data, AIOEData)

    def test_load_has_expected_occupations(self):
        """AIOE has approximately 774 occupations."""
        data = load_aioe()
        assert data.n_occupations == 774

    def test_scores_dataframe_has_expected_columns(self):
        """Scores DataFrame has required columns."""
        data = load_aioe(include_lm=True)
        required_cols = ["soc_code", "occupation_title", "aioe"]
        for col in required_cols:
            assert col in data.scores.columns

    def test_lm_aioe_included_when_requested(self):
        """LM AIOE column present when include_lm=True."""
        data = load_aioe(include_lm=True)
        assert "lm_aioe" in data.scores.columns
        assert data.has_lm_aioe is True

    def test_lm_aioe_excluded_when_not_requested(self):
        """LM AIOE column absent when include_lm=False."""
        data = load_aioe(include_lm=False)
        assert "lm_aioe" not in data.scores.columns
        assert data.has_lm_aioe is False

    def test_soc_code_format(self):
        """SOC codes have expected format XX-XXXX."""
        data = load_aioe()
        # Check first few codes match pattern
        for code in data.scores["soc_code"].head(10):
            assert len(code) == 7, f"SOC code {code} has wrong length"
            assert code[2] == "-", f"SOC code {code} missing dash"

    def test_aioe_scores_reasonable_range(self):
        """AIOE scores are approximately standardized (mean~0, std~1)."""
        data = load_aioe()
        mean = data.scores["aioe"].mean()
        std = data.scores["aioe"].std()

        # Should be close to standard normal
        assert abs(mean) < 0.1, f"AIOE mean {mean} not near 0"
        assert 0.9 < std < 1.1, f"AIOE std {std} not near 1"

    def test_no_missing_aioe_scores(self):
        """All occupations have AIOE scores (no NaN)."""
        data = load_aioe()
        assert data.scores["aioe"].notna().all()


class TestGetAIOEBySOC:
    """Tests for get_aioe_by_soc function."""

    def test_returns_float_for_valid_code(self):
        """Returns float score for valid SOC code."""
        score = get_aioe_by_soc("11-1011")  # Chief Executives
        assert isinstance(score, float)

    def test_returns_none_for_invalid_code(self):
        """Returns None for non-existent SOC code."""
        score = get_aioe_by_soc("99-9999")
        assert score is None

    def test_lm_aioe_when_requested(self):
        """Returns LM AIOE when use_lm=True."""
        standard = get_aioe_by_soc("11-1011", use_lm=False)
        lm = get_aioe_by_soc("11-1011", use_lm=True)

        # Both should exist and be different (but close)
        assert standard is not None
        assert lm is not None
        # They correlate ~0.98 but are not identical
        assert standard != lm


class TestAIOEDataIntegrity:
    """Integration tests for AIOE data quality."""

    def test_chief_executives_high_aioe(self):
        """Chief Executives (11-1011) have above-average AIOE."""
        data = load_aioe()
        ceo = data.scores[data.scores["soc_code"] == "11-1011"]
        assert len(ceo) == 1
        # CEOs should have positive AIOE (above mean)
        assert ceo["aioe"].iloc[0] > 0

    def test_construction_laborers_low_aioe(self):
        """Construction Laborers (47-2061) have below-average AIOE."""
        data = load_aioe()
        laborers = data.scores[data.scores["soc_code"] == "47-2061"]
        if len(laborers) > 0:
            # Should have negative AIOE (below mean)
            assert laborers["aioe"].iloc[0] < 0

    def test_scores_not_all_identical(self):
        """AIOE scores vary across occupations."""
        data = load_aioe()
        unique_scores = data.scores["aioe"].nunique()
        # Should have many unique values
        assert unique_scores > 100
