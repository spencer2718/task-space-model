"""
Tests for crosswalk module.

Tests O*NET-SOC to SOC conversion, OES data loading, and wage comovement.
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from task_space.crosswalk import (
    onet_to_soc,
    build_onet_oes_crosswalk,
    load_oes_year,
    load_oes_panel,
    compute_wage_comovement,
    aggregate_occupation_measures,
)


class TestOnetToSoc:
    """Tests for O*NET-SOC to SOC conversion."""

    def test_standard_conversion(self):
        """Standard O*NET code converts correctly."""
        assert onet_to_soc("15-1252.00") == "15-1252"
        assert onet_to_soc("53-3032.00") == "53-3032"

    def test_subdivision_codes(self):
        """O*NET subdivisions (.01, .02) map to same SOC."""
        assert onet_to_soc("15-1252.00") == "15-1252"
        assert onet_to_soc("15-1252.01") == "15-1252"
        assert onet_to_soc("15-1252.02") == "15-1252"

    def test_whitespace_handling(self):
        """Whitespace is stripped."""
        assert onet_to_soc("  15-1252.00  ") == "15-1252"

    def test_short_codes(self):
        """Short codes return as-is."""
        assert onet_to_soc("15-125") == "15-125"


class TestCrosswalk:
    """Tests for crosswalk building."""

    def test_crosswalk_structure(self):
        """Crosswalk has expected structure."""
        onet_codes = ["15-1252.00", "15-1252.01", "53-3032.00"]
        crosswalk = build_onet_oes_crosswalk(onet_codes)

        assert crosswalk.n_onet == 3
        assert crosswalk.n_soc == 2  # 15-1252 and 53-3032
        assert crosswalk.coverage == 1.0  # All match when no OES codes provided

    def test_many_to_one_aggregation(self):
        """Multiple O*NET codes map to one SOC."""
        onet_codes = ["15-1252.00", "15-1252.01", "15-1252.02"]
        crosswalk = build_onet_oes_crosswalk(onet_codes)

        assert crosswalk.n_soc == 1
        assert "15-1252" in crosswalk.aggregation_map
        assert len(crosswalk.aggregation_map["15-1252"]) == 3

    def test_oes_matching(self):
        """OES matching flags work correctly."""
        onet_codes = ["15-1252.00", "53-3032.00", "99-9999.00"]
        oes_codes = ["15-1252", "53-3032"]  # 99-9999 not in OES
        crosswalk = build_onet_oes_crosswalk(onet_codes, oes_codes)

        assert crosswalk.n_matched == 2
        assert crosswalk.coverage == 2 / 3


class TestOesLoading:
    """Tests for OES data loading."""

    @pytest.fixture
    def oes_dir(self):
        """Path to OES data directory."""
        return Path(__file__).parent.parent / "data" / "external" / "oes"

    def test_load_single_year(self, oes_dir):
        """Loading a single year works."""
        if not (oes_dir / "oesm23nat" / "national_M2023_dl.xlsx").exists():
            pytest.skip("OES 2023 data not available")

        df = load_oes_year(2023, oes_dir)

        assert "OCC_CODE" in df.columns
        assert "A_MEAN" in df.columns
        assert len(df) > 700  # Should have many occupations
        # Check codes are detailed (not aggregates)
        assert not df["OCC_CODE"].str.endswith("0000").any()

    def test_load_panel(self, oes_dir):
        """Loading multi-year panel works."""
        # Check which years are available
        available_years = []
        for year in [2019, 2020, 2021, 2022, 2023]:
            yy = str(year)[-2:]
            if (oes_dir / f"oesm{yy}nat" / f"national_M{year}_dl.xlsx").exists():
                available_years.append(year)

        if len(available_years) < 2:
            pytest.skip("Need at least 2 years of OES data")

        panel = load_oes_panel(available_years, oes_dir)

        assert "year" in panel.columns
        assert panel["year"].nunique() == len(available_years)

    def test_missing_year_raises(self, oes_dir):
        """Missing year raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_oes_year(1990, oes_dir)


class TestWageComovement:
    """Tests for wage comovement computation."""

    @pytest.fixture
    def oes_dir(self):
        """Path to OES data directory."""
        return Path(__file__).parent.parent / "data" / "external" / "oes"

    def test_comovement_structure(self, oes_dir):
        """Wage comovement has expected structure."""
        # Check which years are available
        available_years = []
        for year in [2019, 2020, 2021, 2022, 2023]:
            yy = str(year)[-2:]
            if (oes_dir / f"oesm{yy}nat" / f"national_M{year}_dl.xlsx").exists():
                available_years.append(year)

        if len(available_years) < 3:
            pytest.skip("Need at least 3 years of OES data")

        panel = load_oes_panel(available_years, oes_dir)
        comovement = compute_wage_comovement(panel, min_years=2)

        # Check structure
        assert comovement.comovement_matrix.shape[0] == comovement.n_occupations
        assert comovement.comovement_matrix.shape[1] == comovement.n_occupations
        assert len(comovement.occupation_codes) == comovement.n_occupations

        # Matrix should be symmetric (ignoring NaNs)
        diff = comovement.comovement_matrix - comovement.comovement_matrix.T
        assert np.nanmax(np.abs(diff)) < 1e-10

        # Diagonal should be 1 (self-correlation) where not NaN
        diagonal = np.diag(comovement.comovement_matrix)
        valid_diag = diagonal[~np.isnan(diagonal)]
        np.testing.assert_array_almost_equal(
            valid_diag,
            np.ones(len(valid_diag)),
            decimal=10,
        )


class TestAggregation:
    """Tests for occupation measure aggregation."""

    def test_aggregation_averaging(self):
        """Aggregation averages O*NET codes to SOC level."""
        # Create mock data: 3 O*NET codes, 2 SOC codes
        onet_codes = ["15-1252.00", "15-1252.01", "53-3032.00"]
        n_activities = 4

        # Create occupation matrix with known values
        occupation_matrix = np.array([
            [0.4, 0.3, 0.2, 0.1],  # 15-1252.00
            [0.2, 0.3, 0.4, 0.1],  # 15-1252.01
            [0.1, 0.1, 0.1, 0.7],  # 53-3032.00
        ])

        crosswalk = build_onet_oes_crosswalk(onet_codes)
        agg_matrix, soc_codes = aggregate_occupation_measures(
            occupation_matrix, onet_codes, crosswalk
        )

        assert len(soc_codes) == 2
        assert agg_matrix.shape == (2, n_activities)

        # Check normalization (rows sum to 1)
        np.testing.assert_array_almost_equal(
            agg_matrix.sum(axis=1),
            np.ones(2),
        )

    def test_aggregation_preserves_single(self):
        """Single O*NET code passes through unchanged (after renormalization)."""
        onet_codes = ["53-3032.00"]
        occupation_matrix = np.array([[0.25, 0.25, 0.25, 0.25]])

        crosswalk = build_onet_oes_crosswalk(onet_codes)
        agg_matrix, soc_codes = aggregate_occupation_measures(
            occupation_matrix, onet_codes, crosswalk
        )

        assert len(soc_codes) == 1
        np.testing.assert_array_almost_equal(
            agg_matrix[0], occupation_matrix[0]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
