"""
Manual test runner for crosswalk module (no pytest required).
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from task_space.crosswalk import (
    onet_to_soc,
    build_onet_oes_crosswalk,
    load_oes_year,
    load_oes_panel,
    compute_wage_comovement,
    aggregate_occupation_measures,
)

def test_onet_to_soc():
    """Test O*NET-SOC to SOC conversion."""
    print("Testing onet_to_soc...")
    assert onet_to_soc("15-1252.00") == "15-1252"
    assert onet_to_soc("53-3032.00") == "53-3032"
    assert onet_to_soc("15-1252.01") == "15-1252"
    assert onet_to_soc("  15-1252.00  ") == "15-1252"
    print("  PASSED")


def test_crosswalk():
    """Test crosswalk building."""
    print("Testing crosswalk...")
    onet_codes = ["15-1252.00", "15-1252.01", "53-3032.00"]
    crosswalk = build_onet_oes_crosswalk(onet_codes)

    assert crosswalk.n_onet == 3
    assert crosswalk.n_soc == 2
    assert crosswalk.coverage == 1.0

    # Test aggregation map
    onet_codes2 = ["15-1252.00", "15-1252.01", "15-1252.02"]
    crosswalk2 = build_onet_oes_crosswalk(onet_codes2)
    assert crosswalk2.n_soc == 1
    assert len(crosswalk2.aggregation_map["15-1252"]) == 3
    print("  PASSED")


def test_oes_loading():
    """Test OES data loading."""
    print("Testing OES loading...")
    oes_dir = Path(__file__).parent.parent / "data" / "external" / "oes"

    # Try to load 2023
    if not (oes_dir / "oesm23nat" / "national_M2023_dl.xlsx").exists():
        print("  SKIPPED (no 2023 data)")
        return

    df = load_oes_year(2023, oes_dir)
    assert "OCC_CODE" in df.columns
    assert "A_MEAN" in df.columns
    assert len(df) > 700
    assert not df["OCC_CODE"].str.endswith("0000").any()
    print(f"  Loaded {len(df)} occupations for 2023")
    print("  PASSED")


def test_oes_panel():
    """Test OES panel loading."""
    print("Testing OES panel...")
    oes_dir = Path(__file__).parent.parent / "data" / "external" / "oes"

    available_years = []
    for year in [2019, 2020, 2021, 2022, 2023]:
        yy = str(year)[-2:]
        if (oes_dir / f"oesm{yy}nat" / f"national_M{year}_dl.xlsx").exists():
            available_years.append(year)

    if len(available_years) < 2:
        print("  SKIPPED (need at least 2 years)")
        return

    panel = load_oes_panel(available_years, oes_dir)
    assert "year" in panel.columns
    assert panel["year"].nunique() == len(available_years)
    print(f"  Loaded panel for years: {available_years}")
    print("  PASSED")


def test_wage_comovement():
    """Test wage comovement computation."""
    print("Testing wage comovement...")
    oes_dir = Path(__file__).parent.parent / "data" / "external" / "oes"

    available_years = []
    for year in [2019, 2020, 2021, 2022, 2023]:
        yy = str(year)[-2:]
        if (oes_dir / f"oesm{yy}nat" / f"national_M{year}_dl.xlsx").exists():
            available_years.append(year)

    if len(available_years) < 3:
        print("  SKIPPED (need at least 3 years)")
        return

    panel = load_oes_panel(available_years, oes_dir)
    comovement = compute_wage_comovement(panel, min_years=2)

    # Check structure
    assert comovement.comovement_matrix.shape[0] == comovement.n_occupations
    assert len(comovement.occupation_codes) == comovement.n_occupations

    # Symmetric (ignoring NaNs)
    diff = comovement.comovement_matrix - comovement.comovement_matrix.T
    assert np.nanmax(np.abs(diff)) < 1e-10, "Matrix not symmetric"

    # Diagonal should be 1 (self-correlation)
    diagonal = np.diag(comovement.comovement_matrix)
    valid_diag = diagonal[~np.isnan(diagonal)]
    assert np.allclose(valid_diag, np.ones(len(valid_diag))), "Diagonal not all 1s"

    print(f"  {comovement.n_occupations} occupations, {comovement.n_years} years")
    print(f"  Coverage: {comovement.coverage:.1%}")
    print("  PASSED")


def test_aggregation():
    """Test occupation measure aggregation."""
    print("Testing aggregation...")
    onet_codes = ["15-1252.00", "15-1252.01", "53-3032.00"]

    occupation_matrix = np.array([
        [0.4, 0.3, 0.2, 0.1],
        [0.2, 0.3, 0.4, 0.1],
        [0.1, 0.1, 0.1, 0.7],
    ])

    crosswalk = build_onet_oes_crosswalk(onet_codes)
    agg_matrix, soc_codes = aggregate_occupation_measures(
        occupation_matrix, onet_codes, crosswalk
    )

    assert len(soc_codes) == 2
    assert agg_matrix.shape == (2, 4)

    # Check normalization
    assert np.allclose(agg_matrix.sum(axis=1), np.ones(2))
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("Running crosswalk tests...")
    print("=" * 50)

    test_onet_to_soc()
    test_crosswalk()
    test_oes_loading()
    test_oes_panel()
    test_wage_comovement()
    test_aggregation()

    print("=" * 50)
    print("All tests PASSED")
    print("=" * 50)
