"""
Regression tests for canonical paper results (v0.7.7+).

Verifies that experiment output JSONs contain expected values.
If these fail, either the data has changed or paper results need updating.
"""

import json
from pathlib import Path

import pytest


EXPERIMENTS_DIR = Path("outputs/experiments")


class TestDistanceComparison:
    """Verify 2x2 distance comparison results (Table 2)."""

    @pytest.fixture
    def results(self):
        path = EXPERIMENTS_DIR / "distance_head_to_head_v0732.json"
        if not path.exists():
            pytest.skip(f"Not found: {path}")
        with open(path) as f:
            return json.load(f)

    def test_centroid_pseudo_r2(self, results):
        """Centroid pseudo-R² should be ~14.08%."""
        r2 = results["models"]["cosine_embed"]["pseudo_r2"]
        assert abs(r2 - 0.1408) < 0.005, f"Centroid R²={r2:.4f}, expected ~0.1408"

    def test_centroid_alpha(self, results):
        """Centroid α should be ~7.40."""
        alpha = results["models"]["cosine_embed"]["alpha"]
        assert abs(alpha - 7.404) < 0.05, f"α={alpha:.3f}, expected ~7.404"

    def test_onet_cosine_pseudo_r2(self, results):
        """O*NET cosine pseudo-R² should be ~8.05%."""
        r2 = results["models"]["cosine_onet"]["pseudo_r2"]
        assert abs(r2 - 0.0805) < 0.005, f"O*NET cosine R²={r2:.4f}, expected ~0.0805"

    def test_sample_size(self, results):
        assert results["sample_n"] == 89329


class TestDiagonalAudit:
    """Verify diagonal correction results (v0.7.7.0)."""

    @pytest.fixture
    def results(self):
        path = EXPERIMENTS_DIR / "diagonal_audit_v0770.json"
        if not path.exists():
            pytest.skip(f"Not found: {path}")
        with open(path) as f:
            return json.load(f)

    def test_corrected_pseudo_r2(self, results):
        """Corrected Wasserstein should be ~13.76%."""
        r2 = results["models"]["wasserstein_corrected"]["pseudo_r2"]
        assert abs(r2 - 0.1376) < 0.005, f"Corrected R²={r2:.4f}, expected ~0.1376"

    def test_correction_impact(self, results):
        """Correction should reduce R² by ~0.79pp."""
        impact = results["comparison"]["correction_impact_pp"]
        assert abs(impact - (-0.79)) < 0.1, f"Impact={impact:.2f}pp, expected ~-0.79"

    def test_nonzero_diagonal_count(self, results):
        assert results["diagonal_profiles"]["wasserstein"]["nonzero_count"] == 170


class TestCOVIDStability:
    """Verify pre/post COVID centroid results (v0.7.7.3)."""

    @pytest.fixture
    def results(self):
        path = EXPERIMENTS_DIR / "covid_centroid_v0773.json"
        if not path.exists():
            pytest.skip(f"Not found: {path}")
        with open(path) as f:
            return json.load(f)

    def test_pre_covid_alpha(self, results):
        alpha = results["pre_covid"]["alpha"]
        assert abs(alpha - 7.394) < 0.05, f"Pre α={alpha:.3f}, expected ~7.394"

    def test_alpha_stability(self, results):
        """α should change by less than 1%."""
        pct = abs(results["alpha_pct_change"])
        assert pct < 1.0, f"α changed by {pct:.2f}%, expected <1%"

    def test_no_structural_break(self, results):
        """Structural break test p-value should be > 0.05."""
        p = results["structural_break_test"]["p_value"]
        assert p > 0.05, f"Break test p={p:.4f}, expected >0.05"
