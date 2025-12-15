"""
Regression tests for canonical paper results.

These tests verify that the canonical results files contain expected values.
If these tests fail, either the data has been corrupted or the paper
results need to be updated.

Canonical values (paper v0.6.6):
- CPS Symmetric: α = 2.994, β = 0.215
- CPS Asymmetric: β_up = 0.282, β_down = 0.270, ratio = 1.04
- Wage Comovement: R² = 0.00523
"""

import json
from pathlib import Path

import numpy as np
import pytest


CANONICAL_DIR = Path("outputs/canonical")


@pytest.mark.slow
class TestCPSSymmetricCanonical:
    """Verify CPS symmetric mobility results."""

    @pytest.fixture
    def results(self):
        """Load canonical CPS results."""
        path = CANONICAL_DIR / "mobility_cps.json"
        if not path.exists():
            pytest.skip(f"Canonical file not found: {path}")
        with open(path) as f:
            return json.load(f)

    def test_alpha_coefficient(self, results):
        """α (semantic) coefficient should be ~2.994."""
        alpha = results["alpha_coef"]
        assert abs(alpha - 2.994) < 0.001, f"α = {alpha}, expected ~2.994"

    def test_beta_coefficient(self, results):
        """β (institutional) coefficient should be ~0.215."""
        beta = results["beta_coef"]
        assert abs(beta - 0.215) < 0.001, f"β = {beta}, expected ~0.215"

    def test_alpha_t_statistic(self, results):
        """α should have t > 90 (highly significant)."""
        t_stat = results["alpha_t"]
        assert t_stat > 90, f"α t-stat = {t_stat}, expected > 90"

    def test_beta_t_statistic(self, results):
        """β should have t > 60 (highly significant)."""
        t_stat = results["beta_t"]
        assert t_stat > 60, f"β t-stat = {t_stat}, expected > 60"

    def test_sample_size(self, results):
        """Should have ~89K transitions."""
        n = results["n_transitions"]
        assert 80000 < n < 100000, f"n_transitions = {n}, expected ~89K"


@pytest.mark.slow
class TestCPSAsymmetricCanonical:
    """Verify CPS asymmetric mobility results."""

    @pytest.fixture
    def results(self):
        """Load canonical asymmetric results."""
        path = CANONICAL_DIR / "mobility_asymmetric.json"
        if not path.exists():
            pytest.skip(f"Canonical file not found: {path}")
        with open(path) as f:
            return json.load(f)

    def test_beta_up_coefficient(self, results):
        """β_up (upward barrier) should be ~0.282."""
        beta_up = results["models"]["asymmetric"]["beta_up"]
        assert abs(beta_up - 0.282) < 0.01, f"β_up = {beta_up}, expected ~0.282"

    def test_beta_down_coefficient(self, results):
        """β_down (downward barrier) should be ~0.270."""
        beta_down = results["models"]["asymmetric"]["beta_down"]
        assert abs(beta_down - 0.270) < 0.01, f"β_down = {beta_down}, expected ~0.270"

    def test_asymmetry_ratio(self, results):
        """Asymmetry ratio should be ~1.04."""
        ratio = results["models"]["asymmetric"]["asymmetry_ratio"]
        assert abs(ratio - 1.04) < 0.05, f"ratio = {ratio}, expected ~1.04"

    def test_lr_test_pvalue(self, results):
        """LR test p-value should be ~0.0375."""
        pval = results["hypothesis_tests"]["lr_test_pvalue"]
        assert 0.01 < pval < 0.10, f"LR p-value = {pval}, expected ~0.0375"

    def test_verdict(self, results):
        """Verdict should indicate symmetric barriers."""
        verdict = results["verdict"].lower()
        assert "symmetric" in verdict or "null" in verdict


@pytest.mark.slow
class TestWageComovementCanonical:
    """Verify wage comovement results."""

    @pytest.fixture
    def results(self):
        """Load canonical wage comovement results."""
        path = CANONICAL_DIR / "wage_comovement.json"
        if not path.exists():
            pytest.skip(f"Canonical file not found: {path}")
        with open(path) as f:
            return json.load(f)

    def test_r_squared(self, results):
        """R² should be ~0.00523."""
        r2 = results["regression"]["r2"]
        assert abs(r2 - 0.00523) < 0.001, f"R² = {r2}, expected ~0.00523"

    def test_permutation_significant(self, results):
        """Permutation test should be highly significant."""
        pval = results["permutation"]["p_value"]
        assert pval < 0.01, f"Permutation p = {pval}, expected < 0.01"


@pytest.mark.slow
class TestAutomationCanonical:
    """Verify automation prediction results."""

    @pytest.fixture
    def results(self):
        """Load canonical automation results."""
        path = CANONICAL_DIR / "automation_v0653.json"
        if not path.exists():
            pytest.skip(f"Canonical file not found: {path}")
        with open(path) as f:
            return json.load(f)

    def test_rti_only_r_squared(self, results):
        """RTI-only R² should be ~9.8%."""
        models = results["models"]
        rti_model = models.get("model_1_rti_only")
        if rti_model is None:
            pytest.skip("RTI-only model not found in results")
        r2 = rti_model["r2"]
        assert abs(r2 - 0.098) < 0.01, f"RTI R² = {r2}, expected ~0.098"

    def test_rti_semantic_r_squared(self, results):
        """RTI + Semantic R² should be ~12.0%."""
        models = results["models"]
        # Look for model with RTI + semantic but not AIOE
        combined = models.get("model_2_rti_semantic")
        if combined is None:
            pytest.skip("RTI+Semantic model not found in results")
        r2 = combined["r2"]
        assert abs(r2 - 0.12) < 0.02, f"RTI+Semantic R² = {r2}, expected ~0.12"


class TestCanonicalFilesExist:
    """Basic existence checks (fast, run always)."""

    def test_canonical_directory_exists(self):
        """Canonical directory should exist."""
        assert CANONICAL_DIR.exists(), f"Missing: {CANONICAL_DIR}"

    def test_provenance_file_exists(self):
        """PROVENANCE.md should exist."""
        path = CANONICAL_DIR / "PROVENANCE.md"
        assert path.exists(), f"Missing: {path}"

    def test_mobility_cps_exists(self):
        """mobility_cps.json should exist."""
        path = CANONICAL_DIR / "mobility_cps.json"
        assert path.exists(), f"Missing: {path}"

    def test_mobility_asymmetric_exists(self):
        """mobility_asymmetric.json should exist."""
        path = CANONICAL_DIR / "mobility_asymmetric.json"
        assert path.exists(), f"Missing: {path}"

    def test_wage_comovement_exists(self):
        """wage_comovement.json should exist."""
        path = CANONICAL_DIR / "wage_comovement.json"
        assert path.exists(), f"Missing: {path}"

    def test_automation_exists(self):
        """automation_v0653.json should exist."""
        path = CANONICAL_DIR / "automation_v0653.json"
        assert path.exists(), f"Missing: {path}"
