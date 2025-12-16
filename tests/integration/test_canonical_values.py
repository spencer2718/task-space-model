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


@pytest.mark.slow
class TestWassersteinCanonicalValues:
    """
    Protect v0.6.8 Wasserstein mobility results.

    These values are reported in paper v0.6.8 Section 5.1.
    If these change, either:
    1. The computation is broken (fix it), or
    2. The paper needs updating (coordinate with lead researcher)
    """

    @pytest.fixture
    def results(self):
        """Load Wasserstein comparison results."""
        path = Path("outputs/experiments/path_a_wasserstein_comparison_v0672.json")
        if not path.exists():
            pytest.skip(f"Wasserstein results not computed: {path}")
        with open(path) as f:
            return json.load(f)

    def test_wasserstein_alpha_coefficient(self, results):
        """α (semantic) with Wasserstein should be ~8.936."""
        alpha = results["comparison"]["wasserstein"]["alpha"]
        assert abs(alpha - 8.936) < 0.01, f"α changed: {alpha}, expected ~8.936"

    def test_wasserstein_beta_coefficient(self, results):
        """β (institutional) with Wasserstein should be ~0.142."""
        beta = results["comparison"]["wasserstein"]["beta"]
        assert abs(beta - 0.142) < 0.01, f"β changed: {beta}, expected ~0.142"

    def test_wasserstein_log_likelihood(self, results):
        """Log-likelihood with Wasserstein should be ~-183,051."""
        ll = results["comparison"]["wasserstein"]["log_likelihood"]
        assert abs(ll - (-183051)) < 10, f"LL changed: {ll}, expected ~-183051"

    def test_wasserstein_alpha_t_statistic(self, results):
        """α t-stat with Wasserstein should be > 200."""
        t_stat = results["comparison"]["wasserstein"]["alpha_t"]
        assert t_stat > 200, f"α t-stat = {t_stat}, expected > 200"

    def test_wasserstein_improvement_over_kernel(self, results):
        """Verify Wasserstein >> Kernel finding persists (Δ LL > 5000)."""
        delta_ll = results["comparison"]["differences"]["delta_log_lik"]
        # Pre-committed threshold was +100; we observed +9,576
        assert delta_ll > 5000, f"Δ LL dropped below safety threshold: {delta_ll}"

    def test_wasserstein_alpha_improvement(self, results):
        """Verify α improvement > 50% over kernel."""
        delta_alpha_pct = results["comparison"]["differences"]["delta_alpha_pct"]
        assert delta_alpha_pct > 50, f"Δα% dropped: {delta_alpha_pct}, expected > 50%"


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

    def test_mobility_wasserstein_exists(self):
        """mobility_wasserstein.json should exist and have required fields."""
        path = CANONICAL_DIR / "mobility_wasserstein.json"
        assert path.exists(), f"Missing: {path}"

        with open(path) as f:
            data = json.load(f)

        # Check required fields
        assert "results" in data, "Missing 'results' field"
        assert "alpha_semantic" in data["results"], "Missing alpha_semantic"
        assert "comparison_vs_kernel" in data, "Missing comparison_vs_kernel"
        assert abs(data["results"]["alpha_semantic"] - 8.936) < 0.01, "alpha mismatch"
