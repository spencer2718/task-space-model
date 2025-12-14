import pytest
from pathlib import Path


@pytest.mark.slow
def test_regression_matches_v063():
    """
    Regression output should match v0.6.3 validation run.

    This test verifies that the refactored infrastructure produces
    results consistent with the validated Phase I findings.

    v0.6.3 reference values:
    - Normalized kernel: t ≈ 7.14-7.39, R² ≈ 0.00485-0.00523
    """
    # Skip if experiment infrastructure not available
    pytest.importorskip("task_space.experiments")

    from task_space.experiments import ExperimentConfig, run_experiment

    config = ExperimentConfig(
        name="regression_consistency_test",
        similarity="normalized_kernel",
        run_permutation=False,
        run_cv=False,
        output_dir=Path("outputs/tests"),
    )

    results = run_experiment(config)

    # v0.6.3 reference: t ≈ 7.14-7.39, R² ≈ 0.00485-0.00523
    # Allow tolerance for minor implementation differences
    assert 6.5 < results['regression']['t'] < 8.0, f"t-stat out of range: {results['regression']['t']}"
    assert 0.004 < results['regression']['r2'] < 0.006, f"R² out of range: {results['regression']['r2']}"
