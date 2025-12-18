"""
Battery runner for orchestrating retrospective tests.

Executes multiple tests and produces the interpretation matrix showing
where geometry-based exposure adds signal vs. benchmark measures.

Output format matches project experiment JSON standard.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from .exposure import ExposureMeasure, ExposureResult
from .evaluator import BatteryEvaluator, EvaluatorResult, Verdict


@dataclass
class InterpretationCell:
    """
    Single cell in the interpretation matrix.

    Attributes:
        test_name: Name of the test (row)
        outcome_name: Name of the outcome (column)
        verdict: +, -, or 0
        beta_residual: Coefficient on orthogonalized continuous exposure
        beta_residual_se: Standard error
        beta_residual_p: p-value
        delta_r2: Incremental R² from adding geometry
    """
    test_name: str
    outcome_name: str
    verdict: str
    beta_residual: float
    beta_residual_se: float
    beta_residual_p: float
    delta_r2: float


@dataclass
class InterpretationMatrix:
    """
    Interpretation matrix for the retrospective battery.

    The matrix has:
    - Rows: Tests (A, B, C from Appendix A)
    - Columns: Outcomes (ΔL employment, ΔW wages, etc.)
    - Cells: Verdict (+, -, 0) based on geometry's residual contribution

    Interpretation:
    - "+" means geometry adds significant positive predictive signal
      beyond the benchmark measure (β₃ > 0, p < α)
    - "-" means geometry contradicts the benchmark (β₃ < 0, p < α)
    - "0" means no significant residual effect

    A robust framework should show mostly "+" or "0", few "-".
    """
    cells: List[InterpretationCell] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)

    def add(self, result: EvaluatorResult):
        """Add an evaluator result to the matrix."""
        cell = InterpretationCell(
            test_name=result.test_name,
            outcome_name=result.outcome_name,
            verdict=result.verdict.value,
            beta_residual=result.beta_residual.beta,
            beta_residual_se=result.beta_residual.se,
            beta_residual_p=result.beta_residual.p,
            delta_r2=result.model_comparison.delta_r2,
        )
        self.cells.append(cell)

        if result.test_name not in self.tests:
            self.tests.append(result.test_name)
        if result.outcome_name not in self.outcomes:
            self.outcomes.append(result.outcome_name)

    def get_verdict(self, test_name: str, outcome_name: str) -> Optional[str]:
        """Get verdict for a specific test × outcome combination."""
        for cell in self.cells:
            if cell.test_name == test_name and cell.outcome_name == outcome_name:
                return cell.verdict
        return None

    def summary(self) -> Dict[str, int]:
        """
        Summarize verdict counts.

        Returns:
            Dict with counts: {"positive": n, "negative": n, "null": n}
        """
        counts = {"positive": 0, "negative": 0, "null": 0}
        for cell in self.cells:
            if cell.verdict == "+":
                counts["positive"] += 1
            elif cell.verdict == "-":
                counts["negative"] += 1
            else:
                counts["null"] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "tests": self.tests,
            "outcomes": self.outcomes,
            "cells": [
                {
                    "test": c.test_name,
                    "outcome": c.outcome_name,
                    "verdict": c.verdict,
                    "beta_residual": c.beta_residual,
                    "beta_residual_se": c.beta_residual_se,
                    "beta_residual_p": c.beta_residual_p,
                    "delta_r2": c.delta_r2,
                }
                for c in self.cells
            ],
            "summary": self.summary(),
        }

    def to_markdown_table(self) -> str:
        """Render as markdown table for display."""
        if not self.cells:
            return "No results yet."

        # Build matrix
        header = "| Test | " + " | ".join(self.outcomes) + " |"
        separator = "|------|" + "|".join(["------"] * len(self.outcomes)) + "|"

        rows = []
        for test in self.tests:
            row_cells = [test]
            for outcome in self.outcomes:
                verdict = self.get_verdict(test, outcome)
                row_cells.append(verdict if verdict else "—")
            rows.append("| " + " | ".join(row_cells) + " |")

        return "\n".join([header, separator] + rows)


@dataclass
class BatteryResult:
    """
    Complete result from running the battery.

    Attributes:
        version: Codebase version
        timestamp: When the battery was run
        interpretation_matrix: The main result
        test_results: Detailed results for each test
        metadata: Additional information about the run
    """
    version: str
    timestamp: str
    interpretation_matrix: InterpretationMatrix
    test_results: List[EvaluatorResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "interpretation_matrix": self.interpretation_matrix.to_dict(),
            "test_results": [r.to_dict() for r in self.test_results],
            "metadata": self.metadata,
        }

    def save(self, path: Path):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class TestConfig:
    """
    Configuration for a single battery test.

    Attributes:
        name: Test identifier (e.g., "Test_A_Computer")
        exposure_measure: ExposureMeasure instance
        outcome_data: Dict mapping outcome_name -> (values, unit_ids)
        cluster_var: Optional variable to cluster on
        controls: Optional control variables
    """
    name: str
    exposure_measure: ExposureMeasure
    outcome_data: Dict[str, tuple]  # outcome_name -> (values, unit_ids)
    cluster_var: Optional[str] = None
    controls: Optional[Dict[str, Any]] = None


class BatteryRunner:
    """
    Orchestrates execution of the retrospective battery.

    The battery tests whether our geometry-based exposure measures
    add predictive signal beyond established benchmark measures.

    Example usage:
        >>> runner = BatteryRunner(version="0.7.3.0")
        >>> runner.add_test(test_config_a)
        >>> runner.add_test(test_config_b)
        >>> runner.add_test(test_config_c)
        >>> result = runner.run()
        >>> result.save(Path("outputs/experiments/battery_v073.json"))

    The runner:
    1. Iterates through configured tests
    2. For each test, evaluates each outcome
    3. Builds the interpretation matrix
    4. Produces JSON output in project format
    """

    def __init__(
        self,
        version: str,
        alpha: float = 0.05,
    ):
        """
        Initialize the runner.

        Args:
            version: Codebase version string (e.g., "0.7.3.0")
            alpha: Significance level for verdicts
        """
        self.version = version
        self.alpha = alpha
        self.tests: List[TestConfig] = []

    def add_test(self, config: TestConfig):
        """Add a test configuration to the battery."""
        self.tests.append(config)

    def run(self) -> BatteryResult:
        """
        Execute all configured tests and build interpretation matrix.

        Returns:
            BatteryResult with interpretation matrix and detailed results
        """
        matrix = InterpretationMatrix()
        all_results: List[EvaluatorResult] = []

        for test in self.tests:
            # Get exposure data
            exposure_result = test.exposure_measure.compute_all()

            # Run evaluation for each outcome
            for outcome_name, (outcome_values, unit_ids) in test.outcome_data.items():
                result = self._evaluate_single(
                    test_name=test.name,
                    outcome_name=outcome_name,
                    outcome_values=outcome_values,
                    unit_ids=unit_ids,
                    exposure_result=exposure_result,
                    cluster_var=test.cluster_var,
                    controls=test.controls,
                )

                if result is not None:
                    matrix.add(result)
                    all_results.append(result)

        return BatteryResult(
            version=self.version,
            timestamp=datetime.now().isoformat(),
            interpretation_matrix=matrix,
            test_results=all_results,
            metadata={
                "n_tests": len(self.tests),
                "n_outcomes": len(matrix.outcomes),
                "alpha": self.alpha,
            },
        )

    def _evaluate_single(
        self,
        test_name: str,
        outcome_name: str,
        outcome_values: Any,
        unit_ids: List[str],
        exposure_result: ExposureResult,
        cluster_var: Optional[str],
        controls: Optional[Dict[str, Any]],
    ) -> Optional[EvaluatorResult]:
        """
        Evaluate a single test × outcome combination.

        Aligns exposure and outcome data by unit_id, then runs evaluation.
        """
        import numpy as np

        # Align data by unit_id
        aligned_outcome = []
        aligned_discrete = []
        aligned_continuous = []
        aligned_cluster = []

        for i, uid in enumerate(unit_ids):
            if uid in exposure_result.discrete and uid in exposure_result.continuous:
                aligned_outcome.append(outcome_values[i])
                aligned_discrete.append(exposure_result.discrete[uid])
                aligned_continuous.append(exposure_result.continuous[uid])
                if cluster_var is not None and controls is not None:
                    aligned_cluster.append(controls.get(cluster_var, {}).get(uid, i))
                else:
                    aligned_cluster.append(i)

        if len(aligned_outcome) < 10:
            # Not enough data for meaningful regression
            return None

        outcome_arr = np.array(aligned_outcome)
        discrete_arr = np.array(aligned_discrete)
        continuous_arr = np.array(aligned_continuous)
        cluster_arr = np.array(aligned_cluster) if cluster_var else None

        evaluator = BatteryEvaluator(
            outcome=outcome_arr,
            discrete_exposure=discrete_arr,
            continuous_exposure=continuous_arr,
            cluster_ids=cluster_arr,
            alpha=self.alpha,
        )

        return evaluator.evaluate(
            test_name=test_name,
            outcome_name=outcome_name,
        )

    @staticmethod
    def create_output_path(base_dir: Path, version: str) -> Path:
        """
        Create standardized output path for battery results.

        Args:
            base_dir: Base output directory (e.g., outputs/experiments)
            version: Version string

        Returns:
            Path like outputs/experiments/battery_v073.json
        """
        version_clean = version.replace(".", "")
        return base_dir / f"battery_v{version_clean}.json"


# =============================================================================
# Convenience functions for standard outputs
# =============================================================================

def load_battery_result(path: Path) -> Dict[str, Any]:
    """Load a battery result JSON file."""
    with open(path) as f:
        return json.load(f)


def print_interpretation_matrix(result: Dict[str, Any]):
    """Pretty-print the interpretation matrix from a loaded result."""
    matrix_data = result.get("interpretation_matrix", {})

    tests = matrix_data.get("tests", [])
    outcomes = matrix_data.get("outcomes", [])
    cells = matrix_data.get("cells", [])

    # Build lookup
    lookup = {}
    for cell in cells:
        key = (cell["test"], cell["outcome"])
        lookup[key] = cell["verdict"]

    # Print
    print("\nInterpretation Matrix")
    print("=" * 60)

    # Header
    header = f"{'Test':<25}"
    for o in outcomes:
        header += f" {o[:12]:<12}"
    print(header)
    print("-" * 60)

    # Rows
    for test in tests:
        row = f"{test:<25}"
        for outcome in outcomes:
            verdict = lookup.get((test, outcome), "—")
            row += f" {verdict:<12}"
        print(row)

    # Summary
    summary = matrix_data.get("summary", {})
    print("-" * 60)
    print(f"Summary: {summary.get('positive', 0)} positive, "
          f"{summary.get('negative', 0)} negative, "
          f"{summary.get('null', 0)} null")
