"""
Retrospective battery for validating task-space geometry.

This module provides infrastructure for running Tests A/B/C from Appendix A,
which compare geometry-based exposure measures against established benchmarks.

Submodules:
    exposure: ExposureMeasure base class for discrete/continuous exposure
    evaluator: BatteryEvaluator for regression and interpretation matrix
    runner: BatteryRunner for orchestrating tests and producing output

Example usage:
    >>> from task_space.battery import (
    ...     ExposureMeasure,
    ...     BatteryEvaluator,
    ...     BatteryRunner,
    ...     InterpretationMatrix,
    ... )
    >>> # Define custom exposure measure
    >>> class MyExposure(ExposureMeasure):
    ...     def discrete_exposure(self, unit_id): ...
    ...     def continuous_exposure(self, unit_id): ...
    ...     def metadata(self): ...
    ...     def get_unit_ids(self): ...
    >>> # Run evaluation
    >>> evaluator = BatteryEvaluator(outcome, discrete, continuous)
    >>> result = evaluator.evaluate("test_name", "outcome_name")
    >>> print(result.verdict)  # Verdict.POSITIVE, NEGATIVE, or NULL

Interpretation Matrix:
    The battery produces a matrix mapping (test, outcome) → verdict:
    - "+" : Geometry adds significant positive signal beyond benchmark
    - "-" : Geometry contradicts benchmark prediction
    - "0" : No significant residual effect

    A robust framework should show mostly "+" or "0", few "-".
"""

from .exposure import (
    ExposureType,
    ExposureMetadata,
    ExposureResult,
    ExposureMeasure,
    ComputerExposure,
    RSHExposure,
    RobotExposure,
)

from .evaluator import (
    Verdict,
    RegressionCoefficients,
    ModelComparison,
    EvaluatorResult,
    BatteryEvaluator,
)

from .runner import (
    InterpretationCell,
    InterpretationMatrix,
    BatteryResult,
    TestConfig,
    BatteryRunner,
    load_battery_result,
    print_interpretation_matrix,
)

__all__ = [
    # Exposure
    'ExposureType',
    'ExposureMetadata',
    'ExposureResult',
    'ExposureMeasure',
    'ComputerExposure',
    'RSHExposure',
    'RobotExposure',
    # Evaluator
    'Verdict',
    'RegressionCoefficients',
    'ModelComparison',
    'EvaluatorResult',
    'BatteryEvaluator',
    # Runner
    'InterpretationCell',
    'InterpretationMatrix',
    'BatteryResult',
    'TestConfig',
    'BatteryRunner',
    'load_battery_result',
    'print_interpretation_matrix',
]
