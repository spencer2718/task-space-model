"""
Validation utilities for task-space model.

Submodules:
    regression: Clustered standard errors regression
    diagnostics: Kernel and distance diagnostics
    permutation: Permutation tests and cross-validation
"""

from .regression import (
    RegressionResult,
    compute_clustered_se,
    run_validation_regression,
    simple_regression,
)

from .diagnostics import (
    DistanceDiagnostics,
    KernelDiagnostics,
    ValidationDiagnostics,
    diagnose_distances,
    diagnose_kernel,
    run_diagnostics,
)

from .permutation import (
    PermutationResult,
    CrossValidationResult,
    run_permutation_test,
    run_cross_validation,
    run_random_baseline_comparison,
)

__all__ = [
    # Regression
    'RegressionResult',
    'compute_clustered_se',
    'run_validation_regression',
    'simple_regression',
    # Diagnostics
    'DistanceDiagnostics',
    'KernelDiagnostics',
    'ValidationDiagnostics',
    'diagnose_distances',
    'diagnose_kernel',
    'run_diagnostics',
    # Permutation
    'PermutationResult',
    'CrossValidationResult',
    'run_permutation_test',
    'run_cross_validation',
    'run_random_baseline_comparison',
]
