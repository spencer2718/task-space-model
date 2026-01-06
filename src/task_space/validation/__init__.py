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

from .shock_integration import (
    ShockIntegrationResult,
    get_aioe_by_soc_dataframe,
    map_aioe_to_census,
    compute_aioe_geometry_correlations,
    partition_transitions_by_exposure,
    compute_historical_baseline,
    compute_uniform_baseline,
    compute_model_probabilities,
    evaluate_model_on_holdout,
    compute_verdict,
)

from .scaled_costs import (
    ScaledCostsResult,
    ScaledModelResult,
    load_oes_wages_by_census,
    get_wage_coverage,
    build_choice_dataset_with_wages,
    estimate_scaled_model,
    compute_switching_costs,
    compute_median_distances,
    compute_externally_calibrated_costs,
    compute_example_transition_costs,
    lookup_wasserstein_distance,
)

from .reallocation import (
    ReallocationResult,
    load_employment_by_census,
    get_exposed_occupations,
    compute_destination_probabilities,
    aggregate_reallocation_flows,
    compute_absorption_ranking,
    validate_against_holdout,
    load_occupation_names,
    run_reallocation_analysis,
    flag_capacity_constraints,
    split_feasible_constrained,
    compute_validation_verdict,
    CREDENTIAL_GATED_OCCUPATIONS,
)

from .metrics import (
    PerformanceBatteryResult,
    compute_mean_percentile_rank,
    compute_realized_cumulative_mass,
    compute_effective_consideration_set,
    compute_full_destination_probabilities,
    compute_performance_battery,
)

# v0.7.4.0: Spearman correlation utilities
from .spearman import (
    SpearmanResult,
    PerOriginSpearmanResult,
    aggregate_spearman_model_prob,
    per_origin_spearman_model_prob,
    aggregate_spearman_inv_distance,
    per_origin_spearman_inv_distance,
    spearman_with_bootstrap,
    compute_model_probability_matrix,
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
    # Shock Integration
    'ShockIntegrationResult',
    'get_aioe_by_soc_dataframe',
    'map_aioe_to_census',
    'compute_aioe_geometry_correlations',
    'partition_transitions_by_exposure',
    'compute_historical_baseline',
    'compute_uniform_baseline',
    'compute_model_probabilities',
    'evaluate_model_on_holdout',
    'compute_verdict',
    # Scaled Costs
    'ScaledCostsResult',
    'ScaledModelResult',
    'load_oes_wages_by_census',
    'get_wage_coverage',
    'build_choice_dataset_with_wages',
    'estimate_scaled_model',
    'compute_switching_costs',
    'compute_median_distances',
    'compute_externally_calibrated_costs',
    'compute_example_transition_costs',
    'lookup_wasserstein_distance',
    # Reallocation
    'ReallocationResult',
    'load_employment_by_census',
    'get_exposed_occupations',
    'compute_destination_probabilities',
    'aggregate_reallocation_flows',
    'compute_absorption_ranking',
    'validate_against_holdout',
    'load_occupation_names',
    'run_reallocation_analysis',
    'flag_capacity_constraints',
    'split_feasible_constrained',
    'compute_validation_verdict',
    'CREDENTIAL_GATED_OCCUPATIONS',
    # Performance Battery (MS8)
    'PerformanceBatteryResult',
    'compute_mean_percentile_rank',
    'compute_realized_cumulative_mass',
    'compute_effective_consideration_set',
    'compute_full_destination_probabilities',
    'compute_performance_battery',
    # Spearman utilities (v0.7.4.0)
    'SpearmanResult',
    'PerOriginSpearmanResult',
    'aggregate_spearman_model_prob',
    'per_origin_spearman_model_prob',
    'aggregate_spearman_inv_distance',
    'per_origin_spearman_inv_distance',
    'spearman_with_bootstrap',
    'compute_model_probability_matrix',
]
