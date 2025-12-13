# task_space v0.4.1
# Phase I external validation (Diagnostic B from Section 4.4)

from .data import (
    load_work_activities,
    load_content_model_reference,
    load_dwa_reference,
    load_tasks_to_dwas,
    load_task_ratings,
    get_gwa_ids,
    get_occupation_codes,
)

from .domain import (
    ActivityDomain,
    OccupationMeasures,
    build_activity_domain,
    build_occupation_measures,
)

from .distances import (
    ActivityDistances,
    compute_activity_distances,
    get_nearest_activities,
    distance_percentiles,
)

from .kernel import (
    KernelMatrix,
    ExposureResult,
    build_kernel_matrix,
    propagate_shock,
    compute_exposure,
    compute_occupation_exposure,
    create_shock_profile,
    compute_overlap,
)

from .diagnostics import (
    MeasureCoherence,
    FaceValidityCheck,
    DistanceCoherence,
    diagnose_measure_coherence,
    spot_check_occupation,
    spot_check_occupations,
    diagnose_distances,
    generate_diagnostic_report,
)

from .validation import (
    OverlapResult,
    OverlapGrid,
    ValidationDataset,
    RegressionResult,
    ValidationResults,
    MonotonicityResult,
    SIGMA_PERCENTILES,
    compute_overlap_stats,
    compute_validation_overlap,
    compute_overlap_grid,
    save_overlap_result,
    load_overlap_result,
    save_overlap_grid,
    load_overlap_grid,
    build_validation_dataset,
    run_validation_regression,
    run_full_validation,
    check_monotonicity,
    plot_monotonicity,
    save_validation_results,
)

from .crosswalk import (
    OnetOesCrosswalk,
    WageComovement,
    onet_to_soc,
    build_onet_oes_crosswalk,
    load_oes_year,
    load_oes_panel,
    compute_wage_comovement,
    aggregate_occupation_measures,
)

__all__ = [
    # Data loading
    "load_work_activities",
    "load_content_model_reference",
    "load_dwa_reference",
    "load_tasks_to_dwas",
    "load_task_ratings",
    "get_gwa_ids",
    "get_occupation_codes",
    # Domain
    "ActivityDomain",
    "OccupationMeasures",
    "build_activity_domain",
    "build_occupation_measures",
    # Distances
    "ActivityDistances",
    "compute_activity_distances",
    "get_nearest_activities",
    "distance_percentiles",
    # Kernel and exposure
    "KernelMatrix",
    "ExposureResult",
    "build_kernel_matrix",
    "propagate_shock",
    "compute_exposure",
    "compute_occupation_exposure",
    "create_shock_profile",
    "compute_overlap",
    # Diagnostics
    "MeasureCoherence",
    "FaceValidityCheck",
    "DistanceCoherence",
    "diagnose_measure_coherence",
    "spot_check_occupation",
    "spot_check_occupations",
    "diagnose_distances",
    "generate_diagnostic_report",
    # Validation (Phase I)
    "OverlapResult",
    "OverlapGrid",
    "ValidationDataset",
    "RegressionResult",
    "ValidationResults",
    "MonotonicityResult",
    "SIGMA_PERCENTILES",
    "compute_overlap_stats",
    "compute_validation_overlap",
    "compute_overlap_grid",
    "save_overlap_result",
    "load_overlap_result",
    "save_overlap_grid",
    "load_overlap_grid",
    "build_validation_dataset",
    "run_validation_regression",
    "run_full_validation",
    "check_monotonicity",
    "plot_monotonicity",
    "save_validation_results",
    # Crosswalk and OES
    "OnetOesCrosswalk",
    "WageComovement",
    "onet_to_soc",
    "build_onet_oes_crosswalk",
    "load_oes_year",
    "load_oes_panel",
    "compute_wage_comovement",
    "aggregate_occupation_measures",
]
