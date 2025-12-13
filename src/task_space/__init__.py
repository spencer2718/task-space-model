# task_space v0.4.0
# Empirical implementation of Section 4 (GWA-based activity domain, Recipe X distances)

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
]
