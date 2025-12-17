# task_space v0.7.1.0
# Dead code cleanup complete

# Domain (core, unchanged)
from .domain import (
    ActivityDomain,
    OccupationMeasures,
    build_activity_domain,
    build_dwa_activity_domain,
    build_occupation_measures,
    build_dwa_occupation_measures,
)

# Data loading
from .data import (
    # O*NET
    load_onet_data,
    load_work_activities,
    load_content_model_reference,
    load_dwa_reference,
    load_tasks_to_dwas,
    load_task_ratings,
    get_dwa_titles,
    get_task_ratings,
    get_gwa_ids,
    get_occupation_codes,
    # OES
    load_oes_year,
    load_oes_panel,
    compute_wage_comovement,
    WageComovement,
    # Crosswalk
    onet_to_soc,
    build_onet_oes_crosswalk,
    aggregate_occupation_measures,
    OnetOesCrosswalk,
    # Artifacts
    get_embeddings,
    get_distance_matrix,
    clear_cache,
)

# Similarity
from .similarity import (
    # Kernel
    calibrate_sigma,
    check_kernel_discrimination,
    build_kernel_matrix,
    # Overlap
    compute_jaccard_overlap,
    compute_kernel_overlap,
    compute_normalized_overlap,
    # Embeddings
    compute_activity_embeddings,
    compute_embedding_distances,
    embeddings_to_similarity,
    # Distances
    ActivityDistances,
    compute_recipe_x_distances,
    compute_recipe_y_distances,
    get_nearest_activities,
    distance_percentiles,
)

# Shocks
from .shocks import (
    register_shock,
    get_shock,
    list_shocks,
    describe_shock,
    propagate_shock,
    compute_exposure_from_shock,
    exposure_stats,
    PropagationResult,
)

# Validation
from .validation import (
    RegressionResult,
    compute_clustered_se,
    run_validation_regression,
    simple_regression,
    run_diagnostics,
    run_permutation_test,
    run_cross_validation,
    PermutationResult,
    CrossValidationResult,
)

# Experiments
from .experiments import (
    ExperimentConfig,
    run_experiment,
)

__all__ = [
    # Domain
    "ActivityDomain",
    "OccupationMeasures",
    "build_activity_domain",
    "build_dwa_activity_domain",
    "build_occupation_measures",
    "build_dwa_occupation_measures",
    # Data - O*NET
    "load_onet_data",
    "load_work_activities",
    "load_content_model_reference",
    "load_dwa_reference",
    "load_tasks_to_dwas",
    "load_task_ratings",
    "get_dwa_titles",
    "get_task_ratings",
    "get_gwa_ids",
    "get_occupation_codes",
    # Data - OES
    "load_oes_year",
    "load_oes_panel",
    "compute_wage_comovement",
    "WageComovement",
    # Data - Crosswalk
    "onet_to_soc",
    "build_onet_oes_crosswalk",
    "aggregate_occupation_measures",
    "OnetOesCrosswalk",
    # Data - Artifacts
    "get_embeddings",
    "get_distance_matrix",
    "clear_cache",
    # Similarity - Kernel
    "calibrate_sigma",
    "check_kernel_discrimination",
    "build_kernel_matrix",
    # Similarity - Overlap
    "compute_jaccard_overlap",
    "compute_kernel_overlap",
    "compute_normalized_overlap",
    # Similarity - Embeddings
    "compute_activity_embeddings",
    "compute_embedding_distances",
    "embeddings_to_similarity",
    # Similarity - Distances
    "ActivityDistances",
    "compute_recipe_x_distances",
    "compute_recipe_y_distances",
    "get_nearest_activities",
    "distance_percentiles",
    # Shocks
    "register_shock",
    "get_shock",
    "list_shocks",
    "describe_shock",
    "propagate_shock",
    "compute_exposure_from_shock",
    "exposure_stats",
    "PropagationResult",
    # Validation
    "RegressionResult",
    "compute_clustered_se",
    "run_validation_regression",
    "simple_regression",
    "run_diagnostics",
    "run_permutation_test",
    "run_cross_validation",
    "PermutationResult",
    "CrossValidationResult",
    # Experiments
    "ExperimentConfig",
    "run_experiment",
]
