# task_space v0.6.2
# Systematic representation comparison study
# Kernel-weighted overlap validated (t=27.65), continuous structure confirmed
# See spec_0.6.2.md for implementation details

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
    build_dwa_activity_domain,
    build_occupation_measures,
    build_dwa_occupation_measures,
)

from .distances import (
    ActivityDistances,
    compute_activity_distances,
    compute_text_embedding_distances,
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
    DWASparsityReport,
    GeometryComparison,
    diagnose_measure_coherence,
    spot_check_occupation,
    spot_check_occupations,
    diagnose_distances,
    diagnose_dwa_sparsity,
    compare_geometries,
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

from .baseline import (
    BinaryOverlapResult,
    BaselineRegressionResult,
    compute_binary_overlap,
    run_baseline_regression,
    save_baseline_results,
)

from .sae import (
    SAEConfig,
    TrainingLog,
    SparseAutoencoder,
    sae_loss,
    compute_l0,
    train_sae,
    save_sae,
    load_sae,
    extract_sparse_features,
)

from .comparison import (
    # Data classes
    RepresentationResult,
    PermutationResult,
    CrossValidationResult,
    BandwidthSensitivityResult,
    HybridModelResult,
    REPRESENTATION_NAMES,
    # Discrete representations
    compute_binary_jaccard,
    compute_weighted_jaccard,
    compute_cosine_binary,
    # Text embedding representations
    compute_mpnet_embeddings,
    compute_jobbert_embeddings,
    compute_e5_embeddings,
    embeddings_to_kernel_overlap,
    # O*NET structured representations
    load_onet_structured_dimension,
    compute_structured_similarity,
    # Validation functions
    run_validation_regression,
    run_permutation_test,
    run_cross_validation,
    run_hybrid_regression,
    # Output
    generate_phase2_summary,
    save_phase2_results,
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
    "build_dwa_activity_domain",
    "build_occupation_measures",
    "build_dwa_occupation_measures",
    # Distances
    "ActivityDistances",
    "compute_activity_distances",
    "compute_text_embedding_distances",
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
    "DWASparsityReport",
    "GeometryComparison",
    "diagnose_measure_coherence",
    "spot_check_occupation",
    "spot_check_occupations",
    "diagnose_distances",
    "diagnose_dwa_sparsity",
    "compare_geometries",
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
    # Baseline (Phase A)
    "BinaryOverlapResult",
    "BaselineRegressionResult",
    "compute_binary_overlap",
    "run_baseline_regression",
    "save_baseline_results",
    # SAE (Phase B)
    "SAEConfig",
    "TrainingLog",
    "SparseAutoencoder",
    "sae_loss",
    "compute_l0",
    "train_sae",
    "save_sae",
    "load_sae",
    "extract_sparse_features",
    # Comparison (Phase 2)
    "RepresentationResult",
    "PermutationResult",
    "CrossValidationResult",
    "BandwidthSensitivityResult",
    "HybridModelResult",
    "REPRESENTATION_NAMES",
    "compute_binary_jaccard",
    "compute_weighted_jaccard",
    "compute_cosine_binary",
    "compute_mpnet_embeddings",
    "compute_jobbert_embeddings",
    "compute_e5_embeddings",
    "embeddings_to_kernel_overlap",
    "load_onet_structured_dimension",
    "compute_structured_similarity",
    "run_validation_regression",
    "run_permutation_test",
    "run_cross_validation",
    "run_hybrid_regression",
    "generate_phase2_summary",
    "save_phase2_results",
]
