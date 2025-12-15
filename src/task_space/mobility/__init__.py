"""
Mobility module for occupation transition analysis.

This module provides tools for analyzing worker occupation transitions
and validating the semantic-institutional distance decomposition.

Primary Components:
- Institutional distance computation (job zones + certification)
- Census↔O*NET crosswalk for CPS occupation codes
- CPS transition filters for measurement error correction
- Conditional logit model for destination choice analysis

Main Results (paper Section 4.4):
- α (semantic): 2.994 (t = 98.5) - workers prefer semantically similar destinations
- β (institutional): 0.215 (t = 63.4) - workers avoid institutional barriers
- Both components independently predictive (r = 0.36)

Example:
    >>> from task_space.mobility import (
    ...     build_institutional_distance_matrix,
    ...     load_census_onet_crosswalk,
    ...     load_canonical_results,
    ... )
    >>> d_inst = build_institutional_distance_matrix()
    >>> print(f"Occupations: {d_inst.n_occupations}")
    >>> print(f"Cert coverage: {d_inst.cert_coverage:.1%}")
    >>>
    >>> results = load_canonical_results()
    >>> print(f"α = {results.alpha:.3f} (t = {results.alpha_t:.1f})")
    >>> print(f"β = {results.beta:.3f} (t = {results.beta_t:.1f})")
"""

from .institutional import (
    InstitutionalDistanceResult,
    build_institutional_distance_matrix,
    compute_institutional_distance,
    get_zone_difference,
    load_job_zones,
    load_certification_importance,
    # v0.6.6.0: Asymmetric institutional distance
    AsymmetricInstitutionalDistanceResult,
    build_asymmetric_institutional_distance,
    verify_asymmetric_decomposition,
)

from .census_crosswalk import (
    CensusCrosswalkResult,
    load_census_onet_crosswalk,
    aggregate_distances_to_census,
    get_census_distance,
)

from .filters import (
    FilterStats,
    FilterPipeline,
    build_verified_transitions,
    apply_persistence_filter,
    apply_demographic_validation,
    apply_employment_filter,
    load_verified_transitions,
)

from .choice_model import (
    ChoiceModelResult,
    build_choice_dataset,
    fit_conditional_logit,
    compute_odds_ratios,
    load_canonical_results,
    # v0.6.6.0: Asymmetric choice model
    AsymmetricChoiceModelResult,
    build_asymmetric_choice_dataset,
    fit_asymmetric_conditional_logit,
    compute_asymmetric_odds_ratios,
)

__all__ = [
    # Institutional distance
    "InstitutionalDistanceResult",
    "build_institutional_distance_matrix",
    "compute_institutional_distance",
    "get_zone_difference",
    "load_job_zones",
    "load_certification_importance",
    # v0.6.6.0: Asymmetric institutional distance
    "AsymmetricInstitutionalDistanceResult",
    "build_asymmetric_institutional_distance",
    "verify_asymmetric_decomposition",
    # Census crosswalk
    "CensusCrosswalkResult",
    "load_census_onet_crosswalk",
    "aggregate_distances_to_census",
    "get_census_distance",
    # Transition filters
    "FilterStats",
    "FilterPipeline",
    "build_verified_transitions",
    "apply_persistence_filter",
    "apply_demographic_validation",
    "apply_employment_filter",
    "load_verified_transitions",
    # Choice model
    "ChoiceModelResult",
    "build_choice_dataset",
    "fit_conditional_logit",
    "compute_odds_ratios",
    "load_canonical_results",
    # v0.6.6.0: Asymmetric choice model
    "AsymmetricChoiceModelResult",
    "build_asymmetric_choice_dataset",
    "fit_asymmetric_conditional_logit",
    "compute_asymmetric_odds_ratios",
]
