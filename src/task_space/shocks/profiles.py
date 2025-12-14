"""
Built-in shock profiles.

These implement the candidate profiles from the paper:
- uniform: Baseline uniform shock
- gaussian_directed: Shock centered on specific activity (Example 3.4)
- capability_v1: AI capability-only (positive cognitive, negative manual)
- capability_v2: AI capability + structure (adds routine loading)
- rbtc: Routine-biased technological change
"""

import numpy as np
from .registry import register_shock


@register_shock(
    name="uniform",
    description="Uniform shock intensity across all activities",
    optional_args={"intensity": 1.0},
)
def shock_uniform(domain, intensity: float = 1.0, **kwargs) -> np.ndarray:
    """I(a) = intensity for all a."""
    return np.full(len(domain.activity_ids), intensity)


@register_shock(
    name="gaussian_directed",
    description="Gaussian shock centered on activity (Theory Example 3.4)",
    required_args=["center_idx", "dist_matrix"],
    optional_args={"sigma_shock": 0.1, "intensity": 1.0},
)
def shock_gaussian_directed(
    domain,
    center_idx: int,
    dist_matrix: np.ndarray,
    sigma_shock: float = 0.1,
    intensity: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Gaussian shock centered on specific activity.

    I(a) = intensity * exp(-d(a, center)^2 / 2*sigma^2)

    Args:
        domain: Activity domain with activity_ids
        center_idx: Index of center activity in domain
        dist_matrix: (n_act, n_act) distance matrix
        sigma_shock: Shock spread parameter (NOT kernel bandwidth)
        intensity: Peak intensity at center
    """
    distances_from_center = dist_matrix[center_idx, :]
    return intensity * np.exp(-distances_from_center**2 / (2 * sigma_shock**2))


@register_shock(
    name="capability_v1",
    description="AI capability-only: positive cognitive/language, negative manual",
    required_args=["activity_classifications"],
    optional_args={"cognitive_weight": 1.0, "manual_weight": -0.5},
)
def shock_capability_v1(
    domain,
    activity_classifications: dict[str, str],
    cognitive_weight: float = 1.0,
    manual_weight: float = -0.5,
    **kwargs,
) -> np.ndarray:
    """
    v1 shock profile: capability-only.

    Positive loading on cognitive/language activities.
    Negative loading on manual/physical activities.
    Neutral (0) on unclassified activities.

    Args:
        domain: Activity domain
        activity_classifications: Map activity_id -> classification
            Classifications: 'cognitive', 'language', 'manual', 'physical', etc.
    """
    I = np.zeros(len(domain.activity_ids))

    cognitive_tags = {'cognitive', 'language', 'information_processing', 'analytical'}
    manual_tags = {'manual', 'physical', 'motor'}

    for i, act_id in enumerate(domain.activity_ids):
        classification = activity_classifications.get(act_id, '').lower()
        if classification in cognitive_tags:
            I[i] = cognitive_weight
        elif classification in manual_tags:
            I[i] = manual_weight

    return I


@register_shock(
    name="capability_v2",
    description="AI capability + structure: v1 plus routine amplification",
    required_args=["activity_classifications", "routine_scores"],
    optional_args={"cognitive_weight": 1.0, "manual_weight": -0.5, "routine_amplifier": 0.5},
)
def shock_capability_v2(
    domain,
    activity_classifications: dict[str, str],
    routine_scores: np.ndarray,
    cognitive_weight: float = 1.0,
    manual_weight: float = -0.5,
    routine_amplifier: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """
    v2 shock profile: capability + structure.

    v1 loadings plus additional weight on structured/routine activities.
    Hypothesis: structure amplifies AI exposure beyond raw capability.
    """
    I_v1 = shock_capability_v1(
        domain, activity_classifications, cognitive_weight, manual_weight
    )

    # Amplify where routine score is high
    routine_range = routine_scores.max() - routine_scores.min()
    if routine_range > 0:
        routine_normalized = (routine_scores - routine_scores.min()) / routine_range
    else:
        routine_normalized = np.zeros_like(routine_scores)

    return I_v1 + routine_amplifier * routine_normalized


@register_shock(
    name="rbtc",
    description="Routine-biased technological change (retrospective evaluation)",
    required_args=["routine_scores"],
    optional_args={"intensity": 1.0},
)
def shock_rbtc(
    domain,
    routine_scores: np.ndarray,
    intensity: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    RBTC shock for retrospective (1990-2007) evaluation.

    Positive loading proportional to routine/repetitive score.
    """
    return intensity * routine_scores
