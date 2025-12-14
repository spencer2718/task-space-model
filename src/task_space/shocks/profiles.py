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
from pathlib import Path
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
    description="AI capability shock: cognitive positive, physical negative",
    required_args=["onet_path"],
    optional_args={"cognitive_weight": 1.0, "physical_weight": -0.5, "technical_weight": 0.5},
)
def shock_capability_v1(
    domain,
    onet_path: Path,
    cognitive_weight: float = 1.0,
    physical_weight: float = -0.5,
    technical_weight: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """
    v1 shock: capability-only based on GWA classification.

    - Cognitive activities (4.A.1.*, 4.A.2.*): positive
    - Physical activities (4.A.3.a.*): negative
    - Technical activities (4.A.3.b.*): moderate positive
    - Interpersonal (4.A.4.*): neutral

    Classification is EXOGENOUS — derived from O*NET hierarchy structure,
    not from occupation characteristics.
    """
    from ..data.classifications import get_dwa_classifications

    dwa_classes = get_dwa_classifications(Path(onet_path))

    weights = {
        'cognitive': cognitive_weight,
        'physical': physical_weight,
        'technical': technical_weight,
        'interpersonal': 0.0,
    }

    I = np.zeros(len(domain.activity_ids))
    for i, act_id in enumerate(domain.activity_ids):
        category = dwa_classes.get(act_id, 'interpersonal')
        I[i] = weights.get(category, 0.0)

    return I


@register_shock(
    name="capability_v2",
    description="AI capability + projected routine amplification",
    required_args=["onet_path", "occupation_matrix", "occupation_codes"],
    optional_args={"cognitive_weight": 1.0, "physical_weight": -0.5, "routine_amplifier": 0.5},
)
def shock_capability_v2(
    domain,
    onet_path: Path,
    occupation_matrix: np.ndarray,
    occupation_codes: list[str],
    cognitive_weight: float = 1.0,
    physical_weight: float = -0.5,
    routine_amplifier: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """
    v2 shock: v1 + PROJECTED routine amplification.

    ============================================================================
    ENDOGENEITY WARNING: The routine component is PROJECTED from occupation
    routine scores, not an exogenous task attribute. See docstring in
    get_activity_projected_routine_scores() for implications.

    When validating this shock, you MUST control for raw occupation routine
    scores to avoid tautological results.
    ============================================================================

    Hypothesis: Structured/routine tasks more susceptible to AI automation.
    """
    from ..data.classifications import get_activity_projected_routine_scores

    I_v1 = shock_capability_v1(
        domain, onet_path, cognitive_weight, physical_weight, **kwargs
    )

    projected_routine = get_activity_projected_routine_scores(
        Path(onet_path), domain.activity_ids, occupation_matrix, occupation_codes
    )

    # Normalize projected routine to [0, 1]
    routine_norm = (projected_routine - projected_routine.min()) / (projected_routine.max() - projected_routine.min() + 1e-10)

    return I_v1 + routine_amplifier * routine_norm


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
