"""
Activity classifications derived from O*NET structure.

Classification scheme:
- Cognitive: 4.A.1.* (Information Input) + 4.A.2.* (Mental Processes)
- Physical: 4.A.3.a.* (Physical Activities, Handling Objects)
- Technical: 4.A.3.b.* (Machines, Computers, Repair)
- Interpersonal: 4.A.4.* (Interacting with Others)

IMPORTANT: Classifications are derived from O*NET element ID structure,
which is hierarchical (dot-separated), NOT fixed-width.
"""

from pathlib import Path
from typing import Literal
import pandas as pd
import numpy as np

GWACategory = Literal['cognitive', 'physical', 'technical', 'interpersonal']


def classify_gwa(element_id: str) -> GWACategory:
    """
    Classify GWA by O*NET element ID structure.

    Parses dot-separated hierarchy, NOT fixed-width slicing.
    Robust to single/double digit segments (e.g., 4.A.10.a.1).

    Args:
        element_id: O*NET element ID (e.g., '4.A.2.a.1')

    Returns:
        Category: 'cognitive', 'physical', 'technical', or 'interpersonal'

    Raises:
        ValueError: If ID structure is invalid or unrecognized
    """
    parts = element_id.split('.')

    if len(parts) < 3:
        raise ValueError(f"Invalid O*NET ID structure: {element_id}")

    if parts[0] != '4' or parts[1] != 'A':
        raise ValueError(f"Not a Work Activity ID: {element_id}")

    segment = parts[2]  # The category segment: '1', '2', '3', or '4'

    if segment in ('1', '2'):
        return 'cognitive'
    elif segment == '3':
        if len(parts) < 4:
            raise ValueError(f"Incomplete 4.A.3.* ID: {element_id}")
        sub = parts[3]  # 'a' (physical) or 'b' (technical)
        if sub == 'a':
            return 'physical'
        elif sub == 'b':
            return 'technical'
        else:
            raise ValueError(f"Unknown 4.A.3.{sub} subcategory: {element_id}")
    elif segment == '4':
        return 'interpersonal'
    else:
        raise ValueError(f"Unknown GWA segment {segment}: {element_id}")


def get_gwa_classifications(onet_path: Path) -> dict[str, GWACategory]:
    """
    Load all GWAs and classify them.

    Returns:
        Dict mapping GWA element_id -> category
    """
    gwa = pd.read_excel(onet_path / "Work Activities.xlsx")
    gwa_ids = gwa['Element ID'].unique()
    return {eid: classify_gwa(eid) for eid in gwa_ids}


def _extract_parent_gwa(dwa_id: str) -> str:
    """
    Extract parent GWA ID from DWA ID using dot-separated parsing.

    DWA structure: 4.A.x.x.x.Ixx.Dxx (variable length segments)
    Parent GWA: 4.A.x.x.x (first 5 dot-separated parts)

    Args:
        dwa_id: Full DWA ID (e.g., '4.A.1.a.1.I01.D01')

    Returns:
        Parent GWA ID (e.g., '4.A.1.a.1')
    """
    parts = dwa_id.split('.')
    if len(parts) < 5:
        raise ValueError(f"Invalid DWA ID structure: {dwa_id}")

    # Parent GWA is first 5 parts: 4.A.x.x.x
    return '.'.join(parts[:5])


def get_dwa_classifications(onet_path: Path) -> dict[str, GWACategory]:
    """
    Classify DWAs by propagating parent GWA classification.

    Uses dot-separated parsing, not fixed-width slicing.

    Returns:
        Dict mapping DWA element_id -> category
    """
    dwa_ref = pd.read_excel(onet_path / "DWA Reference.xlsx")
    gwa_classes = get_gwa_classifications(onet_path)

    dwa_classes = {}
    for _, row in dwa_ref.iterrows():
        dwa_id = row['DWA ID']
        parent_gwa = _extract_parent_gwa(dwa_id)

        # Find matching GWA (parent_gwa should be in gwa_classes)
        if parent_gwa in gwa_classes:
            dwa_classes[dwa_id] = gwa_classes[parent_gwa]
        else:
            # Try partial match on first 4 parts (4.A.x.x)
            partial = '.'.join(parent_gwa.split('.')[:4])
            for gwa_id, category in gwa_classes.items():
                if gwa_id.startswith(partial):
                    dwa_classes[dwa_id] = category
                    break

    return dwa_classes


def get_routine_scores(onet_path: Path) -> dict[str, float]:
    """
    Get routine scores from Work Context 4.C.3.b.7.

    "Importance of Repeating Same Tasks" — higher = more routine.

    Returns:
        Dict mapping O*NET-SOC code -> routine score (raw Data Value scale)
    """
    wc = pd.read_excel(onet_path / "Work Context.xlsx")

    # Filter to routine dimension
    routine = wc[wc['Element ID'] == '4.C.3.b.7'].copy()

    # Use Data Value (importance rating)
    scores = routine.groupby('O*NET-SOC Code')['Data Value'].mean()

    return scores.to_dict()


def get_activity_projected_routine_scores(
    onet_path: Path,
    activity_ids: list[str],
    occupation_matrix: np.ndarray,
    occupation_codes: list[str],
) -> np.ndarray:
    """
    Compute activity-level routine scores by PROJECTING occupation scores.

    ============================================================================
    ENDOGENEITY WARNING
    ============================================================================
    This function creates PROJECTED routine scores, NOT exogenous task attributes.

    The calculation: routine_a = Σ_j (ρ_j(a) * routine_j) / Σ_j ρ_j(a)

    This defines "Task X is routine" based on "Routine occupations do Task X."
    It is NOT an intrinsic property of the task like "Manual" or "Cognitive"
    (which are derived from the DWA definition hierarchy itself).

    IMPLICATIONS FOR PHASE II:
    - Do NOT treat this as exogenous when computing occupation exposure
    - When using in regressions, you MUST control for raw occupation routine
      scores, or results will be tautological (smoothed projection → exposure)
    - Label clearly in all outputs as "projected_routine" not "routine"
    ============================================================================

    Args:
        onet_path: Path to O*NET data
        activity_ids: List of activity IDs
        occupation_matrix: (n_occ, n_act) occupation measures
        occupation_codes: List of occupation codes

    Returns:
        (n_act,) projected routine scores
    """
    occ_routine = get_routine_scores(onet_path)

    # Map occupation codes to routine scores
    occ_scores = np.array([
        occ_routine.get(code[:10], 50.0)  # Default to median if missing
        for code in occupation_codes
    ])

    # Weight activities by occupations that use them
    activity_weights = occupation_matrix.sum(axis=0)
    activity_weights = np.maximum(activity_weights, 1e-10)

    projected_routine = (occupation_matrix.T @ occ_scores) / activity_weights

    return projected_routine
