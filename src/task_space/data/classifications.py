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

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import pandas as pd
import numpy as np

GWACategory = Literal['cognitive', 'physical', 'technical', 'interpersonal']


# Default O*NET path
DEFAULT_ONET_PATH = Path(__file__).parent.parent.parent.parent / "data" / "onet" / "db_30_0_excel"


# ============================================================================
# Acemoglu-Autor (2011) Task Classification
# ============================================================================
#
# 16 O*NET elements mapped to 5 task dimensions:
# - Non-routine cognitive analytical (abstract reasoning)
# - Non-routine cognitive interpersonal (communication, management)
# - Routine cognitive (repetitive mental tasks)
# - Routine manual (repetitive physical tasks)
# - Non-routine manual physical (operating equipment, dexterity)
#
# Reference: Acemoglu & Autor (2011), "Skills, Tasks and Technologies"
# ============================================================================

AA_ELEMENT_MAP = {
    'nr_cognitive_analytical': [
        '4.A.2.a.4',  # Analyzing Data or Information (WA)
        '4.A.2.b.2',  # Thinking Creatively (WA)
        '4.A.4.a.1',  # Interpreting Meaning of Information (WA)
    ],
    'nr_cognitive_interpersonal': [
        '4.A.4.a.4',  # Establishing Interpersonal Relationships (WA)
        '4.A.4.b.4',  # Guiding, Directing, Motivating Subordinates (WA)
        '4.A.4.b.5',  # Coaching and Developing Others (WA)
    ],
    'routine_cognitive': [
        '4.C.3.b.7',  # Importance of Repeating Same Tasks (WC)
        '4.C.3.b.4',  # Importance of Being Exact or Accurate (WC)
        '4.C.3.b.8',  # Structured vs Unstructured Work (WC) - REVERSED
    ],
    'routine_manual': [
        '4.C.3.d.3',  # Pace Determined by Speed of Equipment (WC)
        '4.A.3.a.3',  # Controlling Machines and Processes (WA)
        '4.C.2.d.1.i',  # Spend Time Making Repetitive Motions (WC)
    ],
    'nr_manual_physical': [
        '4.A.3.a.4',  # Operating Vehicles/Mechanized Equipment (WA)
        '4.C.2.d.1.g',  # Using Hands to Handle/Control Objects (WC)
        '1.A.2.a.2',  # Manual Dexterity (AB)
        '1.A.1.f.1',  # Spatial Orientation (AB)
    ],
}

# Elements that need to be reversed (higher = LESS routine)
AA_REVERSE_ELEMENTS = {'4.C.3.b.8'}  # "Structured vs Unstructured" - high = unstructured


@dataclass
class AATaskScores:
    """
    Acemoglu-Autor (2011) 5-factor task scores for an occupation.

    All scores are standardized (mean=0, std=1 across occupations).
    """
    nr_cognitive_analytical: float
    nr_cognitive_interpersonal: float
    routine_cognitive: float
    routine_manual: float
    nr_manual_physical: float
    occ_code: str = ""

    @property
    def rti(self) -> float:
        """
        Routine Task Intensity index.

        RTI = routine - (abstract + manual) / 2

        Where:
        - routine = (routine_cognitive + routine_manual) / 2
        - abstract = (nr_cognitive_analytical + nr_cognitive_interpersonal) / 2
        - manual = nr_manual_physical

        Higher RTI = more routine-intensive occupation.
        """
        routine = (self.routine_cognitive + self.routine_manual) / 2
        abstract = (self.nr_cognitive_analytical + self.nr_cognitive_interpersonal) / 2
        manual = self.nr_manual_physical
        return routine - (abstract + manual) / 2

    @property
    def abstract(self) -> float:
        """Abstract task intensity (cognitive non-routine)."""
        return (self.nr_cognitive_analytical + self.nr_cognitive_interpersonal) / 2

    @property
    def routine(self) -> float:
        """Routine task intensity (cognitive + manual)."""
        return (self.routine_cognitive + self.routine_manual) / 2

    @property
    def manual(self) -> float:
        """Manual task intensity (non-routine physical)."""
        return self.nr_manual_physical


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


# ============================================================================
# Acemoglu-Autor Task Score Loading
# ============================================================================

def _load_onet_element_scores(
    onet_path: Path,
    element_ids: list[str],
    scale_id: str = "IM",
) -> pd.DataFrame:
    """
    Load O*NET element scores from appropriate source files.

    Args:
        onet_path: Path to O*NET database
        element_ids: List of element IDs to load
        scale_id: Scale to use ("IM" = Importance, "LV" = Level)

    Returns:
        DataFrame with columns: [occ_code, element_id, score]
    """
    # Determine which file(s) to load based on element ID prefixes
    wa_ids = [e for e in element_ids if e.startswith('4.A.')]
    wc_ids = [e for e in element_ids if e.startswith('4.C.')]
    ab_ids = [e for e in element_ids if e.startswith('1.A.')]

    dfs = []

    if wa_ids:
        wa = pd.read_excel(onet_path / "Work Activities.xlsx")
        wa = wa[wa['Element ID'].isin(wa_ids)]
        wa = wa[wa['Scale ID'] == scale_id]
        wa = wa[['O*NET-SOC Code', 'Element ID', 'Data Value']].copy()
        wa.columns = ['occ_code', 'element_id', 'score']
        dfs.append(wa)

    if wc_ids:
        wc = pd.read_excel(onet_path / "Work Context.xlsx")
        wc = wc[wc['Element ID'].isin(wc_ids)]
        # Work Context uses different scales - filter for context scale
        if 'Scale ID' in wc.columns:
            # Some WC elements use CX (context) scale
            wc = wc[wc['Scale ID'].isin(['CX', 'CT', scale_id])]
        wc = wc[['O*NET-SOC Code', 'Element ID', 'Data Value']].copy()
        wc.columns = ['occ_code', 'element_id', 'score']
        dfs.append(wc)

    if ab_ids:
        ab = pd.read_excel(onet_path / "Abilities.xlsx")
        ab = ab[ab['Element ID'].isin(ab_ids)]
        ab = ab[ab['Scale ID'] == scale_id]
        ab = ab[['O*NET-SOC Code', 'Element ID', 'Data Value']].copy()
        ab.columns = ['occ_code', 'element_id', 'score']
        dfs.append(ab)

    if not dfs:
        raise ValueError(f"No data found for element IDs: {element_ids}")

    return pd.concat(dfs, ignore_index=True)


def get_aa_task_scores(
    onet_path: Optional[Path] = None,
) -> dict[str, AATaskScores]:
    """
    Load Acemoglu-Autor (2011) 5-factor task scores for all occupations.

    Loads 16 O*NET elements, standardizes within-element, then averages
    to create 5 task dimensions. Returns RTI and component scores.

    Args:
        onet_path: Path to O*NET database. Defaults to data/onet/db_30_0_excel.

    Returns:
        Dict mapping O*NET-SOC code to AATaskScores dataclass.

    Example:
        >>> scores = get_aa_task_scores()
        >>> scores['43-9021.00'].rti  # Data Entry Keyers - high RTI
        1.234
        >>> scores['11-1011.00'].rti  # Chief Executives - low RTI
        -0.876
    """
    onet_path = Path(onet_path) if onet_path else DEFAULT_ONET_PATH

    # Collect all element IDs
    all_element_ids = []
    for elements in AA_ELEMENT_MAP.values():
        all_element_ids.extend(elements)

    # Load raw scores
    raw_scores = _load_onet_element_scores(onet_path, all_element_ids)

    # Handle missing values
    raw_scores = raw_scores.dropna(subset=['score'])
    raw_scores['score'] = pd.to_numeric(raw_scores['score'], errors='coerce')
    raw_scores = raw_scores.dropna(subset=['score'])

    # Reverse elements where needed (higher original = lower routine)
    for element_id in AA_REVERSE_ELEMENTS:
        mask = raw_scores['element_id'] == element_id
        if mask.any():
            # Reverse by negating (will be standardized anyway)
            raw_scores.loc[mask, 'score'] = -raw_scores.loc[mask, 'score']

    # Standardize within each element
    standardized = raw_scores.copy()
    for element_id in all_element_ids:
        mask = standardized['element_id'] == element_id
        if mask.any():
            scores = standardized.loc[mask, 'score']
            mean = scores.mean()
            std = scores.std()
            if std > 0:
                standardized.loc[mask, 'score'] = (scores - mean) / std
            else:
                standardized.loc[mask, 'score'] = 0.0

    # Aggregate to 5 dimensions per occupation
    occ_codes = standardized['occ_code'].unique()
    results = {}

    for occ_code in occ_codes:
        occ_data = standardized[standardized['occ_code'] == occ_code]

        dimension_scores = {}
        for dim_name, element_ids in AA_ELEMENT_MAP.items():
            dim_data = occ_data[occ_data['element_id'].isin(element_ids)]
            if len(dim_data) > 0:
                dimension_scores[dim_name] = dim_data['score'].mean()
            else:
                dimension_scores[dim_name] = 0.0  # Missing = neutral

        results[occ_code] = AATaskScores(
            nr_cognitive_analytical=dimension_scores['nr_cognitive_analytical'],
            nr_cognitive_interpersonal=dimension_scores['nr_cognitive_interpersonal'],
            routine_cognitive=dimension_scores['routine_cognitive'],
            routine_manual=dimension_scores['routine_manual'],
            nr_manual_physical=dimension_scores['nr_manual_physical'],
            occ_code=occ_code,
        )

    return results


def get_aa_task_scores_df(
    onet_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Get AA task scores as a DataFrame for easy merging.

    Args:
        onet_path: Path to O*NET database.

    Returns:
        DataFrame with columns:
        - occ_code: O*NET-SOC code
        - nr_cognitive_analytical, nr_cognitive_interpersonal,
          routine_cognitive, routine_manual, nr_manual_physical
        - rti: Routine Task Intensity index
        - abstract, routine, manual: Aggregate dimensions
    """
    scores = get_aa_task_scores(onet_path)

    rows = []
    for occ_code, s in scores.items():
        rows.append({
            'occ_code': occ_code,
            'nr_cognitive_analytical': s.nr_cognitive_analytical,
            'nr_cognitive_interpersonal': s.nr_cognitive_interpersonal,
            'routine_cognitive': s.routine_cognitive,
            'routine_manual': s.routine_manual,
            'nr_manual_physical': s.nr_manual_physical,
            'rti': s.rti,
            'abstract': s.abstract,
            'routine': s.routine,
            'manual': s.manual,
        })

    return pd.DataFrame(rows)


# ============================================================================
# Job Zone Loading
# ============================================================================

def get_job_zones(
    onet_path: Optional[Path] = None,
) -> dict[str, int]:
    """
    Load O*NET Job Zones (1-5 scale of preparation required).

    Job Zone definitions:
    - 1: Little or No Preparation Needed
    - 2: Some Preparation Needed
    - 3: Medium Preparation Needed
    - 4: Considerable Preparation Needed
    - 5: Extensive Preparation Needed

    Args:
        onet_path: Path to O*NET database.

    Returns:
        Dict mapping O*NET-SOC code to job zone (1-5).
    """
    onet_path = Path(onet_path) if onet_path else DEFAULT_ONET_PATH

    jz = pd.read_excel(onet_path / "Job Zones.xlsx")

    return dict(zip(jz['O*NET-SOC Code'], jz['Job Zone'].astype(int)))
