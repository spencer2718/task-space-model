"""
Activity domain and occupation measure construction.

Implements the discrete approximation of the task manifold (Definition: Task Domain)
and occupation measures (Definition: Occupation Measures) using O*NET GWA data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data import load_work_activities, get_gwa_ids, get_occupation_codes


@dataclass
class ActivityDomain:
    """
    Discrete activity domain T_n (Definition: Task Domain).

    Attributes:
        activity_ids: List of GWA Element IDs (e.g., ['4.A.1.a.1', ...])
        activity_names: Dict mapping Element ID to human-readable name
        n_activities: Number of activities (41 for GWA)
        reference_measure: Weights μ(a) for each activity (uniform by default)
    """
    activity_ids: list[str]
    activity_names: dict[str, str]
    n_activities: int
    reference_measure: np.ndarray  # Shape: (n_activities,)


@dataclass
class OccupationMeasures:
    """
    Occupation probability measures over activity domain (Definition: Occupation Measures).

    Attributes:
        occupation_codes: List of O*NET-SOC codes
        occupation_matrix: Matrix of ρ_j(a) values, shape (n_occupations, n_activities)
                          Each row is a probability distribution over activities.
        activity_ids: List of activity IDs (columns of matrix)
        raw_matrix: Unnormalized importance matrix (for diagnostics)
    """
    occupation_codes: list[str]
    occupation_matrix: np.ndarray  # Shape: (n_occupations, n_activities)
    activity_ids: list[str]
    raw_matrix: np.ndarray  # Unnormalized, shape: (n_occupations, n_activities)


def build_activity_domain(onet_path: Optional[Path] = None) -> ActivityDomain:
    """
    Construct the GWA-based activity domain T_n.

    Uses Generalized Work Activities (41 activities) as the discrete approximation
    of the continuous task manifold.

    Args:
        onet_path: Path to O*NET database directory.

    Returns:
        ActivityDomain with GWA IDs, names, and uniform reference measure.
    """
    df = load_work_activities(onet_path, scale_id="IM", filter_suppressed=False)

    # Get unique activities
    activity_df = df[["Element ID", "Element Name"]].drop_duplicates()
    activity_df = activity_df.sort_values("Element ID")

    activity_ids = activity_df["Element ID"].tolist()
    activity_names = dict(zip(activity_df["Element ID"], activity_df["Element Name"]))
    n_activities = len(activity_ids)

    # Uniform reference measure (baseline per paper Section 4)
    reference_measure = np.ones(n_activities) / n_activities

    return ActivityDomain(
        activity_ids=activity_ids,
        activity_names=activity_names,
        n_activities=n_activities,
        reference_measure=reference_measure,
    )


def build_occupation_measures(
    onet_path: Optional[Path] = None,
    normalize: bool = True,
    importance_floor: float = 0.0,
) -> OccupationMeasures:
    """
    Construct occupation probability measures ρ_j over the activity domain.

    Each occupation is represented as a probability distribution over activities,
    where weights come from O*NET Importance ratings normalized to [0,1] then
    normalized to sum to 1.

    Args:
        onet_path: Path to O*NET database directory.
        normalize: If True, normalize each row to sum to 1 (probability measure).
        importance_floor: Minimum importance value after normalization (default 0).

    Returns:
        OccupationMeasures with occupation codes and ρ_j matrix.

    Construction follows paper Remark 4.2:
        1. Load Importance ratings (Scale ID = 'IM')
        2. Filter suppressed rows
        3. Normalize Importance to [0,1] via (value - 1) / 4
        4. Pivot to occupation × activity matrix
        5. Normalize rows to probability measures
    """
    df = load_work_activities(onet_path, scale_id="IM", filter_suppressed=True)

    # Normalize importance to [0, 1]: (value - 1) / 4
    # Importance scale is 1-5, so this maps to 0-1
    df["Normalized"] = (df["Data Value"] - 1) / 4

    # Apply floor if specified
    if importance_floor > 0:
        df["Normalized"] = df["Normalized"].clip(lower=importance_floor)

    # Pivot to occupation × activity matrix
    matrix = df.pivot_table(
        index="O*NET-SOC Code",
        columns="Element ID",
        values="Normalized",
        aggfunc="first",
    )

    # Fill missing values with 0 (activity not relevant to occupation)
    matrix = matrix.fillna(0)

    # Sort columns by Element ID for consistency
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)

    occupation_codes = matrix.index.tolist()
    activity_ids = matrix.columns.tolist()
    raw_matrix = matrix.values.copy()

    # Normalize rows to probability measures
    if normalize:
        row_sums = raw_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized_matrix = raw_matrix / row_sums
    else:
        normalized_matrix = raw_matrix

    return OccupationMeasures(
        occupation_codes=occupation_codes,
        occupation_matrix=normalized_matrix,
        activity_ids=activity_ids,
        raw_matrix=raw_matrix,
    )


def get_occupation_name(soc_code: str, onet_path: Optional[Path] = None) -> str:
    """
    Get occupation title for an O*NET-SOC code.

    Args:
        soc_code: O*NET-SOC code (e.g., '15-1252.00')
        onet_path: Path to O*NET database directory.

    Returns:
        Occupation title string.
    """
    df = load_work_activities(onet_path, scale_id="IM", filter_suppressed=False)
    matches = df[df["O*NET-SOC Code"] == soc_code]
    if len(matches) == 0:
        return soc_code
    # Work Activities doesn't have Title column, need to load from another file
    # For now, return the code
    return soc_code
