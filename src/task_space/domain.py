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

from .data import (
    load_work_activities,
    load_dwa_reference,
    load_tasks_to_dwas,
    load_task_ratings,
    get_gwa_ids,
    get_occupation_codes,
)


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


def build_dwa_activity_domain(onet_path: Optional[Path] = None) -> ActivityDomain:
    """
    Construct the DWA-based activity domain T_n.

    Uses Detailed Work Activities (~2,087 activities) as the discrete approximation
    of the continuous task manifold, providing finer granularity than GWA (41).

    Args:
        onet_path: Path to O*NET database directory.

    Returns:
        ActivityDomain with DWA IDs, titles, and uniform reference measure.
    """
    df = load_dwa_reference(onet_path)

    # Get unique DWAs (DWA ID, DWA Title)
    dwa_df = df[["DWA ID", "DWA Title"]].drop_duplicates()
    dwa_df = dwa_df.sort_values("DWA ID")

    activity_ids = dwa_df["DWA ID"].tolist()
    activity_names = dict(zip(dwa_df["DWA ID"], dwa_df["DWA Title"]))
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


def build_dwa_occupation_measures(
    onet_path: Optional[Path] = None,
    aggregator: str = "max",
    normalize: bool = True,
) -> OccupationMeasures:
    """
    Construct occupation probability measures ρ_j over the DWA activity domain.

    DWAs are not directly rated—importance is derived through task linkages.
    Per O*NET methodology ("Ranking Detailed Work Activities", April 2015),
    we aggregate task importance to DWA level.

    Args:
        onet_path: Path to O*NET database directory.
        aggregator: How to aggregate task importance to DWA level.
                   "max" (O*NET default) or "mean".
        normalize: If True, normalize each row to sum to 1 (probability measure).

    Returns:
        OccupationMeasures with ~900 occupations × ~2,087 DWAs.

    Algorithm:
        1. Load task importance ratings (Scale ID = 'IM')
        2. Load task-DWA mappings
        3. Join on (O*NET-SOC Code, Task ID)
        4. Aggregate to (O*NET-SOC Code, DWA ID) using max importance
        5. Pivot to matrix, normalize to [0,1], normalize rows to probabilities
    """
    # Load task importance ratings
    task_ratings = load_task_ratings(onet_path, scale_id="IM", filter_suppressed=True)

    # Load task-DWA mappings
    tasks_to_dwas = load_tasks_to_dwas(onet_path)

    # Load ALL DWA IDs from reference to ensure complete coverage
    dwa_ref = load_dwa_reference(onet_path)
    all_dwa_ids = sorted(dwa_ref["DWA ID"].unique().tolist())

    # Join: get DWA ID for each (occupation, task) pair with importance
    merged = tasks_to_dwas.merge(
        task_ratings,
        on=["O*NET-SOC Code", "Task ID"],
        how="inner",
    )

    # Aggregate to occupation-DWA level
    if aggregator == "max":
        # O*NET default: max importance across linked tasks
        dwa_importance = merged.groupby(
            ["O*NET-SOC Code", "DWA ID"]
        )["Data Value"].max().reset_index()
    elif aggregator == "mean":
        dwa_importance = merged.groupby(
            ["O*NET-SOC Code", "DWA ID"]
        )["Data Value"].mean().reset_index()
    else:
        raise ValueError(f"Unknown aggregator: {aggregator}. Use 'max' or 'mean'.")

    # Pivot to matrix form
    matrix = dwa_importance.pivot_table(
        index="O*NET-SOC Code",
        columns="DWA ID",
        values="Data Value",
        aggfunc="first",
    )

    # Ensure ALL DWAs are included (even those with no task linkages)
    # This ensures alignment with build_dwa_activity_domain()
    for dwa_id in all_dwa_ids:
        if dwa_id not in matrix.columns:
            matrix[dwa_id] = np.nan

    # Fill missing with minimum importance (1.0 on raw scale)
    # DWAs not linked to any task for an occupation get minimum importance
    matrix = matrix.fillna(1.0)

    # Sort columns by DWA ID for consistency
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)

    occupation_codes = matrix.index.tolist()
    activity_ids = matrix.columns.tolist()

    # Normalize importance to [0, 1]: (value - 1) / 4
    # Importance scale is 1-5
    raw_matrix = (matrix.values - 1) / 4

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
