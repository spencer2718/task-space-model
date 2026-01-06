"""
Canonical IO module for mobility data and distance matrices.

This is THE approved entrypoint for experiment scripts to load:
- CPS transition data
- Wasserstein distance matrices
- Institutional distance matrices
- Distance matrix aggregation

All functions use the canonical artifact paths from task_space.data.artifacts.
Do NOT hardcode .cache paths in experiment scripts.

Usage:
    from task_space.mobility.io import (
        load_transitions,
        get_holdout_transitions,
        load_wasserstein_census,
        load_institutional_census,
        load_distance_matrix,
    )

    # Load all transitions
    df = load_transitions()

    # Load 2024+ holdout only
    holdout = get_holdout_transitions()

    # Load distance matrices
    d_wass, codes = load_wasserstein_census()
    d_inst, codes = load_institutional_census()
"""

from pathlib import Path
from typing import List, Literal, Tuple, Any, Optional

import numpy as np
import pandas as pd

from task_space.data.artifacts import CACHE_DIR, CACHE_VERSION


# =============================================================================
# Cache Path Helpers
# =============================================================================

def _get_mobility_cache_path(filename: str) -> Path:
    """
    Get canonical path for mobility artifacts.

    Uses the central CACHE_DIR from artifacts.py.
    """
    return CACHE_DIR / CACHE_VERSION / "mobility" / filename


# =============================================================================
# Transition Data Loading
# =============================================================================

# Default path for verified transitions (canonical location)
_DEFAULT_TRANSITIONS_PATH = "data/processed/mobility/verified_transitions.parquet"

# Default training years (2015-2019, 2022-2023; 2020-2021 excluded due to COVID)
DEFAULT_TRAIN_YEARS = list(range(2015, 2020)) + [2022, 2023]

# Default holdout year
DEFAULT_HOLDOUT_YEAR = 2024


def load_transitions(
    *,
    holdout: bool = False,
    path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load CPS transition data with year column added.

    This is the canonical function for loading verified CPS transitions.
    The year column is computed from YEARMONTH // 100.

    Args:
        holdout: If True, filter to holdout year (2024+). If False, return all.
        path: Override path to transitions parquet. Uses canonical location if None.

    Returns:
        DataFrame with columns including:
            - origin_occ: Census 2010 origin occupation code
            - dest_occ: Census 2010 destination occupation code
            - YEARMONTH: Year-month integer (YYYYMM)
            - year: Extracted year (added by this function)

    Artifacts read:
        data/processed/mobility/verified_transitions.parquet

    Example:
        >>> df = load_transitions()
        >>> print(f"Total transitions: {len(df):,}")
        >>> print(f"Year range: {df['year'].min()}-{df['year'].max()}")
    """
    if path is None:
        path = _DEFAULT_TRANSITIONS_PATH

    df = pd.read_parquet(path)

    # Add year column (canonical pattern from all scripts)
    df["year"] = df["YEARMONTH"] // 100

    if holdout:
        df = df[df["year"] >= DEFAULT_HOLDOUT_YEAR].copy()

    return df


def get_holdout_transitions(
    *,
    year: int = DEFAULT_HOLDOUT_YEAR,
    path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper to load holdout transitions.

    Equivalent to load_transitions(holdout=True) but with explicit year control.

    Args:
        year: Holdout year threshold (>= this year). Default: 2024.
        path: Override path to transitions parquet.

    Returns:
        DataFrame filtered to year >= threshold.

    Example:
        >>> holdout = get_holdout_transitions()
        >>> print(f"Holdout transitions: {len(holdout):,}")
    """
    df = load_transitions(holdout=False, path=path)
    return df[df["year"] >= year].copy()


def get_training_transitions(
    df: Optional[pd.DataFrame] = None,
    *,
    train_years: Optional[List[int]] = None,
    path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter transitions to training years.

    Default training years exclude 2020-2021 (COVID disruption) and 2024+ (holdout).

    Args:
        df: Pre-loaded transitions DataFrame. If None, loads from path.
        train_years: List of years to include. Default: [2015-2019, 2022, 2023].
        path: Override path if loading fresh.

    Returns:
        DataFrame filtered to training years.

    Example:
        >>> train = get_training_transitions()
        >>> print(f"Training transitions: {len(train):,}")
    """
    if df is None:
        df = load_transitions(holdout=False, path=path)

    if train_years is None:
        train_years = DEFAULT_TRAIN_YEARS

    return df[df["year"].isin(train_years)].copy()


# =============================================================================
# Distance Matrix Loading
# =============================================================================

def load_wasserstein_census() -> Tuple[np.ndarray, List[int]]:
    """
    Load Census-level Wasserstein distance matrix.

    This is the primary semantic distance matrix (447x447 Census occupations).

    Returns:
        Tuple of:
            - distances: (447, 447) Wasserstein distance matrix
            - census_codes: List of Census 2010 occupation codes

    Artifacts read:
        .cache/artifacts/v1/mobility/d_wasserstein_census.npz
        Keys: 'distances', 'occupation_codes'

    Example:
        >>> d_wass, codes = load_wasserstein_census()
        >>> print(f"Shape: {d_wass.shape}")
        >>> print(f"Codes: {len(codes)}")
    """
    path = _get_mobility_cache_path("d_wasserstein_census.npz")

    if not path.exists():
        raise FileNotFoundError(
            f"Wasserstein distances not found at {path}. "
            "Run distance computation scripts first."
        )

    data = np.load(path)
    distances = data["distances"]

    # Handle key naming variation (some files use 'occupation_codes', others 'census_codes')
    if "occupation_codes" in data.files:
        codes = [int(c) for c in data["occupation_codes"]]
    elif "census_codes" in data.files:
        codes = [int(c) for c in data["census_codes"]]
    else:
        raise ValueError(f"No occupation codes found in {path}")

    return distances, codes


def load_institutional_census() -> Tuple[np.ndarray, List[int]]:
    """
    Load Census-level institutional distance matrix.

    Institutional distance = Job Zone difference + certification importance.

    Note: The artifact file contains O*NET-level data (923 occupations).
    This function automatically aggregates to Census level (447 occupations)
    using the crosswalk.

    Returns:
        Tuple of:
            - distances: (n_census, n_census) institutional distance matrix
            - census_codes: List of Census 2010 occupation codes

    Artifacts read:
        .cache/artifacts/v1/mobility/d_inst_census.npz
        Keys: 'd_inst_matrix', 'occ_codes' (or variants)

    Example:
        >>> d_inst, codes = load_institutional_census()
        >>> print(f"Shape: {d_inst.shape}")
    """
    path = _get_mobility_cache_path("d_inst_census.npz")

    if not path.exists():
        raise FileNotFoundError(
            f"Institutional distances not found at {path}. "
            "Run institutional distance computation first."
        )

    data = np.load(path, allow_pickle=True)

    # Handle key naming variations across scripts
    if "d_inst_matrix" in data.files:
        distances = data["d_inst_matrix"]
    elif "distances" in data.files:
        distances = data["distances"]
    else:
        raise ValueError(f"No distance matrix found in {path}")

    # Get occupation codes
    if "occ_codes" in data.files:
        raw_codes = list(data["occ_codes"])
    elif "occupation_codes" in data.files:
        raw_codes = list(data["occupation_codes"])
    elif "census_codes" in data.files:
        raw_codes = list(data["census_codes"])
    else:
        raise ValueError(f"No occupation codes found in {path}")

    # Check if codes are O*NET (strings like "11-1011.00") or Census (integers)
    # If O*NET codes, aggregate to Census level
    sample_code = raw_codes[0] if raw_codes else ""
    is_onet = isinstance(sample_code, str) and "-" in str(sample_code)

    if is_onet:
        # Aggregate from O*NET to Census level
        d_census, census_codes = aggregate_institutional_distances(
            distance_matrix=distances,
            onet_codes=raw_codes,
        )
        return d_census, census_codes
    else:
        # Already at Census level
        codes = [int(c) for c in raw_codes]
        return distances, codes


def load_distance_matrix(
    kind: Literal["wasserstein", "institutional", "cosine_onet", "cosine_embed", "euclidean_dwa", "wasserstein_identity"] = "wasserstein",
) -> Tuple[np.ndarray, List[int]]:
    """
    Load a distance matrix by type.

    Provides unified interface for loading different distance metrics.
    All matrices are at Census level (447x447).

    Args:
        kind: Type of distance matrix:
            - "wasserstein": Primary semantic distance (embedding-based)
            - "institutional": Job zone + certification distance
            - "cosine_onet": Cosine distance on O*NET task profiles
            - "cosine_embed": Cosine distance on embedding centroids
            - "euclidean_dwa": Euclidean distance on DWA task profiles
            - "wasserstein_identity": Wasserstein with identity ground metric

    Returns:
        Tuple of (distances array, census codes list)

    Raises:
        FileNotFoundError: If requested matrix not found in cache.
        ValueError: If kind is not recognized.

    Example:
        >>> d_sem, codes = load_distance_matrix("wasserstein")
        >>> d_inst, _ = load_distance_matrix("institutional")
    """
    # Map kind to filename
    filename_map = {
        "wasserstein": "d_wasserstein_census.npz",
        "institutional": "d_inst_census.npz",
        "cosine_onet": "d_cosine_onet_census.npz",
        "cosine_embed": "d_cosine_embed_census.npz",
        "euclidean_dwa": "d_euclidean_dwa_census.npz",
        "wasserstein_identity": "d_wasserstein_identity_census.npz",
    }

    if kind not in filename_map:
        raise ValueError(
            f"Unknown distance kind: {kind}. "
            f"Valid options: {list(filename_map.keys())}"
        )

    # Dispatch to specialized loaders for known types
    if kind == "wasserstein":
        return load_wasserstein_census()
    elif kind == "institutional":
        return load_institutional_census()

    # Generic loading for other types
    path = _get_mobility_cache_path(filename_map[kind])

    if not path.exists():
        raise FileNotFoundError(
            f"Distance matrix not found at {path}. "
            f"Run distance computation for '{kind}' first."
        )

    data = np.load(path, allow_pickle=True)

    # Extract distances (try common key names)
    if "distances" in data.files:
        distances = data["distances"]
    elif "d_matrix" in data.files:
        distances = data["d_matrix"]
    else:
        # Take the first array-like item
        for key in data.files:
            arr = data[key]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                distances = arr
                break
        else:
            raise ValueError(f"No 2D distance matrix found in {path}")

    # Extract codes
    if "census_codes" in data.files:
        codes = [int(c) for c in data["census_codes"]]
    elif "occupation_codes" in data.files:
        codes = [int(c) for c in data["occupation_codes"]]
    else:
        raise ValueError(f"No occupation codes found in {path}")

    return distances, codes


# =============================================================================
# Distance Aggregation
# =============================================================================

def aggregate_institutional_distances(
    *,
    distance_matrix: np.ndarray,
    onet_codes: List[str],
    census_codes: Optional[List[int]] = None,
    aggregation: str = "mean",
) -> Tuple[np.ndarray, List[int]]:
    """
    Aggregate O*NET-level distances to Census level.

    Thin wrapper around the canonical aggregate_distances_to_census function
    from task_space.mobility.census_crosswalk.

    Args:
        distance_matrix: (n_onet, n_onet) O*NET-level distance matrix
        onet_codes: List of O*NET-SOC codes (row/column labels)
        census_codes: Optional list of target Census codes. If None, uses all
            codes from the crosswalk.
        aggregation: Aggregation method ("mean" or "min")

    Returns:
        Tuple of:
            - Aggregated (n_census, n_census) distance matrix
            - Census 2010 codes (row/column labels)

    Example:
        >>> from task_space.mobility.institutional import build_institutional_distance_matrix
        >>> inst_result = build_institutional_distance_matrix()
        >>> d_census, codes = aggregate_institutional_distances(
        ...     distance_matrix=inst_result.matrix,
        ...     onet_codes=inst_result.occupations,
        ... )
    """
    from task_space.mobility.census_crosswalk import (
        load_census_onet_crosswalk,
        aggregate_distances_to_census,
    )

    # Load crosswalk
    crosswalk = load_census_onet_crosswalk()

    # Use canonical aggregation function
    d_census, result_codes = aggregate_distances_to_census(
        onet_distance_matrix=distance_matrix,
        onet_codes=onet_codes,
        crosswalk=crosswalk,
        aggregation=aggregation,
    )

    return d_census, result_codes


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_TRAIN_YEARS",
    "DEFAULT_HOLDOUT_YEAR",
    # Transition loading
    "load_transitions",
    "get_holdout_transitions",
    "get_training_transitions",
    # Distance matrix loading
    "load_wasserstein_census",
    "load_institutional_census",
    "load_distance_matrix",
    # Aggregation
    "aggregate_institutional_distances",
]
