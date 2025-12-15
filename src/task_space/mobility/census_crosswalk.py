"""
Census 2010 to O*NET-SOC crosswalk for CPS mobility analysis.

Maps between Census 2010 occupation codes (used in CPS) and O*NET-SOC codes
(used for task measures). Handles many-to-one aggregation when multiple
O*NET occupations map to a single Census code.

Mapping path: Census 2010 (OCC2010) → 6-digit SOC → O*NET-SOC 2019

References:
- BLS SOC crosswalk: https://www.bls.gov/soc/
- Census occupation codes: https://www.census.gov/topics/employment/industry-occupation.html
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


@dataclass
class CensusCrosswalkResult:
    """
    Result from building Census↔O*NET crosswalk.

    Attributes:
        crosswalk_df: DataFrame with mapping (onet_soc, soc_6digit, census_2010, matched)
        onet_to_census: Dict mapping O*NET-SOC → Census 2010 code
        census_to_onet: Dict mapping Census 2010 code → List of O*NET-SOC codes
        n_onet: Number of O*NET occupations
        n_census: Number of unique Census codes
        n_matched: Number of O*NET codes with Census match
        coverage: Fraction of O*NET codes with Census match
        mean_onet_per_census: Average O*NET codes per Census code
        assumptions: Modeling assumptions and caveats
    """
    crosswalk_df: pd.DataFrame
    onet_to_census: Dict[str, int]
    census_to_onet: Dict[int, List[str]]
    n_onet: int
    n_census: int
    n_matched: int
    coverage: float
    mean_onet_per_census: float
    assumptions: List[str] = field(default_factory=lambda: [
        "Many-to-one: Multiple O*NET codes may map to one Census code",
        "Distance aggregation: When aggregating, use unweighted mean",
        "Census 2010 basis: CPS OCC2010 harmonized codes",
        "Unmatched O*NET codes excluded from analysis (primarily postsecondary teachers)",
    ])


def load_census_onet_crosswalk(
    crosswalk_path: Optional[Path] = None,
) -> CensusCrosswalkResult:
    """
    Load pre-built Census 2010 to O*NET crosswalk.

    Args:
        crosswalk_path: Path to crosswalk CSV. If None, uses cached version.

    Returns:
        CensusCrosswalkResult with mappings and metadata.

    Example:
        >>> xwalk = load_census_onet_crosswalk()
        >>> xwalk.onet_to_census['15-1252.00']
        1020
    """
    if crosswalk_path is None:
        # Try cached location first
        cached = Path(".cache/artifacts/v1/mobility/onet_to_census_improved.csv")
        if cached.exists():
            crosswalk_path = cached
        else:
            # Fallback to killshot location
            crosswalk_path = Path("killshot/cps_mobility/data/onet_to_census_improved.csv")

    if not crosswalk_path.exists():
        raise FileNotFoundError(
            f"Crosswalk not found at {crosswalk_path}. "
            "Run data preparation scripts first."
        )

    df = pd.read_csv(crosswalk_path)

    # Build mappings
    onet_to_census = {}
    census_to_onet = {}

    for _, row in df.iterrows():
        onet_code = row["onet_soc"]
        census_code = int(row["census_2010"]) if pd.notna(row["census_2010"]) else None

        if census_code is not None and row["matched"]:
            onet_to_census[onet_code] = census_code

            if census_code not in census_to_onet:
                census_to_onet[census_code] = []
            census_to_onet[census_code].append(onet_code)

    # Compute statistics
    n_onet = len(df)
    n_matched = df["matched"].sum()
    n_census = len(census_to_onet)
    coverage = n_matched / n_onet if n_onet > 0 else 0.0
    mean_onet_per_census = n_matched / n_census if n_census > 0 else 0.0

    return CensusCrosswalkResult(
        crosswalk_df=df,
        onet_to_census=onet_to_census,
        census_to_onet=census_to_onet,
        n_onet=n_onet,
        n_census=n_census,
        n_matched=int(n_matched),
        coverage=coverage,
        mean_onet_per_census=mean_onet_per_census,
    )


def aggregate_distances_to_census(
    onet_distance_matrix: np.ndarray,
    onet_codes: List[str],
    crosswalk: CensusCrosswalkResult,
    aggregation: str = "mean",
) -> tuple[np.ndarray, List[int]]:
    """
    Aggregate O*NET-level distance matrix to Census 2010 level.

    When multiple O*NET codes map to one Census code, aggregate their
    distances using the specified method.

    Args:
        onet_distance_matrix: (n_onet, n_onet) distance matrix
        onet_codes: O*NET-SOC codes (row/column labels)
        crosswalk: CensusCrosswalkResult from load_census_onet_crosswalk()
        aggregation: "mean" (default) or "min"

    Returns:
        Tuple of:
            - Aggregated (n_census, n_census) distance matrix
            - Census 2010 codes (row/column labels)

    Example:
        >>> d_census, census_codes = aggregate_distances_to_census(d_onet, onet_codes, xwalk)
        >>> d_census.shape
        (447, 447)
    """
    # Build O*NET code to index mapping
    onet_to_idx = {code: i for i, code in enumerate(onet_codes)}

    # Get sorted Census codes
    census_codes = sorted(crosswalk.census_to_onet.keys())
    n_census = len(census_codes)
    census_to_idx = {code: i for i, code in enumerate(census_codes)}

    # Initialize aggregated matrix
    d_census = np.zeros((n_census, n_census))
    count_matrix = np.zeros((n_census, n_census))

    # Aggregate
    for ci, census_i in enumerate(census_codes):
        onet_list_i = crosswalk.census_to_onet[census_i]
        valid_idx_i = [onet_to_idx[o] for o in onet_list_i if o in onet_to_idx]

        for cj, census_j in enumerate(census_codes):
            onet_list_j = crosswalk.census_to_onet[census_j]
            valid_idx_j = [onet_to_idx[o] for o in onet_list_j if o in onet_to_idx]

            if not valid_idx_i or not valid_idx_j:
                continue

            # Get all pairwise distances
            distances = []
            for oi in valid_idx_i:
                for oj in valid_idx_j:
                    distances.append(onet_distance_matrix[oi, oj])

            if distances:
                if aggregation == "mean":
                    d_census[ci, cj] = np.mean(distances)
                elif aggregation == "min":
                    d_census[ci, cj] = np.min(distances)
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")
                count_matrix[ci, cj] = len(distances)

    return d_census, census_codes


def get_census_distance(
    census_i: int,
    census_j: int,
    distance_matrix: np.ndarray,
    census_codes: List[int],
) -> float:
    """
    Look up distance between two Census occupation codes.

    Args:
        census_i: Origin Census 2010 code
        census_j: Destination Census 2010 code
        distance_matrix: Pre-aggregated Census-level distance matrix
        census_codes: Census codes (row/column labels)

    Returns:
        Distance d(i, j)

    Raises:
        ValueError: If Census code not found
    """
    try:
        idx_i = census_codes.index(census_i)
        idx_j = census_codes.index(census_j)
    except ValueError as e:
        raise ValueError(f"Census code not found: {e}")

    return distance_matrix[idx_i, idx_j]
