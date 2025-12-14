"""
Occupation code crosswalks between O*NET-SOC and OES/SOC.

Handles mapping between O*NET-SOC codes and OES/SOC codes.
"""

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def onet_to_soc(onet_code: str) -> str:
    """
    Convert O*NET-SOC code to 6-digit SOC code.

    O*NET-SOC format: XX-XXXX.XX (e.g., '15-1252.00')
    SOC format: XX-XXXX (e.g., '15-1252')

    Args:
        onet_code: O*NET-SOC code with .XX suffix

    Returns:
        6-digit SOC code (first 7 characters)
    """
    # Handle potential whitespace
    onet_code = onet_code.strip()

    # Extract first 7 characters (XX-XXXX)
    if len(onet_code) >= 7:
        return onet_code[:7]
    else:
        return onet_code


def soc_to_onet_pattern(soc_code: str) -> str:
    """
    Convert SOC code to regex pattern matching O*NET codes.

    Args:
        soc_code: 6-digit SOC code (e.g., '15-1252')

    Returns:
        Regex pattern matching O*NET codes (e.g., '15-1252\\.\\d+')
    """
    return re.escape(soc_code) + r"\.\d+"


@dataclass
class OnetOesCrosswalk:
    """
    Crosswalk between O*NET occupations and OES occupations.

    Attributes:
        crosswalk_df: DataFrame with columns:
            - onet_code: Original O*NET-SOC code
            - soc_code: 6-digit SOC code
            - oes_matched: Boolean, True if SOC exists in OES data
        n_onet: Number of O*NET occupations
        n_soc: Number of unique SOC codes
        n_matched: Number of SOC codes found in OES
        coverage: Fraction of O*NET occupations with OES match
        aggregation_map: Dict mapping SOC code to list of O*NET codes (for many-to-one)
    """
    crosswalk_df: pd.DataFrame
    n_onet: int
    n_soc: int
    n_matched: int
    coverage: float
    aggregation_map: dict[str, list[str]]


def build_onet_oes_crosswalk(
    onet_codes: list[str],
    oes_codes: Optional[list[str]] = None,
) -> OnetOesCrosswalk:
    """
    Build crosswalk between O*NET occupations and OES occupations.

    Args:
        onet_codes: List of O*NET-SOC codes (e.g., from occupation_measures.occupation_codes)
        oes_codes: Optional list of OES/SOC codes to match against.
                   If None, all O*NET codes are mapped but oes_matched is set to True
                   (assumes all SOC codes exist in OES).

    Returns:
        OnetOesCrosswalk with mapping information.

    Note: Multiple O*NET codes may map to one SOC (many-to-one).
    """
    # Build crosswalk dataframe
    crosswalk_data = []
    for onet_code in onet_codes:
        soc_code = onet_to_soc(onet_code)
        crosswalk_data.append({
            "onet_code": onet_code,
            "soc_code": soc_code,
        })

    crosswalk_df = pd.DataFrame(crosswalk_data)

    # Check OES matching if codes provided
    if oes_codes is not None:
        oes_set = set(oes_codes)
        crosswalk_df["oes_matched"] = crosswalk_df["soc_code"].isin(oes_set)
    else:
        # Assume all match if no OES codes provided
        crosswalk_df["oes_matched"] = True

    # Build aggregation map (SOC -> list of O*NET codes)
    aggregation_map = {}
    for soc_code, group in crosswalk_df.groupby("soc_code"):
        aggregation_map[soc_code] = group["onet_code"].tolist()

    # Compute statistics
    n_onet = len(onet_codes)
    n_soc = crosswalk_df["soc_code"].nunique()
    n_matched = crosswalk_df[crosswalk_df["oes_matched"]]["soc_code"].nunique()
    coverage = crosswalk_df["oes_matched"].mean()

    return OnetOesCrosswalk(
        crosswalk_df=crosswalk_df,
        n_onet=n_onet,
        n_soc=n_soc,
        n_matched=n_matched,
        coverage=coverage,
        aggregation_map=aggregation_map,
    )


def aggregate_occupation_measures(
    occupation_matrix: np.ndarray,
    occupation_codes: list[str],
    crosswalk: OnetOesCrosswalk,
) -> tuple[np.ndarray, list[str]]:
    """
    Aggregate O*NET occupation measures to SOC level.

    When multiple O*NET occupations map to one SOC code, average their
    probability distributions.

    Args:
        occupation_matrix: (n_onet, n_activities) matrix of rho_j distributions
        occupation_codes: O*NET-SOC codes (row labels)
        crosswalk: OnetOesCrosswalk from build_onet_oes_crosswalk()

    Returns:
        Tuple of:
            - Aggregated matrix (n_soc, n_activities)
            - SOC codes (row labels)
    """
    # Build mapping from O*NET code to row index
    onet_to_idx = {code: i for i, code in enumerate(occupation_codes)}

    # Aggregate by SOC code
    soc_codes = []
    aggregated_rows = []

    for soc_code, onet_list in crosswalk.aggregation_map.items():
        # Get indices for O*NET codes in this SOC
        indices = [onet_to_idx[onet] for onet in onet_list if onet in onet_to_idx]

        if not indices:
            continue

        # Average the probability distributions
        avg_row = occupation_matrix[indices].mean(axis=0)
        # Re-normalize to sum to 1
        avg_row = avg_row / avg_row.sum()

        soc_codes.append(soc_code)
        aggregated_rows.append(avg_row)

    aggregated_matrix = np.array(aggregated_rows)

    return aggregated_matrix, soc_codes
