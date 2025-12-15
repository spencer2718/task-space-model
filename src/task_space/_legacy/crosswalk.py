"""
Occupation code crosswalks and OES data loading.

Handles mapping between O*NET-SOC codes and OES/SOC codes,
and loading OES wage data for wage comovement computation.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd


# Default paths
DEFAULT_OES_PATH = Path(__file__).parent.parent.parent / "data" / "external" / "oes"


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


def load_oes_year(
    year: int,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load OES data for a single year.

    Expects file at: data_dir/oesm{YY}nat/national_M{YYYY}_dl.xlsx
    or similar naming convention.

    Args:
        year: Year to load (e.g., 2019)
        data_dir: Directory containing OES data. Defaults to data/external/oes.

    Returns:
        DataFrame with columns:
            OCC_CODE: 6-digit SOC code
            OCC_TITLE: Occupation name
            TOT_EMP: Total employment
            A_MEAN: Annual mean wage
            A_MEDIAN: Annual median wage
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_OES_PATH

    # Try different file naming conventions
    yy = str(year)[-2:]
    possible_paths = [
        data_dir / f"national_M{year}_dl.xlsx",
        data_dir / f"oesm{yy}nat" / f"national_M{year}_dl.xlsx",
        data_dir / f"national_M{year}_dl.xls",
        data_dir / f"oesm{yy}nat" / f"national_M{year}_dl.xls",
        # Older format
        data_dir / f"nat{yy}nat.xlsx",
        data_dir / f"oesm{yy}nat" / f"nat{yy}nat.xlsx",
    ]

    df = None
    for path in possible_paths:
        if path.exists():
            df = pd.read_excel(path)
            break

    if df is None:
        raise FileNotFoundError(
            f"OES data for {year} not found. Tried: {possible_paths[:2]}...\n"
            f"Download from https://www.bls.gov/oes/tables.htm"
        )

    # Standardize column names (OES uses different names across years)
    column_mapping = {
        "OCC_CODE": "OCC_CODE",
        "occ_code": "OCC_CODE",
        "OCC_TITLE": "OCC_TITLE",
        "occ_title": "OCC_TITLE",
        "TOT_EMP": "TOT_EMP",
        "tot_emp": "TOT_EMP",
        "A_MEAN": "A_MEAN",
        "a_mean": "A_MEAN",
        "A_MEDIAN": "A_MEDIAN",
        "a_median": "A_MEDIAN",
    }

    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Select and clean columns
    required_cols = ["OCC_CODE", "A_MEAN"]
    optional_cols = ["OCC_TITLE", "TOT_EMP", "A_MEDIAN"]

    available_cols = [c for c in required_cols + optional_cols if c in df.columns]
    df = df[available_cols].copy()

    # Filter to detailed occupations (exclude aggregates)
    # Detailed SOC codes have format XX-XXXX where X is digit
    # Exclude codes ending in 0000 (major group) or 0 (minor group/broad)
    df = df[df["OCC_CODE"].str.match(r"^\d{2}-\d{4}$", na=False)]
    # Exclude major groups (XX-0000) and minor groups (XX-X000)
    df = df[~df["OCC_CODE"].str.endswith("0000")]
    df = df[~df["OCC_CODE"].str.match(r"^\d{2}-\d000$", na=False)]

    # Convert wage columns to numeric, coercing errors
    for col in ["A_MEAN", "A_MEDIAN", "TOT_EMP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing wages
    df = df.dropna(subset=["A_MEAN"])

    return df


def load_oes_panel(
    years: list[int],
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load multi-year OES panel.

    Args:
        years: List of years to load (e.g., [2015, 2016, ..., 2023])
        data_dir: Directory containing OES data.

    Returns:
        DataFrame with columns:
            OCC_CODE: 6-digit SOC code
            year: Survey year
            A_MEAN: Annual mean wage
            A_MEDIAN: Annual median wage (if available)
            TOT_EMP: Total employment (if available)
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_OES_PATH

    panels = []
    for year in years:
        try:
            df = load_oes_year(year, data_dir)
            df["year"] = year
            panels.append(df)
        except FileNotFoundError as e:
            warnings.warn(f"Skipping {year}: {e}")
            continue

    if not panels:
        raise FileNotFoundError(f"No OES data found for years {years}")

    panel = pd.concat(panels, ignore_index=True)

    # Ensure consistent columns
    cols = ["OCC_CODE", "year", "A_MEAN"]
    for optional in ["A_MEDIAN", "TOT_EMP", "OCC_TITLE"]:
        if optional in panel.columns:
            cols.append(optional)

    return panel[cols]


@dataclass
class WageComovement:
    """
    Occupation-pair wage comovement result.

    Attributes:
        comovement_matrix: Correlation matrix of log wage changes
        occupation_codes: SOC codes (row/column labels)
        years: Years used in computation
        n_years: Number of years
        n_occupations: Number of occupations with sufficient data
        coverage: Fraction of possible pairs with valid correlation
    """
    comovement_matrix: np.ndarray
    occupation_codes: list[str]
    years: list[int]
    n_years: int
    n_occupations: int
    coverage: float


def compute_wage_comovement(
    oes_panel: pd.DataFrame,
    min_years: int = 5,
) -> WageComovement:
    """
    Compute pairwise correlations of log wage changes.

    Args:
        oes_panel: Multi-year OES panel from load_oes_panel()
        min_years: Minimum years of data required per occupation

    Returns:
        WageComovement with correlation matrix.
    """
    # Compute log wages
    panel = oes_panel.copy()
    panel["log_wage"] = np.log(panel["A_MEAN"])

    # Pivot to wide format: rows = years, columns = occupations
    wage_wide = panel.pivot_table(
        index="year",
        columns="OCC_CODE",
        values="log_wage",
        aggfunc="first",
    )

    # Compute year-over-year changes
    # Drop first row (all NaN after diff), but keep rows with partial data
    wage_changes = wage_wide.diff().iloc[1:]

    # Filter occupations with sufficient data
    valid_counts = wage_changes.notna().sum()
    valid_occs = valid_counts[valid_counts >= min_years].index.tolist()

    if len(valid_occs) < 2:
        raise ValueError(
            f"Insufficient data: only {len(valid_occs)} occupations have >= {min_years} years"
        )

    wage_changes = wage_changes[valid_occs]

    # Compute pairwise correlations
    # Use pandas corr() which handles missing values pairwise
    corr_matrix = wage_changes.corr()

    # Convert to numpy
    comovement_matrix = corr_matrix.values
    occupation_codes = corr_matrix.columns.tolist()

    # Compute coverage (fraction of valid correlations)
    n_occ = len(occupation_codes)
    n_pairs = n_occ * (n_occ - 1) // 2
    # Upper triangle excluding diagonal
    triu_indices = np.triu_indices(n_occ, k=1)
    valid_corrs = ~np.isnan(comovement_matrix[triu_indices])
    coverage = valid_corrs.sum() / n_pairs if n_pairs > 0 else 0.0

    years = sorted(oes_panel["year"].unique().tolist())

    return WageComovement(
        comovement_matrix=comovement_matrix,
        occupation_codes=occupation_codes,
        years=years,
        n_years=len(years),
        n_occupations=len(occupation_codes),
        coverage=coverage,
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
        occupation_matrix: (n_onet, n_activities) matrix of ρ_j distributions
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
