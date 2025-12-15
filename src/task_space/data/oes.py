"""
OES (Occupational Employment and Wage Statistics) data loading.

Loads BLS OES data for wage comovement computation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd


# Default path to OES data
DEFAULT_OES_PATH = Path(__file__).parent.parent.parent.parent / "data" / "external" / "oes"


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


def load_oes_employment(
    year: int,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load OES employment data for a single year.

    Convenience function that loads OES data and returns employment-focused columns.

    Args:
        year: Year to load (e.g., 2019, 2024)
        data_dir: Directory containing OES data.

    Returns:
        DataFrame with columns:
            soc_code: 6-digit SOC code (standardized name)
            occupation_title: Occupation name
            tot_emp: Total employment
            year: Survey year

    Raises:
        ValueError: If TOT_EMP not available for this year
    """
    df = load_oes_year(year, data_dir)

    if "TOT_EMP" not in df.columns:
        raise ValueError(f"Employment data (TOT_EMP) not available for {year}")

    # Standardize column names for consistency with other loaders
    result = pd.DataFrame({
        "soc_code": df["OCC_CODE"],
        "occupation_title": df.get("OCC_TITLE", ""),
        "tot_emp": df["TOT_EMP"],
        "year": year,
    })

    # Drop rows with missing employment
    result = result.dropna(subset=["tot_emp"])
    result["tot_emp"] = result["tot_emp"].astype(int)

    return result


def compute_wage_comovement(
    oes_panel: pd.DataFrame,
    min_years: int = 4,
) -> WageComovement:
    """
    Compute pairwise correlations of log wage changes.

    Args:
        oes_panel: Multi-year OES panel from load_oes_panel()
        min_years: Minimum years of CHANGES required per occupation.
            With N years of data, you get N-1 years of changes.
            Default is 4 (requires 5 years of raw data).

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
