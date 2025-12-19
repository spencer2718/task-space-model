"""
Build CZ × occupation employment matrix from IPUMS Census microdata.

Uses Dorn crosswalks to map:
- CNTYGP98/PUMA → Commuting Zone
- OCC → occ1990dd (harmonized occupation codes)

Output: Employment counts by CZ × occ1990dd for each Census year.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from ipumspy import readers


DORN_CROSSWALK_PATH = Path("data/external/dorn_replication/dorn_extracted") / \
    "Autor-Dorn-LowSkillServices-FileArchive.zip Folder/crosswalks"


def load_puma_to_cz(year: int) -> pd.DataFrame:
    """
    Load PUMA/county group to CZ crosswalk for given year.

    The crosswalks have afactor (allocation factor) for cases where
    a PUMA spans multiple CZs. We use afactor to distribute employment.

    Args:
        year: Census year (1980, 1990, 2000)

    Returns:
        DataFrame with columns: puma, czone, afactor
    """
    if year == 1980:
        path = DORN_CROSSWALK_PATH / "cw_ctygrp1980_czone.dta"
        df = pd.read_stata(path)
        df = df.rename(columns={"ctygrp1980": "puma"})
    elif year == 1990:
        path = DORN_CROSSWALK_PATH / "cw_puma1990_czone.dta"
        df = pd.read_stata(path)
        df = df.rename(columns={"puma1990": "puma"})
    elif year == 2000:
        path = DORN_CROSSWALK_PATH / "cw_puma2000_czone.dta"
        df = pd.read_stata(path)
        df = df.rename(columns={"puma2000": "puma"})
    else:
        raise ValueError(f"Unsupported year: {year}")

    return df[["puma", "czone", "afactor"]]


def load_occ_to_occ1990dd(year: int, occ_var: str) -> pd.DataFrame:
    """
    Load occupation code to occ1990dd crosswalk for given year/variable.

    Args:
        year: Census year (1980, 1990, 2000)
        occ_var: IPUMS occupation variable (OCC1950, OCC1990, OCC)

    Returns:
        DataFrame with columns: occ_orig, occ1990dd
    """
    # Map IPUMS variable to crosswalk file
    if occ_var == "OCC1950":
        path = DORN_CROSSWALK_PATH / "occ1950_occ1990dd.dta"
    elif occ_var == "OCC1990":
        path = DORN_CROSSWALK_PATH / "occ1990_occ1990dd.dta"
    elif occ_var == "OCC":  # 2000 native codes
        path = DORN_CROSSWALK_PATH / "occ2000_occ1990dd.dta"
    else:
        raise ValueError(f"Unsupported occ variable: {occ_var}")

    df = pd.read_stata(path)
    # All crosswalks use "occ" column for original codes
    df = df.rename(columns={"occ": "occ_orig"})

    return df[["occ_orig", "occ1990dd"]]


def read_ipums_extract(
    data_path: Path,
    ddi_path: Path,
) -> pd.DataFrame:
    """
    Read IPUMS extract using DDI codebook.

    Args:
        data_path: Path to .dat.gz file
        ddi_path: Path to .xml codebook

    Returns:
        DataFrame with IPUMS data
    """
    ddi = readers.read_ipums_ddi(ddi_path)
    df = readers.read_microdata(ddi, data_path)
    return df


def process_census_year(
    year: int,
    data_dir: Path,
    extract_num: int,
) -> pd.DataFrame:
    """
    Process a single Census year's IPUMS extract.

    Args:
        year: Census year
        data_dir: Directory containing IPUMS extracts
        extract_num: Extract number (1, 2, or 3)

    Returns:
        DataFrame with columns: czone, occ1990dd, employment
    """
    # Read IPUMS data
    data_path = data_dir / f"usa_{extract_num:05d}.dat.gz"
    ddi_path = data_dir / f"usa_{extract_num:05d}.xml"

    print(f"  Reading {data_path.name}...")
    df = read_ipums_extract(data_path, ddi_path)
    print(f"  Loaded {len(df):,} records")

    # Filter to employed workers (EMPSTAT == 1 means employed)
    df = df[df["EMPSTAT"] == 1].copy()
    print(f"  After employment filter: {len(df):,}")

    # Filter to working-age (16-64)
    df = df[(df["AGE"] >= 16) & (df["AGE"] <= 64)].copy()
    print(f"  After age filter: {len(df):,}")

    # Get appropriate column names based on year
    if year == 1980:
        geo_col = "CNTYGP98"
        occ_col = "OCC1950"  # 1980 uses OCC1950 harmonized codes
    elif year == 1990:
        geo_col = "PUMA"
        occ_col = "OCC1990"
    else:  # 2000
        geo_col = "PUMA"
        occ_col = "OCC"  # 2000 uses native OCC

    # Load crosswalks
    puma_to_cz = load_puma_to_cz(year)
    occ_to_occ1990dd = load_occ_to_occ1990dd(year, occ_col)

    # Create state-prefixed geo identifier for merging with crosswalk
    # Crosswalk uses format: statefip * 10000 + puma (for 1990/2000)
    # or statefip * 1000 + ctygrp (for 1980)
    df["statefip"] = df["STATEFIP"]
    if year == 1980:
        # 1980 county groups are already state-prefixed in a different way
        df["geo"] = df["STATEFIP"] * 1000 + df[geo_col]
    else:
        # 1990/2000 PUMA needs state prefix
        df["geo"] = df["STATEFIP"] * 10000 + df[geo_col]

    # Merge PUMA/county group to CZ
    # Use afactor to handle split geographies
    df = df.merge(
        puma_to_cz[["puma", "czone", "afactor"]],
        left_on="geo",
        right_on="puma",
        how="left"
    )

    matched_cz = df["czone"].notna().sum()
    print(f"  CZ matches: {matched_cz:,} / {len(df):,} ({100*matched_cz/len(df):.1f}%)")

    # Merge occupation to occ1990dd
    df["occ_orig"] = df[occ_col]
    df = df.merge(
        occ_to_occ1990dd,
        on="occ_orig",
        how="left"
    )

    matched_occ = df["occ1990dd"].notna().sum()
    print(f"  Occupation matches: {matched_occ:,} / {len(df):,} ({100*matched_occ/len(df):.1f}%)")

    # Filter to valid matches
    df = df[df["czone"].notna() & df["occ1990dd"].notna()].copy()
    print(f"  Final valid records: {len(df):,}")

    # Weight person weight by afactor (for split geographies)
    df["weighted_emp"] = df["PERWT"] * df["afactor"]

    # Aggregate employment by CZ × occupation
    employment = df.groupby(["czone", "occ1990dd"])["weighted_emp"].sum().reset_index()
    employment = employment.rename(columns={"weighted_emp": "employment"})

    print(f"  Unique CZs: {employment['czone'].nunique()}")
    print(f"  Unique occupations: {employment['occ1990dd'].nunique()}")
    print(f"  Total CZ×occ cells: {len(employment)}")

    return employment


def build_all_years(
    data_dir: Path = Path("data/external/ipums/census"),
    output_dir: Path = Path("data/processed/cz_employment"),
) -> dict[int, pd.DataFrame]:
    """
    Build CZ × occupation employment matrices for all Census years.

    Returns:
        Dict mapping year to employment DataFrame
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    year_extract_map = {
        1980: 1,
        1990: 2,
        2000: 3,
    }

    for year, extract_num in year_extract_map.items():
        print(f"\nProcessing {year}...")
        df = process_census_year(year, data_dir, extract_num)

        # Save to CSV
        output_path = output_dir / f"cz_occ_employment_{year}.csv"
        df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

        results[year] = df

    return results


def compute_cz_employment_shares(
    employment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute within-CZ employment shares for each occupation.

    Args:
        employment_df: DataFrame with czone, occ1990dd, employment

    Returns:
        DataFrame with added emp_share column
    """
    df = employment_df.copy()

    # Total employment by CZ
    cz_totals = df.groupby("czone")["employment"].sum().reset_index()
    cz_totals = cz_totals.rename(columns={"employment": "cz_total_emp"})

    # Merge and compute shares
    df = df.merge(cz_totals, on="czone")
    df["emp_share"] = df["employment"] / df["cz_total_emp"]

    return df


if __name__ == "__main__":
    print("Building CZ × occupation employment matrices from IPUMS data")
    print("=" * 60)

    results = build_all_years()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for year, df in results.items():
        print(f"\n{year}:")
        print(f"  CZs: {df['czone'].nunique()}")
        print(f"  Occupations: {df['occ1990dd'].nunique()}")
        print(f"  Total employment: {df['employment'].sum():,.0f}")
