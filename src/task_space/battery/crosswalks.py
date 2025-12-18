"""
Historical occupation crosswalks for retrospective battery.

Maps occ1990dd (Dorn harmonized codes) to O*NET-SOC via Census chain:
    occ1990dd → Census 1990 → OCC2010 → O*NET-SOC

Key insight: occ1990dd codes are essentially Census 1990 codes (97.6% match).
We leverage existing IPUMS and O*NET crosswalks to complete the chain.

Coverage: 91.9% employment-weighted (1980 employment).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Set
import pandas as pd
import numpy as np


@dataclass
class Occ1990ddCrosswalkResult:
    """
    Result from building occ1990dd → O*NET-SOC crosswalk.

    Attributes:
        crosswalk_df: DataFrame with mapping columns
        n_occ1990dd: Total occ1990dd codes
        n_mapped: Codes successfully mapped to O*NET
        n_unmapped: Codes that failed to map
        unweighted_coverage: Fraction of codes mapped
        weighted_coverage: Employment-weighted coverage (1980)
        unmapped_employment_share: Employment share of unmapped codes
        confidence_distribution: Count by confidence tier
    """
    crosswalk_df: pd.DataFrame
    n_occ1990dd: int
    n_mapped: int
    n_unmapped: int
    unweighted_coverage: float
    weighted_coverage: float
    unmapped_employment_share: float
    confidence_distribution: Dict[str, int] = field(default_factory=dict)


def build_occ1990dd_to_onet_crosswalk(
    ipums_crosswalk_path: Path,
    onet_census_crosswalk_path: Path,
    dorn_employment_path: Optional[Path] = None,
    dorn_task_path: Optional[Path] = None,
) -> Occ1990ddCrosswalkResult:
    """
    Build occ1990dd → O*NET-SOC crosswalk via Census chain.

    Chain: occ1990dd → Census 1990 → OCC2010 → O*NET-SOC

    The occ1990dd codes are Dorn's harmonized Census occupation codes.
    They correspond almost exactly to Census 1990 codes (97.6% identity match).
    We use IPUMS to map Census 1990 → OCC2010, then existing crosswalk
    for OCC2010 → O*NET-SOC.

    Args:
        ipums_crosswalk_path: Path to IPUMS Census 1990 → OCC2010 crosswalk
        onet_census_crosswalk_path: Path to O*NET → Census 2010 crosswalk
        dorn_employment_path: Optional path to Dorn employment data (for weights)
        dorn_task_path: Optional path to Dorn ALM task scores (for occ1990dd list)

    Returns:
        Occ1990ddCrosswalkResult with crosswalk and coverage statistics.

    Output schema:
        - occ1990dd: Dorn harmonized code
        - onet_soc: O*NET-SOC code (e.g., "11-1011.00")
        - weight: probability weight for many-to-many (sums to 1 per occ1990dd)
        - confidence_tier: "high" / "medium" / "low"
        - emp_1980: 1980 employment share (if available)
    """
    # Load IPUMS crosswalk: Census 1990 → OCC2010
    ipums = pd.read_excel(ipums_crosswalk_path)
    ipums_clean = ipums[ipums['1990 Census code'].astype(str).str.match(r'^\d+$', na=False)].copy()
    ipums_clean['census_1990'] = ipums_clean['1990 Census code'].astype(int)
    ipums_clean['occ2010'] = ipums_clean['OCC2010'].astype(int)

    # Build Census 1990 → OCC2010 mapping (one-to-many possible)
    c1990_to_occ2010: Dict[int, Set[int]] = {}
    for _, row in ipums_clean.iterrows():
        c1990 = row['census_1990']
        occ2010 = row['occ2010']
        if c1990 not in c1990_to_occ2010:
            c1990_to_occ2010[c1990] = set()
        c1990_to_occ2010[c1990].add(occ2010)

    # Load O*NET → Census 2010 crosswalk
    onet_census = pd.read_csv(onet_census_crosswalk_path)
    occ2010_to_onet: Dict[int, List[str]] = {}
    for _, row in onet_census[onet_census['matched']].iterrows():
        occ2010 = int(row['census_2010'])
        onet_soc = row['onet_soc']
        if occ2010 not in occ2010_to_onet:
            occ2010_to_onet[occ2010] = []
        occ2010_to_onet[occ2010].append(onet_soc)

    # Load Dorn employment data if available
    emp_data = {}
    if dorn_employment_path and dorn_employment_path.exists():
        emp_df = pd.read_stata(dorn_employment_path)
        for _, row in emp_df.iterrows():
            occ1990dd = int(row['occ1990dd'])
            emp_1980 = row['sh_empl1980'] if pd.notna(row['sh_empl1980']) else 0
            emp_data[occ1990dd] = emp_1980

    # Get list of occ1990dd codes from Dorn task file or employment file
    occ1990dd_codes = set()
    if dorn_task_path and dorn_task_path.exists():
        task_df = pd.read_stata(dorn_task_path)
        occ1990dd_codes = set(task_df['occ1990dd'].astype(int))
    elif emp_data:
        occ1990dd_codes = set(emp_data.keys())
    else:
        # Fallback: use all Census 1990 codes from IPUMS
        occ1990dd_codes = set(c1990_to_occ2010.keys())

    # Build crosswalk
    rows = []
    for occ1990dd in sorted(occ1990dd_codes):
        emp_1980 = emp_data.get(occ1990dd, 0)

        # Step 1: occ1990dd → Census 1990 (identity)
        # Step 2: Census 1990 → OCC2010
        occ2010_set = c1990_to_occ2010.get(occ1990dd, set())

        # Step 3: OCC2010 → O*NET-SOC
        onet_codes = []
        for occ2010 in occ2010_set:
            if occ2010 in occ2010_to_onet:
                onet_codes.extend(occ2010_to_onet[occ2010])

        # Remove duplicates and assign weights
        onet_codes = list(set(onet_codes))

        if len(onet_codes) == 0:
            # No mapping found
            rows.append({
                'occ1990dd': occ1990dd,
                'onet_soc': None,
                'weight': 0.0,
                'confidence_tier': 'unmapped',
                'emp_1980': emp_1980,
            })
        elif len(onet_codes) == 1:
            # Clean one-to-one mapping
            rows.append({
                'occ1990dd': occ1990dd,
                'onet_soc': onet_codes[0],
                'weight': 1.0,
                'confidence_tier': 'high',
                'emp_1980': emp_1980,
            })
        else:
            # Many-to-many: equal weights
            weight = 1.0 / len(onet_codes)
            tier = 'medium' if len(onet_codes) <= 5 else 'low'
            for onet_soc in onet_codes:
                rows.append({
                    'occ1990dd': occ1990dd,
                    'onet_soc': onet_soc,
                    'weight': weight,
                    'confidence_tier': tier,
                    'emp_1980': emp_1980,
                })

    crosswalk_df = pd.DataFrame(rows)

    # Compute coverage statistics
    mapped_codes = crosswalk_df[crosswalk_df['onet_soc'].notna()]['occ1990dd'].unique()
    unmapped_codes = crosswalk_df[crosswalk_df['onet_soc'].isna()]['occ1990dd'].unique()

    n_mapped = len(mapped_codes)
    n_unmapped = len(unmapped_codes)
    n_total = n_mapped + n_unmapped

    unweighted_coverage = n_mapped / n_total if n_total > 0 else 0

    # Employment-weighted coverage
    total_emp = sum(emp_data.values()) if emp_data else 1
    mapped_emp = sum(emp_data.get(code, 0) for code in mapped_codes)
    unmapped_emp = sum(emp_data.get(code, 0) for code in unmapped_codes)

    weighted_coverage = mapped_emp / total_emp if total_emp > 0 else unweighted_coverage

    # Confidence tier distribution
    tier_counts = crosswalk_df[crosswalk_df['onet_soc'].notna()]['confidence_tier'].value_counts().to_dict()
    tier_counts['unmapped'] = n_unmapped

    return Occ1990ddCrosswalkResult(
        crosswalk_df=crosswalk_df,
        n_occ1990dd=n_total,
        n_mapped=n_mapped,
        n_unmapped=n_unmapped,
        unweighted_coverage=unweighted_coverage,
        weighted_coverage=weighted_coverage,
        unmapped_employment_share=unmapped_emp / total_emp if total_emp > 0 else 0,
        confidence_distribution=tier_counts,
    )


def load_default_crosswalk() -> Occ1990ddCrosswalkResult:
    """
    Load crosswalk using default paths in the repository.

    Convenience function that locates data files in standard locations.

    Returns:
        Occ1990ddCrosswalkResult with crosswalk and coverage statistics.
    """
    # Find repo root
    repo_root = Path(__file__).parent.parent.parent.parent

    ipums_path = repo_root / "data/external/ipums/cps_1992-2002-occ2010-xwalk.xlsx"
    onet_census_path = repo_root / ".cache/artifacts/v1/mobility/onet_to_census_improved.csv"
    dorn_emp_path = repo_root / "data/external/dorn_replication/dorn_extracted/Autor-Dorn-LowSkillServices-FileArchive.zip Folder/dta/occ1990dd_data2012.dta"
    dorn_task_path = repo_root / "data/external/dorn_replication/occ1990dd_task_alm.dta"

    return build_occ1990dd_to_onet_crosswalk(
        ipums_crosswalk_path=ipums_path,
        onet_census_crosswalk_path=onet_census_path,
        dorn_employment_path=dorn_emp_path,
        dorn_task_path=dorn_task_path,
    )


def save_crosswalk(
    result: Occ1990ddCrosswalkResult,
    output_path: Path,
) -> None:
    """
    Save crosswalk to CSV file.

    Args:
        result: Crosswalk result from build_occ1990dd_to_onet_crosswalk()
        output_path: Path for output CSV
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.crosswalk_df.to_csv(output_path, index=False)


def get_onet_codes_for_occ1990dd(
    crosswalk: Occ1990ddCrosswalkResult,
    occ1990dd: int,
) -> List[tuple[str, float]]:
    """
    Get O*NET-SOC codes and weights for a given occ1990dd code.

    Args:
        crosswalk: Crosswalk result
        occ1990dd: Dorn harmonized occupation code

    Returns:
        List of (onet_soc, weight) tuples
    """
    df = crosswalk.crosswalk_df
    matches = df[(df['occ1990dd'] == occ1990dd) & (df['onet_soc'].notna())]

    return [(row['onet_soc'], row['weight']) for _, row in matches.iterrows()]


def aggregate_onet_to_occ1990dd(
    onet_values: Dict[str, float],
    crosswalk: Occ1990ddCrosswalkResult,
) -> Dict[int, float]:
    """
    Aggregate O*NET-level values to occ1990dd level.

    Uses crosswalk weights to compute weighted average when
    multiple O*NET codes map to one occ1990dd.

    Args:
        onet_values: Dict mapping O*NET-SOC code → value
        crosswalk: Crosswalk result

    Returns:
        Dict mapping occ1990dd → aggregated value
    """
    df = crosswalk.crosswalk_df
    df_valid = df[df['onet_soc'].notna()].copy()

    # Add O*NET values
    df_valid['value'] = df_valid['onet_soc'].map(onet_values)

    # Filter to codes with values
    df_valid = df_valid[df_valid['value'].notna()]

    # Weighted aggregation
    result = {}
    for occ1990dd, group in df_valid.groupby('occ1990dd'):
        total_weight = group['weight'].sum()
        if total_weight > 0:
            weighted_val = (group['value'] * group['weight']).sum() / total_weight
            result[int(occ1990dd)] = weighted_val

    return result
