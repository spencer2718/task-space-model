"""
Task B: Measurement Error Filter Logic for CPS Occupation Transitions

Implements three filtering approaches to handle spurious occupation transitions:
1. Persistence filter: Require transition to persist for 2+ months
2. CPSIDV validation: Use validated links with demographic consistency
3. Employment status filter: Focus on continuous EE transitions

Based on:
- Kambourov and Manovskii (2004) "A Cautionary Note on Using (March) CPS Data"
- Neal (1999) employer-switch methodology
- Moscarini and Thomsson (2007) on measurement error in CPS

Literature estimates:
- Raw monthly switching: ~3.2% at 3-digit level
- After measurement error correction: ~0.5-1.0%
- Implies ~70% of raw transitions are spurious
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class FilterStats:
    """Statistics from applying a transition filter."""
    n_input: int
    n_output: int
    retention_rate: float
    description: str


def validate_cps_panel(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """Check that DataFrame has required CPS columns."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


# =============================================================================
# FILTER 1: PERSISTENCE FILTER
# =============================================================================

def apply_persistence_filter(
    df: pd.DataFrame,
    person_id: str = "CPSIDP",
    time_col: str = "YEARMONTH",
    occ_col: str = "OCC2010",
    min_persistence: int = 2
) -> Tuple[pd.DataFrame, FilterStats]:
    """
    Filter occupation transitions to those that persist.

    A transition from OCC(t) to OCC(t+1) is considered "real" only if:
    - OCC(t) != OCC(t+1)
    - OCC(t+1) == OCC(t+2) (the new occupation persists)

    This filters out coding errors where occupation temporarily changes
    then reverts to original.

    Args:
        df: CPS panel data, sorted by person_id and time
        person_id: Column identifying individuals
        time_col: Column identifying time periods
        occ_col: Column with occupation codes
        min_persistence: Minimum months new occupation must persist (default 2)

    Returns:
        DataFrame with transitions flagged
        FilterStats object

    Expected retention: ~25-50% of raw transitions (per literature)
    """
    validate_cps_panel(df, [person_id, time_col, occ_col])

    # Sort and create lagged occupation variables
    df = df.sort_values([person_id, time_col]).copy()

    # Create leads and lags within person
    df["occ_lag1"] = df.groupby(person_id)[occ_col].shift(1)
    df["occ_lead1"] = df.groupby(person_id)[occ_col].shift(-1)

    if min_persistence >= 2:
        df["occ_lead2"] = df.groupby(person_id)[occ_col].shift(-2)

    # Identify raw transitions: occ changed from previous period
    df["raw_transition"] = (df[occ_col] != df["occ_lag1"]) & df["occ_lag1"].notna()

    # Identify persistent transitions: new occupation matches next period
    if min_persistence == 1:
        df["persistent_transition"] = df["raw_transition"]
    elif min_persistence == 2:
        df["persistent_transition"] = (
            df["raw_transition"] &
            (df[occ_col] == df["occ_lead1"])
        )
    else:  # min_persistence >= 3
        df["persistent_transition"] = (
            df["raw_transition"] &
            (df[occ_col] == df["occ_lead1"]) &
            (df[occ_col] == df["occ_lead2"])
        )

    # Handle NaN in leads (end of observation window)
    df["persistent_transition"] = df["persistent_transition"].fillna(False)

    n_raw = df["raw_transition"].sum()
    n_persistent = df["persistent_transition"].sum()
    retention = n_persistent / n_raw if n_raw > 0 else 0

    stats = FilterStats(
        n_input=int(n_raw),
        n_output=int(n_persistent),
        retention_rate=float(retention),
        description=f"Persistence filter (min {min_persistence} months)"
    )

    return df, stats


# =============================================================================
# FILTER 2: CPSIDV VALIDATION FILTER
# =============================================================================

def apply_cpsidv_filter(
    df: pd.DataFrame,
    person_id_raw: str = "CPSIDP",
    person_id_validated: str = "CPSIDV",
    age_col: str = "AGE",
    sex_col: str = "SEX",
    race_col: str = "RACE",
    time_col: str = "YEARMONTH"
) -> Tuple[pd.DataFrame, FilterStats]:
    """
    Filter to validated person links using CPSIDV.

    CPSIDV links respondents across months with consistent:
    - Census Bureau-assigned identifiers
    - SEX and RACE values
    - AGE values that change in expected ways

    This removes false positive matches from CPSIDP.

    Args:
        df: CPS panel data
        person_id_raw: Raw linking variable (CPSIDP)
        person_id_validated: Validated linking variable (CPSIDV)
        age_col, sex_col, race_col: Demographic validation columns
        time_col: Time identifier

    Returns:
        DataFrame filtered to validated links
        FilterStats object

    Expected retention: ~80-90% of CPSIDP links (per IPUMS documentation)
    """
    # Check for CPSIDV availability
    if person_id_validated not in df.columns:
        print("WARNING: CPSIDV not available, falling back to demographic validation")
        return apply_demographic_validation(df, person_id_raw, age_col, sex_col, race_col, time_col)

    df = df.copy()

    # CPSIDV == 0 or missing indicates unvalidated link
    df["validated_link"] = (df[person_id_validated] > 0) & df[person_id_validated].notna()

    n_raw = len(df)
    n_validated = df["validated_link"].sum()

    # Keep only validated observations
    df_filtered = df[df["validated_link"]].copy()

    stats = FilterStats(
        n_input=int(n_raw),
        n_output=int(n_validated),
        retention_rate=float(n_validated / n_raw) if n_raw > 0 else 0,
        description="CPSIDV validation filter"
    )

    return df_filtered, stats


def apply_demographic_validation(
    df: pd.DataFrame,
    person_id: str = "CPSIDP",
    age_col: str = "AGE",
    sex_col: str = "SEX",
    race_col: str = "RACE",
    time_col: str = "YEARMONTH",
    max_age_change: int = 2  # Allow for birthday + rounding
) -> Tuple[pd.DataFrame, FilterStats]:
    """
    Validate person links using demographic consistency.

    Following IPUMS recommendation:
    - SEX must be constant
    - RACE must be constant
    - AGE must increase by 0 or 1 between adjacent months

    Args:
        df: CPS panel with CPSIDP links
        max_age_change: Maximum allowed age increase per month

    Returns:
        DataFrame with invalid links flagged
        FilterStats object
    """
    df = df.sort_values([person_id, time_col]).copy()

    # Create lagged demographics
    df["age_lag"] = df.groupby(person_id)[age_col].shift(1)
    df["sex_lag"] = df.groupby(person_id)[sex_col].shift(1)
    df["race_lag"] = df.groupby(person_id)[race_col].shift(1)

    # Check consistency
    df["age_valid"] = (
        df["age_lag"].isna() |  # First observation
        ((df[age_col] >= df["age_lag"]) &
         (df[age_col] <= df["age_lag"] + max_age_change))
    )
    df["sex_valid"] = df["sex_lag"].isna() | (df[sex_col] == df["sex_lag"])
    df["race_valid"] = df["race_lag"].isna() | (df[race_col] == df["race_lag"])

    df["demo_valid"] = df["age_valid"] & df["sex_valid"] & df["race_valid"]

    n_raw = len(df)
    n_valid = df["demo_valid"].sum()

    stats = FilterStats(
        n_input=int(n_raw),
        n_output=int(n_valid),
        retention_rate=float(n_valid / n_raw) if n_raw > 0 else 0,
        description="Demographic validation filter"
    )

    return df, stats


# =============================================================================
# FILTER 3: EMPLOYMENT STATUS FILTER
# =============================================================================

def apply_employment_filter(
    df: pd.DataFrame,
    person_id: str = "CPSIDP",
    time_col: str = "YEARMONTH",
    empstat_col: str = "EMPSTAT",
    employed_codes: List[int] = [10, 12],  # IPUMS employed codes
    require_continuous: bool = True
) -> Tuple[pd.DataFrame, FilterStats]:
    """
    Filter to employed-to-employed (EE) transitions only.

    Motivation: Occupation codes are only meaningful for employed workers.
    Transitions through unemployment (EUE) may involve occupation change
    unrelated to labor market frictions.

    Args:
        df: CPS panel data
        empstat_col: Employment status column
        employed_codes: EMPSTAT values indicating employment
        require_continuous: If True, require employment in t-1, t, t+1

    Returns:
        DataFrame filtered to EE transitions
        FilterStats object

    Expected retention: ~60-70% of employed sample (EE vs EU/UE)
    """
    validate_cps_panel(df, [person_id, time_col, empstat_col])

    df = df.sort_values([person_id, time_col]).copy()

    # Flag employed observations
    df["employed"] = df[empstat_col].isin(employed_codes)

    if require_continuous:
        # Create leads and lags
        df["employed_lag"] = df.groupby(person_id)["employed"].shift(1)
        df["employed_lead"] = df.groupby(person_id)["employed"].shift(-1)

        # Require employment in t-1, t, and t+1
        df["ee_eligible"] = (
            df["employed"] &
            df["employed_lag"].fillna(True) &  # Don't exclude first obs
            df["employed_lead"].fillna(True)    # Don't exclude last obs
        )
    else:
        df["ee_eligible"] = df["employed"]

    n_raw = len(df)
    n_ee = df["ee_eligible"].sum()

    stats = FilterStats(
        n_input=int(n_raw),
        n_output=int(n_ee),
        retention_rate=float(n_ee / n_raw) if n_raw > 0 else 0,
        description="EE employment filter" + (" (continuous)" if require_continuous else "")
    )

    return df[df["ee_eligible"]].copy(), stats


# =============================================================================
# COMBINED FILTER PIPELINE
# =============================================================================

def apply_all_filters(
    df: pd.DataFrame,
    use_cpsidv: bool = True,
    persistence_months: int = 2,
    require_continuous_ee: bool = True
) -> Tuple[pd.DataFrame, Dict[str, FilterStats]]:
    """
    Apply all filters in sequence.

    Recommended order (per memo):
    1. Employment filter (removes non-employed)
    2. CPSIDV/demographic validation (removes bad links)
    3. Persistence filter (removes spurious transitions)

    Args:
        df: Raw CPS panel
        use_cpsidv: Use CPSIDV if available
        persistence_months: Minimum persistence for transitions
        require_continuous_ee: Require continuous employment

    Returns:
        Filtered DataFrame
        Dictionary of FilterStats by filter name
    """
    all_stats = {}

    # Step 1: Employment filter
    df, stats = apply_employment_filter(
        df,
        require_continuous=require_continuous_ee
    )
    all_stats["employment"] = stats
    print(f"Employment filter: {stats.retention_rate:.1%} retained")

    # Step 2: Link validation
    if use_cpsidv and "CPSIDV" in df.columns:
        df, stats = apply_cpsidv_filter(df)
    else:
        df, stats = apply_demographic_validation(df)
    all_stats["validation"] = stats
    print(f"Validation filter: {stats.retention_rate:.1%} retained")

    # Step 3: Persistence filter
    df, stats = apply_persistence_filter(
        df,
        min_persistence=persistence_months
    )
    all_stats["persistence"] = stats
    print(f"Persistence filter: {stats.retention_rate:.1%} of transitions retained")

    # Compute overall stats
    if df["persistent_transition"].any():
        n_final = df["persistent_transition"].sum()
    else:
        n_final = 0

    return df, all_stats


# =============================================================================
# EXPECTED SAMPLE LOSS ESTIMATES
# =============================================================================

def estimate_sample_loss() -> Dict[str, float]:
    """
    Estimate expected sample loss from each filter based on literature.

    Returns dictionary with retention rates for each filter.
    """
    estimates = {
        "rotation_attrition": {
            "description": "CPS 4-8-4 rotation (only ~50% linked month-to-month)",
            "retention": 0.50,
            "source": "CPS design; IPUMS documentation"
        },
        "employment_filter": {
            "description": "Keep only employed workers with continuous employment",
            "retention": 0.65,
            "source": "BLS employment-population ratio; EU/UE flows"
        },
        "demographic_validation": {
            "description": "Remove links with inconsistent AGE/SEX/RACE",
            "retention": 0.90,
            "source": "IPUMS documentation on CPSIDV"
        },
        "persistence_filter": {
            "description": "Require occupation change to persist 2+ months",
            "retention": 0.35,
            "source": "Kambourov & Manovskii (2004); ~70% of raw switches spurious"
        },
        "combined": {
            "description": "All filters applied sequentially",
            "retention": 0.50 * 0.65 * 0.90 * 0.35,  # ≈ 10%
            "note": "Of raw transitions in matched sample"
        }
    }

    print("=" * 60)
    print("EXPECTED SAMPLE LOSS ESTIMATES")
    print("=" * 60)

    cumulative = 1.0
    for name, info in estimates.items():
        if name == "combined":
            continue
        cumulative *= info["retention"]
        print(f"\n{name}:")
        print(f"  {info['description']}")
        print(f"  Retention: {info['retention']:.0%}")
        print(f"  Cumulative: {cumulative:.1%}")

    print(f"\n{'='*60}")
    print(f"OVERALL EXPECTED RETENTION: {cumulative:.1%}")
    print(f"{'='*60}")

    return estimates


# =============================================================================
# MAIN: Document filter logic and expected losses
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TASK B: MEASUREMENT ERROR FILTER DOCUMENTATION")
    print("=" * 70)

    # Document the filters
    print("\n## Filter 1: Persistence Filter")
    print("Logic: OCC(t) != OCC(t+1) AND OCC(t+1) == OCC(t+2)")
    print("Purpose: Remove coding errors that revert in next period")
    print("Expected retention: 25-50% of raw transitions")

    print("\n## Filter 2: CPSIDV/Demographic Validation")
    print("Logic: Use CPSIDV (validated links) OR check AGE/SEX/RACE consistency")
    print("Purpose: Remove false positive person matches")
    print("Expected retention: 80-90% of CPSIDP links")

    print("\n## Filter 3: Employment Status Filter")
    print("Logic: Require EMPSTAT = employed in t-1, t, t+1")
    print("Purpose: Focus on EE transitions, exclude EU/UE")
    print("Expected retention: 60-70% of sample")

    # Estimate losses
    print("\n")
    estimates = estimate_sample_loss()

    # Save documentation
    import json
    output = {
        "filters": {
            "persistence": {
                "logic": "OCC(t) != OCC(t+1) AND OCC(t+1) == OCC(t+2)",
                "purpose": "Remove coding errors",
                "expected_retention": 0.35,
            },
            "validation": {
                "logic": "CPSIDV > 0 OR (AGE consistent AND SEX constant AND RACE constant)",
                "purpose": "Remove false positive person matches",
                "expected_retention": 0.90,
            },
            "employment": {
                "logic": "EMPSTAT in [10, 12] for t-1, t, t+1",
                "purpose": "Focus on EE transitions",
                "expected_retention": 0.65,
            }
        },
        "overall_retention_estimate": 0.50 * 0.65 * 0.90 * 0.35,
        "note": "Combined retention ~10% of raw transitions in matched sample"
    }

    output_path = "temp/mobility_feasibility/outputs/filter_documentation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved filter documentation to: {output_path}")
