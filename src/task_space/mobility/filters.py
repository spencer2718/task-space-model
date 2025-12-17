"""
CPS occupation transition filters for measurement error correction.

Implements filtering approaches to identify true occupation transitions from
noisy CPS self-reported occupation codes. Raw CPS occupation switching rates
(~6.7% monthly) substantially exceed true mobility rates (~0.5-1.0%).

Measurement Error Sources:
1. Coding inconsistency: Same job coded differently across interviews
2. Recall error: Respondent uncertainty about occupation classification
3. Interviewer variation: Different coders for same response

Filter Approaches:
1. Persistence filter: Require transition to persist 2+ months
2. Demographic validation: Check AGE/SEX/RACE consistency in linked records
3. Employment filter: Focus on employed-to-employed (EE) transitions

References:
- Kambourov & Manovskii (2008) "Rising Occupational Mobility in the US"
- Moscarini & Thomsson (2007) "Occupational and Job Mobility"
- Neal (1999) "The Complexity of Job Mobility"
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd


@dataclass
class FilterStats:
    """Statistics from applying a transition filter."""
    n_input: int
    n_output: int
    retention_rate: float
    description: str


@dataclass
class FilterPipeline:
    """
    Results from full filter pipeline.

    Attributes:
        transitions_df: DataFrame of verified transitions
        n_raw_transitions: Number of raw (unfiltered) transitions
        n_verified_transitions: Number of transitions after all filters
        filter_stats: Dict of FilterStats by filter name
        overall_retention: Fraction retained after all filters
        assumptions: Modeling assumptions
    """
    transitions_df: pd.DataFrame
    n_raw_transitions: int
    n_verified_transitions: int
    filter_stats: Dict[str, FilterStats]
    overall_retention: float
    assumptions: List[str] = field(default_factory=lambda: [
        "Persistence filter: OCC(t+1) = OCC(t+2) removes coding fluctuations",
        "Demographic validation: AGE ±1, SEX/RACE constant removes false links",
        "Employment filter: EMPSTAT ∈ {10, 12} focuses on E-E transitions",
        "Literature estimate: ~70% of raw CPS transitions are measurement error",
    ])


def validate_cps_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """Check that DataFrame has required CPS columns."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def apply_persistence_filter(
    df: pd.DataFrame,
    person_id: str = "CPSIDP",
    time_col: str = "YEARMONTH",
    occ_col: str = "OCC2010",
    min_persistence: int = 2,
) -> Tuple[pd.DataFrame, FilterStats]:
    """
    Filter occupation transitions to those that persist.

    A transition from OCC(t-1) to OCC(t) is considered "real" only if:
    - OCC(t-1) stable: OCC(t-2) == OCC(t-1) (was in origin for prior period)
    - Transition: OCC(t) != OCC(t-1)
    - Persistence: OCC(t) == OCC(t+1) [== OCC(t+2)] (new occupation persists)

    This filters out coding errors where occupation temporarily changes
    then reverts to original.

    Rationale (Kambourov & Manovskii 2008):
        Raw CPS occupation switches include substantial measurement error.
        True occupational mobility is ~0.5-1.0% monthly, but raw CPS shows
        ~3-6%. The persistence filter removes "bounce-back" errors where
        a person's coded occupation changes for one month then returns.

    Args:
        df: CPS panel data, sorted by person_id and time
        person_id: Column identifying individuals
        time_col: Column identifying time periods
        occ_col: Column with occupation codes
        min_persistence: Minimum months new occupation must persist

    Returns:
        DataFrame with transition flags added
        FilterStats object

    Expected retention: ~25-50% of raw transitions
    """
    validate_cps_columns(df, [person_id, time_col, occ_col])

    df = df.sort_values([person_id, time_col]).copy()

    # Create leads and lags within person
    df["occ_lag1"] = df.groupby(person_id)[occ_col].shift(1)
    df["occ_lag2"] = df.groupby(person_id)[occ_col].shift(2)
    df["occ_lead1"] = df.groupby(person_id)[occ_col].shift(-1)

    if min_persistence >= 3:
        df["occ_lead2"] = df.groupby(person_id)[occ_col].shift(-2)

    # Origin stability: was in same occupation for at least one prior period
    df["origin_stable"] = (df["occ_lag1"] == df["occ_lag2"]) | df["occ_lag2"].isna()

    # Raw transition: occupation changed from previous period
    df["raw_transition"] = (df[occ_col] != df["occ_lag1"]) & df["occ_lag1"].notna()

    # Persistent transition: new occupation matches next period(s)
    if min_persistence == 1:
        df["destination_stable"] = True
    elif min_persistence == 2:
        df["destination_stable"] = (df[occ_col] == df["occ_lead1"])
    else:  # min_persistence >= 3
        df["destination_stable"] = (
            (df[occ_col] == df["occ_lead1"]) &
            (df[occ_col] == df["occ_lead2"])
        )

    # Verified transition: stable origin + raw transition + stable destination
    df["verified_transition"] = (
        df["origin_stable"] &
        df["raw_transition"] &
        df["destination_stable"]
    )

    # Handle NaN in leads (end of observation window)
    df["verified_transition"] = df["verified_transition"].fillna(False)

    n_raw = int(df["raw_transition"].sum())
    n_verified = int(df["verified_transition"].sum())
    retention = n_verified / n_raw if n_raw > 0 else 0.0

    stats = FilterStats(
        n_input=n_raw,
        n_output=n_verified,
        retention_rate=retention,
        description=f"Persistence filter (min {min_persistence} months)",
    )

    return df, stats


def apply_demographic_validation(
    df: pd.DataFrame,
    person_id: str = "CPSIDP",
    time_col: str = "YEARMONTH",
    age_col: str = "AGE",
    sex_col: str = "SEX",
    race_col: str = "RACE",
    max_age_change: int = 2,
) -> Tuple[pd.DataFrame, FilterStats]:
    """
    Validate person links using demographic consistency.

    IPUMS recommendation:
    - SEX must be constant across linked records
    - RACE must be constant across linked records
    - AGE must increase by 0 or 1 between adjacent months

    Args:
        df: CPS panel with person links
        max_age_change: Maximum allowed age increase per month

    Returns:
        DataFrame with validation flags
        FilterStats object

    Expected retention: ~80-95% of linked records
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
    n_valid = int(df["demo_valid"].sum())

    stats = FilterStats(
        n_input=n_raw,
        n_output=n_valid,
        retention_rate=n_valid / n_raw if n_raw > 0 else 0.0,
        description="Demographic validation filter",
    )

    return df, stats


def apply_employment_filter(
    df: pd.DataFrame,
    person_id: str = "CPSIDP",
    time_col: str = "YEARMONTH",
    empstat_col: str = "EMPSTAT",
    employed_codes: List[int] = None,
    require_continuous: bool = True,
) -> Tuple[pd.DataFrame, FilterStats]:
    """
    Filter to employed-to-employed (EE) transitions only.

    Motivation: Occupation codes are only meaningful for employed workers.
    Transitions through unemployment (EUE) may involve occupation change
    unrelated to standard labor market frictions.

    Args:
        df: CPS panel data
        empstat_col: Employment status column
        employed_codes: EMPSTAT values indicating employment (default [10, 12])
        require_continuous: If True, require employment in t-1, t, t+1

    Returns:
        DataFrame filtered to EE eligible observations
        FilterStats object

    Expected retention: ~60-70% of sample
    """
    if employed_codes is None:
        employed_codes = [10, 12]  # IPUMS employed codes

    validate_cps_columns(df, [person_id, time_col, empstat_col])

    df = df.sort_values([person_id, time_col]).copy()

    # Flag employed observations
    df["employed"] = df[empstat_col].isin(employed_codes)

    if require_continuous:
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
    n_ee = int(df["ee_eligible"].sum())

    stats = FilterStats(
        n_input=n_raw,
        n_output=n_ee,
        retention_rate=n_ee / n_raw if n_raw > 0 else 0.0,
        description="EE employment filter" + (" (continuous)" if require_continuous else ""),
    )

    return df[df["ee_eligible"]].copy(), stats


def build_verified_transitions(
    df: pd.DataFrame,
    use_demographic_validation: bool = True,
    persistence_months: int = 2,
    require_continuous_ee: bool = True,
    person_id: str = "CPSIDP",
    time_col: str = "YEARMONTH",
    occ_col: str = "OCC2010",
) -> FilterPipeline:
    """
    Apply full filter pipeline to identify verified occupation transitions.

    Filter order (per measurement error literature):
    1. Employment filter: Remove non-employed observations
    2. Demographic validation: Remove inconsistent person links
    3. Persistence filter: Remove spurious transitions

    Args:
        df: Raw CPS panel data
        use_demographic_validation: Apply demographic consistency checks
        persistence_months: Minimum months new occupation must persist
        require_continuous_ee: Require continuous employment around transition
        person_id: Person identifier column
        time_col: Time period column
        occ_col: Occupation code column

    Returns:
        FilterPipeline with verified transitions and metadata.

    Example:
        >>> result = build_verified_transitions(cps_data)
        >>> print(f"Verified {result.n_verified_transitions} transitions")
        >>> print(f"Overall retention: {result.overall_retention:.1%}")
    """
    all_stats = {}

    # Step 1: Employment filter
    df, emp_stats = apply_employment_filter(
        df,
        person_id=person_id,
        time_col=time_col,
        require_continuous=require_continuous_ee,
    )
    all_stats["employment"] = emp_stats

    # Step 2: Demographic validation
    if use_demographic_validation:
        df, demo_stats = apply_demographic_validation(
            df,
            person_id=person_id,
            time_col=time_col,
        )
        all_stats["demographic"] = demo_stats
        df = df[df["demo_valid"]].copy()

    # Step 3: Persistence filter
    df, persist_stats = apply_persistence_filter(
        df,
        person_id=person_id,
        time_col=time_col,
        occ_col=occ_col,
        min_persistence=persistence_months,
    )
    all_stats["persistence"] = persist_stats

    # Extract verified transitions
    transitions_df = df[df["verified_transition"]].copy()

    # Add origin occupation
    transitions_df["origin_occ"] = transitions_df["occ_lag1"]
    transitions_df["dest_occ"] = transitions_df[occ_col]

    n_raw = persist_stats.n_input
    n_verified = len(transitions_df)
    overall_retention = n_verified / n_raw if n_raw > 0 else 0.0

    return FilterPipeline(
        transitions_df=transitions_df,
        n_raw_transitions=n_raw,
        n_verified_transitions=n_verified,
        filter_stats=all_stats,
        overall_retention=overall_retention,
    )


def load_verified_transitions(
    path: Optional[str] = None,
    year_range: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """
    Load pre-computed verified transitions from parquet file.

    Args:
        path: Path to parquet file. If None, uses default location.
        year_range: Optional (start_year, end_year) inclusive filter.
                   E.g., (2015, 2019) for pre-COVID only.

    Returns:
        DataFrame of verified transitions.
    """
    from pathlib import Path

    if path is None:
        default_paths = [
            Path("data/processed/mobility/verified_transitions.parquet"),
            Path("killshot/cps_mobility/data/verified_transitions.parquet"),
        ]
        for p in default_paths:
            if p.exists():
                path = p
                break

    if path is None:
        raise FileNotFoundError(
            "Verified transitions not found. Run CPS processing pipeline first."
        )

    df = pd.read_parquet(path)

    if year_range is not None:
        df['_year'] = df['YEARMONTH'] // 100
        df = df[(df['_year'] >= year_range[0]) & (df['_year'] <= year_range[1])]
        df = df.drop(columns=['_year'])

    return df
