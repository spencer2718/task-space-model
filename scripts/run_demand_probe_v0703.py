#!/usr/bin/env python3
"""
Experiment v0.7.0.3: BLS Demand Probe

Tests whether occupation-level demand (projected openings) improves
destination prediction beyond geometry alone.

Compares:
- Geometry-only: rank destinations by 1/d_wasserstein
- Demand-weighted: rank by openings / d_wasserstein

Usage:
    python scripts/run_demand_probe_v0703.py
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Canonical imports from task_space
from task_space.mobility.io import (
    load_transitions,
    get_holdout_transitions,
    load_wasserstein_census,
    DEFAULT_HOLDOUT_YEAR,
)
from task_space.utils.experiments import save_experiment_output


# =============================================================================
# Data Loading (script-specific)
# =============================================================================

def load_bls_projections() -> pd.DataFrame:
    """
    Load BLS Employment Projections (Table 1.10).

    Returns DataFrame with SOC code and annual openings.
    """
    xl = pd.ExcelFile("data/external/bls_projections/occupation.xlsx")
    df = pd.read_excel(xl, sheet_name="Table 1.10", header=1)

    # Filter to detailed occupations (Line item, not Summary)
    detailed = df[df["Occupation type"] == "Line item"].copy()

    # Rename columns for clarity
    detailed = detailed.rename(columns={
        "2024 National Employment Matrix code": "soc_code",
        "2024 National Employment Matrix title": "title",
        "Occupational openings, 2024–34 annual average": "openings",
    })

    return detailed[["soc_code", "title", "openings"]].copy()


def load_soc_to_census_crosswalk() -> Dict[str, int]:
    """
    Load SOC 6-digit to Census 2010 crosswalk.

    Returns dict mapping SOC code (11-1011) to Census code (10).
    """
    xwalk = pd.read_csv(".cache/artifacts/v1/mobility/onet_to_census_improved.csv")

    # Get unique SOC -> Census mappings
    # Multiple O*NET codes may map to same SOC
    soc_census = xwalk[["soc_6digit", "census_2010"]].drop_duplicates()
    soc_census = soc_census.dropna()

    # Take first Census code for each SOC (they should be same)
    soc_to_census = {}
    for _, row in soc_census.iterrows():
        soc = row["soc_6digit"]
        census = int(row["census_2010"])
        if soc not in soc_to_census:
            soc_to_census[soc] = census

    return soc_to_census


# =============================================================================
# Prediction Methods
# =============================================================================

def compute_geometry_scores(
    origin: int,
    destinations: List[int],
    d_sem: np.ndarray,
    census_codes: List[int],
) -> Dict[int, float]:
    """
    Compute geometry-only scores for destinations.

    Score = 1 / d_wasserstein (higher = more attractive)
    """
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    if origin not in code_to_idx:
        return {}

    origin_idx = code_to_idx[origin]
    scores = {}

    for dest in destinations:
        if dest not in code_to_idx:
            continue
        dest_idx = code_to_idx[dest]
        d = d_sem[origin_idx, dest_idx]
        if d > 0:
            scores[dest] = 1.0 / d
        else:
            scores[dest] = float("inf")  # Same occupation

    return scores


def compute_demand_weighted_scores(
    origin: int,
    destinations: List[int],
    d_sem: np.ndarray,
    census_codes: List[int],
    openings_by_census: Dict[int, float],
    median_openings: float,
) -> Dict[int, float]:
    """
    Compute demand-weighted scores for destinations.

    Score = openings / d_wasserstein (higher = more attractive)

    Missing openings assigned median value.
    """
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    if origin not in code_to_idx:
        return {}

    origin_idx = code_to_idx[origin]
    scores = {}

    for dest in destinations:
        if dest not in code_to_idx:
            continue
        dest_idx = code_to_idx[dest]
        d = d_sem[origin_idx, dest_idx]

        # Get openings, use median if missing
        openings = openings_by_census.get(dest, median_openings)

        if d > 0:
            scores[dest] = openings / d
        else:
            scores[dest] = float("inf") * openings  # Same occupation

    return scores


def rank_destinations(scores: Dict[int, float]) -> List[int]:
    """Rank destinations by score (descending)."""
    return [k for k, v in sorted(scores.items(), key=lambda x: -x[1])]


# =============================================================================
# Evaluation
# =============================================================================

def compute_overlap(predicted: List[int], observed: List[int], k: int) -> float:
    """Compute top-k overlap between predicted and observed."""
    pred_set = set(predicted[:k])
    obs_set = set(observed[:k])
    return len(pred_set & obs_set) / k if k > 0 else 0.0


def evaluate_predictions(
    holdout_df: pd.DataFrame,
    d_sem: np.ndarray,
    census_codes: List[int],
    openings_by_census: Dict[int, float],
    median_openings: float,
) -> Tuple[Dict, Dict]:
    """
    Evaluate geometry-only and demand-weighted predictions on holdout.

    Returns (geometry_results, demand_results).
    """
    # Get unique origins in holdout
    origins = holdout_df["origin_occ"].unique()

    # Get all possible destinations (any Census code that appears in data)
    all_destinations = list(set(holdout_df["dest_occ"].unique()) |
                           set(holdout_df["origin_occ"].unique()))

    # Aggregate observed transition counts
    observed_counts = holdout_df.groupby("dest_occ").size().to_dict()
    observed_ranking = [k for k, v in sorted(observed_counts.items(), key=lambda x: -x[1])]

    # Aggregate predicted scores
    geometry_scores_agg = {}
    demand_scores_agg = {}

    for origin in origins:
        # Get geometry scores
        geo_scores = compute_geometry_scores(
            origin, all_destinations, d_sem, census_codes
        )
        for dest, score in geo_scores.items():
            geometry_scores_agg[dest] = geometry_scores_agg.get(dest, 0) + score

        # Get demand-weighted scores
        dem_scores = compute_demand_weighted_scores(
            origin, all_destinations, d_sem, census_codes,
            openings_by_census, median_openings
        )
        for dest, score in dem_scores.items():
            demand_scores_agg[dest] = demand_scores_agg.get(dest, 0) + score

    geometry_ranking = rank_destinations(geometry_scores_agg)
    demand_ranking = rank_destinations(demand_scores_agg)

    # Compute metrics
    # Build ranking vectors for Spearman
    # Need common set of destinations that appear in both predicted and observed
    common_dests = set(geometry_ranking) & set(observed_ranking)

    if len(common_dests) < 3:
        return (
            {"error": "Too few common destinations for Spearman"},
            {"error": "Too few common destinations for Spearman"},
        )

    # Get ranks for common destinations
    obs_ranks = {d: i for i, d in enumerate(observed_ranking)}
    geo_ranks = {d: i for i, d in enumerate(geometry_ranking)}
    dem_ranks = {d: i for i, d in enumerate(demand_ranking)}

    common_list = list(common_dests)
    obs_rank_vec = [obs_ranks[d] for d in common_list]
    geo_rank_vec = [geo_ranks[d] for d in common_list]
    dem_rank_vec = [dem_ranks[d] for d in common_list]

    geo_spearman, _ = stats.spearmanr(obs_rank_vec, geo_rank_vec)
    dem_spearman, _ = stats.spearmanr(obs_rank_vec, dem_rank_vec)

    geometry_results = {
        "spearman_rho": round(geo_spearman, 4),
        "top5_overlap": round(compute_overlap(geometry_ranking, observed_ranking, 5), 2),
        "top10_overlap": round(compute_overlap(geometry_ranking, observed_ranking, 10), 2),
        "n_ranked": len(geometry_ranking),
        "top5_predicted": [str(d) for d in geometry_ranking[:5]],
    }

    demand_results = {
        "spearman_rho": round(dem_spearman, 4),
        "top5_overlap": round(compute_overlap(demand_ranking, observed_ranking, 5), 2),
        "top10_overlap": round(compute_overlap(demand_ranking, observed_ranking, 10), 2),
        "n_ranked": len(demand_ranking),
        "top5_predicted": [str(d) for d in demand_ranking[:5]],
    }

    return geometry_results, demand_results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Experiment v0.7.0.3: BLS Demand Probe")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load BLS projections
    # =========================================================================
    print("\n[1] Loading BLS Employment Projections...")

    bls_df = load_bls_projections()
    print(f"    BLS occupations: {len(bls_df)}")
    print(f"    Total annual openings: {bls_df['openings'].sum():,.0f}k")
    print(f"    Openings column: 'Occupational openings, 2024–34 annual average'")

    # =========================================================================
    # Step 2: Build SOC → Census crosswalk
    # =========================================================================
    print("\n[2] Building SOC → Census crosswalk...")

    soc_to_census = load_soc_to_census_crosswalk()
    print(f"    SOC codes with Census mapping: {len(soc_to_census)}")

    # Map BLS openings to Census codes
    openings_by_census = {}
    matched = 0
    unmatched = 0

    for _, row in bls_df.iterrows():
        soc = row["soc_code"]
        if soc in soc_to_census:
            census = soc_to_census[soc]
            # Sum openings for SOCs that map to same Census code
            openings_by_census[census] = openings_by_census.get(census, 0) + row["openings"]
            matched += 1
        else:
            unmatched += 1

    linkage_rate = matched / len(bls_df) if len(bls_df) > 0 else 0
    print(f"    BLS → Census matched: {matched}/{len(bls_df)} ({linkage_rate:.1%})")
    print(f"    Census codes with openings: {len(openings_by_census)}")

    if linkage_rate < 0.5:
        print("\n    STOP: Crosswalk linkage rate < 50%")
        return None

    # Compute median openings for missing values
    median_openings = np.median(list(openings_by_census.values()))
    print(f"    Median openings (for missing): {median_openings:.1f}k")

    # =========================================================================
    # Step 3: Load holdout transitions
    # =========================================================================
    print("\n[3] Loading holdout transitions...")

    transitions = load_transitions()
    holdout_df = get_holdout_transitions(year=DEFAULT_HOLDOUT_YEAR)
    print(f"    Total transitions: {len(transitions):,}")
    print(f"    {DEFAULT_HOLDOUT_YEAR} holdout: {len(holdout_df):,}")

    # Get unique origins
    holdout_origins = holdout_df["origin_occ"].unique()
    print(f"    Unique origins: {len(holdout_origins)}")

    # =========================================================================
    # Step 4: Load Wasserstein distances
    # =========================================================================
    print("\n[4] Loading Wasserstein distances...")

    d_sem, census_codes = load_wasserstein_census()
    print(f"    Distance matrix: {d_sem.shape}")
    print(f"    Census codes: {len(census_codes)}")

    # Check overlap with holdout
    holdout_in_matrix = sum(1 for o in holdout_origins if o in census_codes)
    print(f"    Holdout origins in matrix: {holdout_in_matrix}/{len(holdout_origins)}")

    # =========================================================================
    # Step 5: Evaluate predictions
    # =========================================================================
    print("\n[5] Evaluating predictions...")

    geometry_results, demand_results = evaluate_predictions(
        holdout_df, d_sem, census_codes, openings_by_census, median_openings
    )

    print("\n    Geometry-only:")
    print(f"      Spearman ρ: {geometry_results.get('spearman_rho', 'N/A')}")
    print(f"      Top-5 overlap: {geometry_results.get('top5_overlap', 'N/A')}")
    print(f"      Top-10 overlap: {geometry_results.get('top10_overlap', 'N/A')}")

    print("\n    Demand-weighted:")
    print(f"      Spearman ρ: {demand_results.get('spearman_rho', 'N/A')}")
    print(f"      Top-5 overlap: {demand_results.get('top5_overlap', 'N/A')}")
    print(f"      Top-10 overlap: {demand_results.get('top10_overlap', 'N/A')}")

    # =========================================================================
    # Step 6: Compute comparison
    # =========================================================================
    print("\n[6] Comparison...")

    geo_rho = geometry_results.get("spearman_rho", 0) or 0
    dem_rho = demand_results.get("spearman_rho", 0) or 0
    rho_delta = dem_rho - geo_rho

    geo_top5 = geometry_results.get("top5_overlap", 0) or 0
    dem_top5 = demand_results.get("top5_overlap", 0) or 0
    top5_delta = dem_top5 - geo_top5

    # Check if demand-weighted produces different rankings
    geo_top5_list = geometry_results.get("top5_predicted", [])
    dem_top5_list = demand_results.get("top5_predicted", [])
    rankings_identical = geo_top5_list == dem_top5_list

    if rankings_identical:
        interpretation = "WARNING: Rankings identical - may indicate implementation error or dominant geometry signal"
    elif rho_delta > 0.05:
        interpretation = f"Demand improves correlation by {rho_delta:.3f} (meaningful improvement)"
    elif rho_delta > 0:
        interpretation = f"Demand marginally improves correlation by {rho_delta:.3f}"
    elif rho_delta < -0.05:
        interpretation = f"Demand WORSENS correlation by {abs(rho_delta):.3f} (geometry alone is better)"
    else:
        interpretation = f"No meaningful difference (Δρ = {rho_delta:.3f})"

    print(f"    ρ delta (demand - geometry): {rho_delta:+.4f}")
    print(f"    Top-5 delta: {top5_delta:+.2f}")
    print(f"    Rankings identical: {rankings_identical}")
    print(f"    Interpretation: {interpretation}")

    comparison = {
        "rho_delta": round(rho_delta, 4),
        "top5_delta": round(top5_delta, 2),
        "rankings_identical": rankings_identical,
        "interpretation": interpretation,
    }

    # =========================================================================
    # Step 7: Assemble and save output
    # =========================================================================
    print("\n[7] Saving results...")

    output = {
        "experiment": "demand_probe_v0703",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_summary": {
            "bls_occupations": len(bls_df),
            "crosswalk_linkage_rate": round(linkage_rate, 3),
            "census_with_openings": len(openings_by_census),
            "holdout_origins": len(holdout_origins),
            "holdout_transitions": len(holdout_df),
        },
        "geometry_only": geometry_results,
        "demand_weighted": demand_results,
        "comparison": comparison,
        "observed_top5": list(holdout_df.groupby("dest_occ").size().nlargest(5).index),
    }

    output_path = save_experiment_output("demand_probe_v0703", output)
    print(f"    Saved to: {output_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("DEMAND PROBE SUMMARY")
    print("=" * 60)

    print(f"\nData:")
    print(f"  BLS occupations: {len(bls_df)}")
    print(f"  Crosswalk linkage: {linkage_rate:.1%}")
    print(f"  Holdout transitions: {len(holdout_df):,}")

    print(f"\nResults:")
    print(f"  Geometry-only:   ρ = {geo_rho:.4f}, top-5 = {geo_top5:.2f}")
    print(f"  Demand-weighted: ρ = {dem_rho:.4f}, top-5 = {dem_top5:.2f}")
    print(f"  Delta:           Δρ = {rho_delta:+.4f}, Δtop-5 = {top5_delta:+.2f}")

    print(f"\nInterpretation: {interpretation}")

    print("\n" + "=" * 60)

    return output


if __name__ == "__main__":
    main()
