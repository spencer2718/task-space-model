#!/usr/bin/env python3
"""
Experiment v0.7.0.3b: Demand Probe Decomposition

Decomposes prediction problem to understand why geometry negatively
correlates with aggregate inflows.

Tests:
1. Why is Census 2050 predicted #1?
2. Origin-side outflow patterns
3. Full flow model: outflow × openings × (1/distance)
4. Per-origin Spearman validation

Usage:
    python scripts/run_demand_decomposition_v0703b.py
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Canonical imports from task_space
from task_space.mobility.io import (
    load_transitions,
    get_holdout_transitions,
    load_wasserstein_census,
)
from task_space.utils.experiments import save_experiment_output


# =============================================================================
# Data Loading (script-specific - BLS data)
# =============================================================================

def load_bls_projections() -> pd.DataFrame:
    """Load BLS Employment Projections (Table 1.10)."""
    xl = pd.ExcelFile("data/external/bls_projections/occupation.xlsx")
    df = pd.read_excel(xl, sheet_name="Table 1.10", header=1)
    detailed = df[df["Occupation type"] == "Line item"].copy()
    detailed = detailed.rename(columns={
        "2024 National Employment Matrix code": "soc_code",
        "2024 National Employment Matrix title": "title",
        "Occupational openings, 2024–34 annual average": "openings",
    })
    return detailed[["soc_code", "title", "openings"]].copy()


def load_soc_to_census_crosswalk() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load crosswalks in both directions."""
    xwalk = pd.read_csv(".cache/artifacts/v1/mobility/onet_to_census_improved.csv")
    soc_census = xwalk[["soc_6digit", "census_2010"]].drop_duplicates().dropna()

    soc_to_census = {}
    census_to_soc = {}
    for _, row in soc_census.iterrows():
        soc = row["soc_6digit"]
        census = int(row["census_2010"])
        if soc not in soc_to_census:
            soc_to_census[soc] = census
        if census not in census_to_soc:
            census_to_soc[census] = soc

    return soc_to_census, census_to_soc


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_occupation_2050(
    d_sem: np.ndarray,
    census_codes: List[int],
    holdout_df: pd.DataFrame,
    bls_df: pd.DataFrame,
    census_to_soc: Dict[int, str],
) -> Dict:
    """Analyze why Census 2050 is predicted #1."""
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    if 2050 not in code_to_idx:
        return {"error": "Census 2050 not in distance matrix"}

    idx_2050 = code_to_idx[2050]

    # Get SOC and title
    soc = census_to_soc.get(2050, "Unknown")
    bls_match = bls_df[bls_df["soc_code"] == soc]
    title = bls_match["title"].iloc[0].strip() if len(bls_match) > 0 else "Unknown"
    openings = bls_match["openings"].iloc[0] if len(bls_match) > 0 else 0

    # Compute average distance FROM all origins to 2050
    origins_in_holdout = holdout_df["origin_occ"].unique()
    distances_to_2050 = []
    for origin in origins_in_holdout:
        if origin in code_to_idx:
            d = d_sem[code_to_idx[origin], idx_2050]
            if d > 0:  # Exclude self-transitions
                distances_to_2050.append(d)

    avg_distance = np.mean(distances_to_2050) if distances_to_2050 else 0

    # Compute observed rank (by inflow count)
    dest_counts = holdout_df.groupby("dest_occ").size().sort_values(ascending=False)
    if 2050 in dest_counts.index:
        observed_rank = list(dest_counts.index).index(2050) + 1
        observed_count = dest_counts[2050]
    else:
        observed_rank = None
        observed_count = 0

    return {
        "census_code": 2050,
        "soc_code": soc,
        "title": title,
        "bls_openings": float(openings),
        "avg_distance_from_origins": round(float(avg_distance), 4),
        "observed_rank": observed_rank,
        "observed_inflow_count": int(observed_count),
        "interpretation": "Low average distance = central in task space. But low openings = not high demand."
    }


def analyze_origin_outflow(
    holdout_df: pd.DataFrame,
    census_to_soc: Dict[int, str],
    bls_df: pd.DataFrame,
) -> Dict:
    """Analyze origin-side outflow patterns."""
    # Count outflows by origin
    origin_counts = holdout_df.groupby("origin_occ").size().sort_values(ascending=False)

    # Top 10 origins
    top10 = []
    for origin, count in origin_counts.head(10).items():
        soc = census_to_soc.get(origin, "Unknown")
        bls_match = bls_df[bls_df["soc_code"] == soc]
        title = bls_match["title"].iloc[0].strip() if len(bls_match) > 0 else "Unknown"
        top10.append({
            "census_code": int(origin),
            "soc_code": soc,
            "title": title[:50],
            "outflow_count": int(count),
        })

    return {
        "top10_origins": top10,
        "total_origins": len(origin_counts),
        "max_outflow": int(origin_counts.max()),
        "median_outflow": float(origin_counts.median()),
    }


def test_full_flow_model(
    holdout_df: pd.DataFrame,
    d_sem: np.ndarray,
    census_codes: List[int],
    openings_by_census: Dict[int, float],
    median_openings: float,
) -> Dict:
    """Test flow model: outflow × openings × (1/distance)."""
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    # Compute origin outflows
    origin_outflows = holdout_df.groupby("origin_occ").size().to_dict()

    # Predict flows for each origin-destination pair
    predicted_inflows = {}
    all_destinations = set(holdout_df["dest_occ"].unique())

    for origin, outflow in origin_outflows.items():
        if origin not in code_to_idx:
            continue
        origin_idx = code_to_idx[origin]

        for dest in all_destinations:
            if dest not in code_to_idx or dest == origin:
                continue
            dest_idx = code_to_idx[dest]

            d = d_sem[origin_idx, dest_idx]
            if d <= 0:
                continue

            openings = openings_by_census.get(dest, median_openings)

            # Flow = outflow × openings × (1/distance)
            flow = outflow * openings / d

            predicted_inflows[dest] = predicted_inflows.get(dest, 0) + flow

    # Compare to observed
    observed_inflows = holdout_df.groupby("dest_occ").size().to_dict()

    # Get common destinations
    common = set(predicted_inflows.keys()) & set(observed_inflows.keys())

    if len(common) < 3:
        return {"error": "Too few common destinations"}

    pred_vec = [predicted_inflows[d] for d in common]
    obs_vec = [observed_inflows[d] for d in common]

    spearman, _ = stats.spearmanr(pred_vec, obs_vec)

    # Also compute geometry-only and demand-only for comparison
    # Geometry-only: sum(1/d) over all origins
    geo_scores = {}
    for origin in origin_outflows.keys():
        if origin not in code_to_idx:
            continue
        origin_idx = code_to_idx[origin]
        for dest in all_destinations:
            if dest not in code_to_idx or dest == origin:
                continue
            dest_idx = code_to_idx[dest]
            d = d_sem[origin_idx, dest_idx]
            if d > 0:
                geo_scores[dest] = geo_scores.get(dest, 0) + 1/d

    geo_common = set(geo_scores.keys()) & set(observed_inflows.keys())
    geo_vec = [geo_scores[d] for d in geo_common]
    obs_geo_vec = [observed_inflows[d] for d in geo_common]
    geo_spearman, _ = stats.spearmanr(geo_vec, obs_geo_vec)

    # Demand-only: openings
    dem_common = set(openings_by_census.keys()) & set(observed_inflows.keys())
    dem_vec = [openings_by_census[d] for d in dem_common]
    obs_dem_vec = [observed_inflows[d] for d in dem_common]
    dem_spearman, _ = stats.spearmanr(dem_vec, obs_dem_vec)

    return {
        "full_flow_spearman": round(float(spearman), 4),
        "geometry_only_spearman": round(float(geo_spearman), 4),
        "demand_only_spearman": round(float(dem_spearman), 4),
        "n_destinations_compared": len(common),
        "formula": "outflow × openings × (1/distance)",
    }


def characterize_destinations(
    observed_top5: List[int],
    predicted_top5: List[int],
    d_sem: np.ndarray,
    census_codes: List[int],
    holdout_df: pd.DataFrame,
    openings_by_census: Dict[int, float],
    census_to_soc: Dict[int, str],
    bls_df: pd.DataFrame,
) -> Dict:
    """Compare observed vs predicted top-5 destinations."""
    code_to_idx = {c: i for i, c in enumerate(census_codes)}
    origins_in_holdout = [o for o in holdout_df["origin_occ"].unique() if o in code_to_idx]

    def analyze_dest(census_code: int) -> Dict:
        soc = census_to_soc.get(census_code, "Unknown")
        bls_match = bls_df[bls_df["soc_code"] == soc]
        title = bls_match["title"].iloc[0].strip() if len(bls_match) > 0 else "Unknown"

        openings = openings_by_census.get(census_code, 0)

        # Average distance from all origins
        if census_code in code_to_idx:
            dest_idx = code_to_idx[census_code]
            distances = [d_sem[code_to_idx[o], dest_idx] for o in origins_in_holdout
                        if d_sem[code_to_idx[o], dest_idx] > 0]
            avg_dist = np.mean(distances) if distances else 0
        else:
            avg_dist = None

        # Observed inflow count
        dest_counts = holdout_df.groupby("dest_occ").size()
        inflow = dest_counts.get(census_code, 0)

        return {
            "census_code": census_code,
            "soc_code": soc,
            "title": title[:50],
            "bls_openings": float(openings),
            "avg_distance_from_origins": round(float(avg_dist), 4) if avg_dist else None,
            "observed_inflow": int(inflow),
        }

    observed_analysis = [analyze_dest(c) for c in observed_top5]
    predicted_analysis = [analyze_dest(c) for c in predicted_top5]

    # Summarize distinguishing features
    obs_avg_openings = np.mean([d["bls_openings"] for d in observed_analysis])
    pred_avg_openings = np.mean([d["bls_openings"] for d in predicted_analysis])

    obs_avg_dist = np.mean([d["avg_distance_from_origins"] for d in observed_analysis
                           if d["avg_distance_from_origins"]])
    pred_avg_dist = np.mean([d["avg_distance_from_origins"] for d in predicted_analysis
                            if d["avg_distance_from_origins"]])

    return {
        "observed_top5": observed_analysis,
        "predicted_top5": predicted_analysis,
        "summary": {
            "observed_avg_openings": round(obs_avg_openings, 1),
            "predicted_avg_openings": round(pred_avg_openings, 1),
            "observed_avg_distance": round(obs_avg_dist, 4),
            "predicted_avg_distance": round(pred_avg_dist, 4),
        },
        "distinguishing_features": (
            f"Observed top-5 have {obs_avg_openings/pred_avg_openings:.1f}× more openings "
            f"but are {obs_avg_dist/pred_avg_dist:.1f}× further on average. "
            "Geometry favors centrality; flows favor high-demand destinations."
        )
    }


def validate_per_origin_spearman(
    holdout_df: pd.DataFrame,
    d_sem: np.ndarray,
    census_codes: List[int],
) -> Dict:
    """Replicate v0.7.0c per-origin Spearman methodology."""
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    origins = holdout_df["origin_occ"].unique()
    per_origin_rhos = []

    for origin in origins:
        if origin not in code_to_idx:
            continue

        origin_idx = code_to_idx[origin]

        # Get observed destination distribution for this origin
        origin_df = holdout_df[holdout_df["origin_occ"] == origin]
        observed_counts = origin_df.groupby("dest_occ").size().to_dict()

        if len(observed_counts) < 3:
            continue

        # Get geometry-based predicted ranking (1/distance)
        predicted_scores = {}
        for dest in observed_counts.keys():
            if dest in code_to_idx and dest != origin:
                d = d_sem[origin_idx, code_to_idx[dest]]
                if d > 0:
                    predicted_scores[dest] = 1.0 / d

        common = set(observed_counts.keys()) & set(predicted_scores.keys())
        if len(common) < 3:
            continue

        obs_vec = [observed_counts[d] for d in common]
        pred_vec = [predicted_scores[d] for d in common]

        rho, _ = stats.spearmanr(pred_vec, obs_vec)
        if not np.isnan(rho):
            per_origin_rhos.append(rho)

    mean_rho = np.mean(per_origin_rhos) if per_origin_rhos else 0

    return {
        "mean_per_origin_spearman": round(float(mean_rho), 4),
        "n_origins_evaluated": len(per_origin_rhos),
        "expected_from_v070c": 0.43,
        "matches_v070c": abs(mean_rho - 0.43) < 0.1,
        "interpretation": (
            "Per-origin ρ measures pathway accuracy: given origin A, do we rank "
            "destinations correctly? Aggregate ρ measures total flow prediction."
        )
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Experiment v0.7.0.3b: Demand Probe Decomposition")
    print("=" * 60)

    # Load all data
    print("\n[1] Loading data...")
    bls_df = load_bls_projections()
    soc_to_census, census_to_soc = load_soc_to_census_crosswalk()
    d_sem, census_codes = load_wasserstein_census()
    holdout_df = get_holdout_transitions()

    # Map openings to Census
    openings_by_census = {}
    for _, row in bls_df.iterrows():
        soc = row["soc_code"]
        if soc in soc_to_census:
            census = soc_to_census[soc]
            openings_by_census[census] = openings_by_census.get(census, 0) + row["openings"]
    median_openings = np.median(list(openings_by_census.values()))

    print(f"    Holdout transitions: {len(holdout_df):,}")
    print(f"    Census codes with openings: {len(openings_by_census)}")

    # =========================================================================
    # Task 1: Analyze occupation 2050
    # =========================================================================
    print("\n[2] Analyzing occupation 2050...")
    occ_2050 = analyze_occupation_2050(
        d_sem, census_codes, holdout_df, bls_df, census_to_soc
    )
    print(f"    Title: {occ_2050.get('title')}")
    print(f"    BLS openings: {occ_2050.get('bls_openings'):.1f}k")
    print(f"    Avg distance from origins: {occ_2050.get('avg_distance_from_origins'):.4f}")
    print(f"    Observed rank: {occ_2050.get('observed_rank')}")
    print(f"    Observed inflow: {occ_2050.get('observed_inflow_count')}")

    # =========================================================================
    # Task 2: Origin outflow analysis
    # =========================================================================
    print("\n[3] Analyzing origin outflows...")
    origin_analysis = analyze_origin_outflow(holdout_df, census_to_soc, bls_df)
    print(f"    Total origins: {origin_analysis['total_origins']}")
    print(f"    Max outflow: {origin_analysis['max_outflow']}")
    print("    Top 5 origins by outflow:")
    for o in origin_analysis["top10_origins"][:5]:
        print(f"      {o['census_code']}: {o['title'][:40]} ({o['outflow_count']})")

    # =========================================================================
    # Task 3: Full flow model
    # =========================================================================
    print("\n[4] Testing full flow model...")
    flow_model = test_full_flow_model(
        holdout_df, d_sem, census_codes, openings_by_census, median_openings
    )
    print(f"    Geometry-only ρ: {flow_model.get('geometry_only_spearman')}")
    print(f"    Demand-only ρ: {flow_model.get('demand_only_spearman')}")
    print(f"    Full flow ρ: {flow_model.get('full_flow_spearman')}")

    # =========================================================================
    # Task 4: Characterize destinations
    # =========================================================================
    print("\n[5] Characterizing destinations...")
    observed_top5 = list(holdout_df.groupby("dest_occ").size().nlargest(5).index)
    predicted_top5 = [2050, 4110, 20, 4120, 4130]  # From v0.7.0.3

    dest_comparison = characterize_destinations(
        observed_top5, predicted_top5, d_sem, census_codes, holdout_df,
        openings_by_census, census_to_soc, bls_df
    )
    print("\n    Observed top-5:")
    for d in dest_comparison["observed_top5"]:
        print(f"      {d['title'][:35]:35} openings={d['bls_openings']:>6.1f}k dist={d['avg_distance_from_origins']:.3f}")
    print("\n    Predicted top-5:")
    for d in dest_comparison["predicted_top5"]:
        print(f"      {d['title'][:35]:35} openings={d['bls_openings']:>6.1f}k dist={d['avg_distance_from_origins']:.3f}")

    # =========================================================================
    # Task 5: Per-origin Spearman validation
    # =========================================================================
    print("\n[6] Validating per-origin Spearman...")
    per_origin = validate_per_origin_spearman(holdout_df, d_sem, census_codes)
    print(f"    Mean per-origin ρ: {per_origin['mean_per_origin_spearman']}")
    print(f"    Expected from v0.7.0c: {per_origin['expected_from_v070c']}")
    print(f"    Matches: {per_origin['matches_v070c']}")

    # =========================================================================
    # Assemble output
    # =========================================================================
    print("\n[7] Saving results...")

    output = {
        "experiment": "demand_probe_decomposition_v0703b",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "occupation_2050": occ_2050,
        "origin_outflow": origin_analysis,
        "full_flow_model": flow_model,
        "destination_comparison": dest_comparison,
        "per_origin_validation": per_origin,
    }

    output_path = save_experiment_output("demand_probe_decomposition_v0703b", output)
    print(f"    Saved to: {output_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("DECOMPOSITION SUMMARY")
    print("=" * 60)

    print(f"\n1. Census 2050 Mystery:")
    print(f"   '{occ_2050.get('title')}' - religious education director")
    print(f"   Low openings ({occ_2050.get('bls_openings'):.1f}k) but central in task space")
    print(f"   Observed rank: {occ_2050.get('observed_rank')} (actual inflows: {occ_2050.get('observed_inflow_count')})")

    print(f"\n2. Flow Model Comparison:")
    print(f"   Geometry-only: ρ = {flow_model.get('geometry_only_spearman')}")
    print(f"   Demand-only:   ρ = {flow_model.get('demand_only_spearman')}")
    print(f"   Full flow:     ρ = {flow_model.get('full_flow_spearman')}")

    print(f"\n3. Destination Characteristics:")
    print(f"   Observed avg openings:  {dest_comparison['summary']['observed_avg_openings']:.1f}k")
    print(f"   Predicted avg openings: {dest_comparison['summary']['predicted_avg_openings']:.1f}k")
    print(f"   Ratio: {dest_comparison['summary']['observed_avg_openings']/dest_comparison['summary']['predicted_avg_openings']:.1f}×")

    print(f"\n4. Per-Origin vs Aggregate:")
    print(f"   Per-origin ρ (v0.7.0c method): {per_origin['mean_per_origin_spearman']}")
    print(f"   Aggregate ρ (v0.7.0.3 method): {flow_model.get('geometry_only_spearman')}")
    print(f"   Interpretation: {per_origin['interpretation']}")

    print("\n" + "=" * 60)

    return output


if __name__ == "__main__":
    main()
