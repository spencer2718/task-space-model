#!/usr/bin/env python3
"""
D3: Demand Decomposition on Centroid (v0.7.12.3)

Adapts run_demand_decomposition_v0703b.py to use centroid (cosine embedding)
instead of Wasserstein distances. All other logic identical.

Usage:
    python scripts/run_demand_decomposition_centroid_v07123.py
"""

from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from task_space.mobility.io import (
    get_holdout_transitions,
    load_centroid_census,
)
from task_space.utils.experiments import save_experiment_output


# =============================================================================
# Data Loading (BLS — identical to v0703b)
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
# Analysis Functions (identical to v0703b, just with d_sem = centroid)
# =============================================================================

def test_full_flow_model(
    holdout_df: pd.DataFrame,
    d_sem: np.ndarray,
    census_codes: List[int],
    openings_by_census: Dict[int, float],
    median_openings: float,
) -> Dict:
    """Test flow model: outflow x openings x (1/distance)."""
    code_to_idx = {c: i for i, c in enumerate(census_codes)}
    origin_outflows = holdout_df.groupby("origin_occ").size().to_dict()
    all_destinations = set(holdout_df["dest_occ"].unique())

    # Full flow model
    predicted_inflows = {}
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
            flow = outflow * openings / d
            predicted_inflows[dest] = predicted_inflows.get(dest, 0) + flow

    observed_inflows = holdout_df.groupby("dest_occ").size().to_dict()
    common = set(predicted_inflows.keys()) & set(observed_inflows.keys())

    if len(common) < 3:
        return {"error": "Too few common destinations"}

    pred_vec = [predicted_inflows[d] for d in common]
    obs_vec = [observed_inflows[d] for d in common]
    spearman, _ = stats.spearmanr(pred_vec, obs_vec)

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
                geo_scores[dest] = geo_scores.get(dest, 0) + 1 / d

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
        "formula": "outflow x openings x (1/distance)",
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
        origin_df = holdout_df[holdout_df["origin_occ"] == origin]
        observed_counts = origin_df.groupby("dest_occ").size().to_dict()

        if len(observed_counts) < 3:
            continue

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
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=== D3: Demand Decomposition on Centroid (v0.7.12.3) ===")

    # Load data
    print("[1] Loading data...")
    bls_df = load_bls_projections()
    soc_to_census, census_to_soc = load_soc_to_census_crosswalk()
    d_sem, census_codes = load_centroid_census()  # <-- CENTROID instead of Wasserstein
    holdout_df = get_holdout_transitions()

    # Map openings to Census
    openings_by_census = {}
    for _, row in bls_df.iterrows():
        soc = row["soc_code"]
        if soc in soc_to_census:
            census = soc_to_census[soc]
            openings_by_census[census] = openings_by_census.get(census, 0) + row["openings"]
    median_openings = np.median(list(openings_by_census.values()))

    print(f"  Holdout transitions: {len(holdout_df):,}")
    print(f"  Census codes with openings: {len(openings_by_census)}")

    # Full flow model
    print("[2] Testing full flow model...")
    flow_model = test_full_flow_model(
        holdout_df, d_sem, census_codes, openings_by_census, median_openings
    )
    print(f"  Geometry-only ρ: {flow_model.get('geometry_only_spearman')}")
    print(f"  Demand-only ρ: {flow_model.get('demand_only_spearman')}")
    print(f"  Full flow ρ: {flow_model.get('full_flow_spearman')}")

    # Per-origin validation
    print("[3] Validating per-origin Spearman...")
    per_origin = validate_per_origin_spearman(holdout_df, d_sem, census_codes)
    print(f"  Mean per-origin ρ: {per_origin['mean_per_origin_spearman']}")
    print(f"  N origins evaluated: {per_origin['n_origins_evaluated']}")

    # Summary
    dem = flow_model["demand_only_spearman"]
    po = per_origin["mean_per_origin_spearman"]
    geo = flow_model["geometry_only_spearman"]
    print(f"\nDemand ρ: {dem} (was 0.7978)")
    print(f"Per-origin ρ: {po} (was 0.3165)")
    print(f"Aggregate ρ: {geo} (was 0.0432)")

    # Assemble output
    output = {
        "experiment": "demand_decomposition_centroid_v07123",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "distance_metric": "cosine_embed (centroid)",
        "full_flow_model": flow_model,
        "per_origin_validation": per_origin,
        "comparison_to_wasserstein": {
            "wass_geometry_only": 0.0432,
            "wass_demand_only": 0.7978,
            "wass_per_origin": 0.3165,
            "source": "demand_probe_decomposition_v0703b.json",
        },
    }

    output_path = save_experiment_output("demand_decomposition_centroid_v07123", output)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
