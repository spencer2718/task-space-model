#!/usr/bin/env python
"""
Phase 0.7c-revised: Skill-Compatible Pathway Analysis

Identifies skill-compatible transition pathways for AI-exposed workers using
task-space geometry. Includes validation against 2024 holdout and honest
framing of limitations.

NOTE: This analysis identifies SKILL COMPATIBILITY, not predicted reallocation.
Institutional barriers are underweighted and demand-side constraints are ignored.

Outputs:
    - outputs/experiments/reallocation_v070c.json: Full results with validation
    - Console: Validation metrics and top destinations
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from task_space.validation.reallocation import (
    load_employment_by_census,
    load_occupation_names,
    get_exposed_occupations,
    compute_destination_probabilities,
    aggregate_reallocation_flows,
    compute_absorption_ranking,
    validate_against_holdout,
    flag_capacity_constraints,
    split_feasible_constrained,
    compute_validation_verdict,
    CREDENTIAL_GATED_OCCUPATIONS,
)
from task_space.validation.shock_integration import (
    get_aioe_by_soc_dataframe,
    map_aioe_to_census,
)
from task_space.mobility.census_crosswalk import load_census_onet_crosswalk


# =============================================================================
# Configuration
# =============================================================================

# Model coefficients from Phase 0.7a (fit on pre-2020 data)
GAMMA_SEM = 8.9346  # Semantic distance coefficient
GAMMA_INST = 0.1361  # Institutional distance coefficient

# Analysis parameters
OES_YEAR = 2023
AIOE_QUARTILE = 0.75  # Top 25% by AI exposure
CAPACITY_THRESHOLD = 0.5  # 50% absorption rate = constrained

OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "experiments" / "reallocation_v070c.json"


# =============================================================================
# Data Loading
# =============================================================================


def load_distance_matrices():
    """Load Wasserstein and institutional distance matrices at Census level."""
    cache_dir = Path(__file__).parent.parent / ".cache" / "artifacts" / "v1"

    # Load pre-aggregated Wasserstein at Census level
    wass_path = cache_dir / "mobility" / "d_wasserstein_census.npz"
    wass_data = np.load(wass_path)
    d_wass_census = wass_data["distances"]
    wass_codes = [int(c) for c in wass_data["occupation_codes"]]
    print(f"   Wasserstein matrix: {d_wass_census.shape}, {len(wass_codes)} Census codes")

    # Load institutional distance at O*NET level and aggregate to Census
    inst_path = cache_dir / "mobility" / "d_inst_census.npz"
    inst_data = np.load(inst_path, allow_pickle=True)
    d_inst_onet = inst_data["d_inst_matrix"]
    onet_codes = list(inst_data["occ_codes"])
    print(f"   Institutional matrix (O*NET): {d_inst_onet.shape}, {len(onet_codes)} O*NET codes")

    # Build O*NET to Census mapping
    xwalk = load_census_onet_crosswalk()
    xwalk_df = xwalk.crosswalk_df[xwalk.crosswalk_df["matched"] == True].copy()
    onet_to_census = dict(zip(xwalk_df["onet_soc"], xwalk_df["census_2010"]))

    # Aggregate institutional distance to Census level
    onet_idx = {o: i for i, o in enumerate(onet_codes)}

    n_census = len(wass_codes)
    d_inst_census = np.zeros((n_census, n_census))
    count_matrix = np.zeros((n_census, n_census))

    census_to_wass_idx = {c: i for i, c in enumerate(wass_codes)}

    for onet_i in onet_codes:
        if onet_i not in onet_to_census:
            continue
        census_i = int(onet_to_census[onet_i])
        if census_i not in census_to_wass_idx:
            continue
        i_census = census_to_wass_idx[census_i]
        i_onet = onet_idx[onet_i]

        for onet_j in onet_codes:
            if onet_j not in onet_to_census:
                continue
            census_j = int(onet_to_census[onet_j])
            if census_j not in census_to_wass_idx:
                continue
            j_census = census_to_wass_idx[census_j]
            j_onet = onet_idx[onet_j]

            d_inst_census[i_census, j_census] += d_inst_onet[i_onet, j_onet]
            count_matrix[i_census, j_census] += 1

    count_matrix = np.maximum(count_matrix, 1)
    d_inst_census /= count_matrix

    print(f"   Institutional matrix (Census): {d_inst_census.shape}")

    return d_wass_census, d_inst_census, wass_codes


def load_holdout_transitions():
    """Load 2024 holdout transitions for validation."""
    data_dir = Path(__file__).parent.parent / "data" / "processed" / "mobility"
    holdout_path = data_dir / "verified_transitions.parquet"

    if not holdout_path.exists():
        print(f"   WARNING: Holdout data not found at {holdout_path}")
        return None

    df = pd.read_parquet(holdout_path)

    # Filter to 2024 (post-training period)
    df["year"] = df["YEARMONTH"] // 100
    holdout = df[df["year"] >= 2024].copy()

    print(f"   Holdout transitions (2024+): {len(holdout)}")
    return holdout


# =============================================================================
# Main Analysis
# =============================================================================


def main():
    print("=" * 70)
    print("Phase 0.7c-revised: Skill-Compatible Pathway Analysis")
    print("=" * 70)

    # Load distance matrices
    print("\n1. Loading distance matrices...")
    d_wass, d_inst, census_codes = load_distance_matrices()

    # Load AIOE and employment data
    print("\n2. Loading AIOE and employment data...")
    aioe_df = get_aioe_by_soc_dataframe(use_lm=True)
    aioe_census_df = map_aioe_to_census(aioe_df)
    employment_df = load_employment_by_census(year=OES_YEAR)
    occupation_names = load_occupation_names()

    print(f"   AIOE scores at Census level: {len(aioe_census_df)}")
    print(f"   Employment data: {len(employment_df)} occupations")

    # Get exposed occupations
    print("\n3. Identifying exposed occupations...")
    exposed_df, aioe_threshold = get_exposed_occupations(
        aioe_census_df, employment_df, quartile=AIOE_QUARTILE
    )

    # Filter to Census codes in distance matrices
    census_set = set(census_codes)
    exposed_df = exposed_df[exposed_df["census_code"].isin(census_set)].copy()
    exposed_codes = list(exposed_df["census_code"])

    print(f"   AIOE threshold (Q{int(AIOE_QUARTILE*100)}): {aioe_threshold:.4f}")
    print(f"   Exposed occupations: {len(exposed_df)}")
    print(f"   Total exposed employment: {exposed_df['tot_emp'].sum():,.0f}")

    # Compute destination probabilities
    print("\n4. Computing destination probabilities...")
    print(f"   gamma_sem = {GAMMA_SEM:.4f}")
    print(f"   gamma_inst = {GAMMA_INST:.4f}")
    print(f"   gamma_inst / gamma_sem ratio = {GAMMA_INST / GAMMA_SEM:.4f}")

    prob_matrix = compute_destination_probabilities(
        d_wass, d_inst, census_codes,
        gamma_sem=GAMMA_SEM,
        gamma_inst=GAMMA_INST,
        exclude_self=True,
    )

    # Aggregate flows (hypothetical 100% displacement)
    print("\n5. Aggregating skill-compatible pathways...")
    flows_df = aggregate_reallocation_flows(
        prob_matrix, census_codes, exposed_df,
        displacement_rate=1.0,
    )

    # Compute absorption ranking
    absorption_df = compute_absorption_ranking(
        flows_df, employment_df, census_codes,
        occupation_names=occupation_names,
    )

    # Flag constraints
    absorption_df = flag_capacity_constraints(absorption_df, CAPACITY_THRESHOLD)

    # Split into feasible vs constrained
    feasible_df, constrained_df = split_feasible_constrained(absorption_df, CAPACITY_THRESHOLD)

    # Load holdout for validation
    print("\n6. Validating against 2024 holdout...")
    holdout_df = load_holdout_transitions()

    validation_result = None
    if holdout_df is not None and len(holdout_df) > 0:
        validation_result = validate_against_holdout(
            flows_df=flows_df,
            holdout_df=holdout_df,
            census_codes=census_codes,
            exposed_codes=exposed_codes,
            prob_matrix=prob_matrix,
            occupation_names=occupation_names,
        )

        # Compute verdict
        verdict = compute_validation_verdict(
            validation_result.get("spearman_correlation"),
            validation_result.get("top5_overlap"),
        )
        validation_result["verdict"] = verdict

        print(f"\n   === VALIDATION METRICS ===")
        print(f"   Observed transitions (exposed origins): {validation_result['n_observed_transitions']}")
        print(f"   KL Divergence: {validation_result.get('kl_divergence', 'N/A')}")
        print(f"   Spearman correlation: {validation_result.get('spearman_correlation', 'N/A')}")
        print(f"   Mean per-origin Spearman: {validation_result.get('mean_per_origin_spearman', 'N/A')}")
        print(f"   Top-5 overlap: {validation_result.get('top5_overlap', 'N/A')}")
        print(f"   Top-10 overlap: {validation_result.get('top10_overlap', 'N/A')}")
        print(f"   Verdict: {verdict}")

        print(f"\n   Observed top-5 destinations:")
        for i, dest in enumerate(validation_result.get("observed_top5_destinations", [])[:5]):
            print(f"      {i+1}. {dest}")

        print(f"\n   Predicted top-5 destinations:")
        for i, dest in enumerate(validation_result.get("predicted_top5_destinations", [])[:5]):
            print(f"      {i+1}. {dest}")
    else:
        print("   No holdout data available for validation")

    # Display results
    total_displaced = exposed_df["tot_emp"].sum()

    print("\n" + "=" * 70)
    print("FEASIBLE SKILL-COMPATIBLE DESTINATIONS (unconstrained)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Occupation':<40} {'Inflow':>12} {'AbsRate':>8} {'Constraint':>12}")
    print("-" * 77)

    for idx, (_, row) in enumerate(feasible_df.head(15).iterrows()):
        name = str(row.get("occupation_name", "Unknown"))[:38]
        inflow = row["total_absorption"]
        rate = row["absorption_rate"] if pd.notna(row["absorption_rate"]) else 0
        constraint = row["constraint_type"]
        print(f"{idx+1:<5} {name:<40} {inflow:>12,.0f} {rate:>7.1%} {constraint:>12}")

    print("\n" + "=" * 70)
    print("CONSTRAINED DESTINATIONS (capacity/credential limited)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Occupation':<40} {'Inflow':>12} {'AbsRate':>8} {'Constraint':>12}")
    print("-" * 77)

    for idx, (_, row) in enumerate(constrained_df.head(15).iterrows()):
        name = str(row.get("occupation_name", "Unknown"))[:38]
        inflow = row["total_absorption"]
        rate = row["absorption_rate"] if pd.notna(row["absorption_rate"]) else 0
        constraint = row["constraint_type"]
        print(f"{idx+1:<5} {name:<40} {inflow:>12,.0f} {rate:>7.1%} {constraint:>12}")

    # Build output JSON
    print("\n7. Building results...")

    def build_destination_list(df, top_n=20):
        """Build structured destination list for JSON output."""
        destinations = []
        for _, row in df.head(top_n).iterrows():
            destinations.append({
                "rank": len(destinations) + 1,
                "census_code": int(row["census_code"]),
                "occupation_name": str(row.get("occupation_name", "Unknown")),
                "expected_inflow": float(row["total_absorption"]),
                "pct_of_displaced": float(row["total_absorption"] / total_displaced),
                "current_employment": float(row["current_emp"]) if pd.notna(row["current_emp"]) else None,
                "absorption_rate": float(row["absorption_rate"]) if pd.notna(row["absorption_rate"]) else None,
                "constraint_type": row["constraint_type"],
                "is_constrained": bool(row["is_constrained"]),
            })
        return destinations

    feasible_destinations = build_destination_list(feasible_df, 20)
    constrained_destinations = build_destination_list(constrained_df, 20)

    # Limitations acknowledgment
    limitations = {
        "institutional_barriers_underweighted": True,
        "reason": "Estimated from completed transitions; blocked attempts unobserved",
        "gamma_inst_gamma_sem_ratio": float(GAMMA_INST / GAMMA_SEM),
        "demand_side_ignored": True,
        "capacity_constraints_post_hoc": True,
        "interpretation": "Rankings reflect skill transferability, not realized reallocation",
        "credential_gated_occupations_flagged": len(CREDENTIAL_GATED_OCCUPATIONS),
    }

    # Final output
    result = {
        "version": "0.7c.1",
        "parameters": {
            "gamma_sem": GAMMA_SEM,
            "gamma_inst": GAMMA_INST,
            "oes_year": OES_YEAR,
            "aioe_quartile": AIOE_QUARTILE,
            "aioe_threshold": float(aioe_threshold),
            "capacity_threshold": CAPACITY_THRESHOLD,
        },
        "exposed_summary": {
            "n_exposed_occupations": len(exposed_df),
            "total_exposed_employment": float(exposed_df["tot_emp"].sum()),
            "mean_aioe_exposed": float(exposed_df["aioe_score"].mean()),
        },
        "validation_2024": validation_result,
        "feasible_destinations": feasible_destinations,
        "constrained_destinations": constrained_destinations,
        "flow_statistics": {
            "total_pathways": len(flows_df),
            "n_feasible_destinations": len(feasible_df),
            "n_constrained_destinations": len(constrained_df),
            "feasible_share_of_inflow": float(feasible_df["total_absorption"].sum() / total_displaced),
            "constrained_share_of_inflow": float(constrained_df["total_absorption"].sum() / total_displaced),
        },
        "limitations": limitations,
        "verdict": validation_result.get("verdict", "no_validation") if validation_result else "no_validation",
    }

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n   Saved to: {OUTPUT_PATH}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Exposed occupations: {len(exposed_df)}")
    print(f"Total exposed employment: {total_displaced:,.0f}")
    print(f"Feasible destinations: {len(feasible_df)}")
    print(f"Constrained destinations: {len(constrained_df)}")
    print(f"Feasible share: {result['flow_statistics']['feasible_share_of_inflow']:.1%}")
    if validation_result:
        print(f"\nValidation verdict: {result['verdict']}")

    print("\n" + "=" * 70)
    print("KEY QUESTION: Does geometry predict DIRECTION correctly?")
    print("=" * 70)
    if validation_result and validation_result.get("spearman_correlation") is not None:
        spearman = validation_result["spearman_correlation"]
        if spearman > 0.2:
            print(f"""
Spearman = {spearman:.3f} suggests geometry captures real skill compatibility.
The model predicts which occupations are relatively more/less accessible,
even though:
  - Magnitude is uncalibrated to demand
  - Institutional barriers are underweighted
  - Capacity constraints are ignored

This supports the paper claim: "Wasserstein geometry identifies skill-compatible
transition pathways" rather than "predicts realized reallocation."
""")
        else:
            print(f"""
Spearman = {spearman:.3f} is weak. The geometry may not capture the relevant
dimensions of skill transferability for AI-exposed occupations specifically.
This could be because:
  - AI exposure affects occupations differently than historical shocks
  - The 2024 holdout doesn't yet show AI-driven transitions
  - Revealed preference from 2015-2019 doesn't generalize
""")
    else:
        print("Insufficient validation data to assess directional accuracy.")


if __name__ == "__main__":
    main()
