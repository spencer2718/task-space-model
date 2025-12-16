#!/usr/bin/env python3
"""
Phase 0.7a: Shock Integration Test

Evaluates whether task-space geometry improves reallocation prediction
for AI-exposed occupations.

Steps:
1. Load CPS transitions, partition into train (2015-2019, 2022-2023) and holdout (2024)
2. Compute AIOE-geometry correlations (preliminary check)
3. Estimate conditional logit on training data using existing choice_model.py
4. Compute historical baseline from training data
5. Evaluate both on holdout
6. Write results to outputs/experiments/shock_integration_v070a.json

Usage:
    python scripts/run_shock_integration_v070a.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from task_space.validation.shock_integration import (
    get_aioe_by_soc_dataframe,
    map_aioe_to_census,
    compute_aioe_geometry_correlations,
    partition_transitions_by_exposure,
    compute_historical_baseline,
    compute_model_probabilities,
    evaluate_model_on_holdout,
    compute_verdict,
    ShockIntegrationResult,
)
from task_space.mobility.choice_model import (
    build_choice_dataset,
    fit_conditional_logit,
)


def load_transitions(path: str = "data/processed/mobility/verified_transitions.parquet") -> pd.DataFrame:
    """Load CPS transition data."""
    df = pd.read_parquet(path)
    # Extract year from YEARMONTH
    df["year"] = df["YEARMONTH"] // 100
    return df


def partition_by_year(
    df: pd.DataFrame,
    train_years: list,
    holdout_years: list,
) -> tuple:
    """Split transitions into train and holdout by year."""
    train = df[df["year"].isin(train_years)].copy()
    holdout = df[df["year"].isin(holdout_years)].copy()
    return train, holdout


def load_wasserstein_census():
    """Load Census-level Wasserstein distances."""
    data = np.load(".cache/artifacts/v1/mobility/d_wasserstein_census.npz")
    return data["distances"], list(data["occupation_codes"])


def load_institutional_census(census_codes: list):
    """
    Load/compute Census-level institutional distances.

    The raw d_inst_census.npz is at O*NET level (923 occs).
    We aggregate to Census level (447 occs) using the crosswalk.
    """
    from task_space.mobility.census_crosswalk import load_census_onet_crosswalk
    from task_space.mobility.institutional import build_institutional_distance_matrix

    # Build O*NET-level institutional distances
    inst_result = build_institutional_distance_matrix()
    d_inst_onet = inst_result.matrix
    onet_codes = inst_result.occupations

    # Load crosswalk
    xwalk = load_census_onet_crosswalk()

    # Build O*NET to index mapping
    onet_to_idx = {code: i for i, code in enumerate(onet_codes)}

    # Aggregate to Census level
    n_census = len(census_codes)
    d_inst_census = np.zeros((n_census, n_census))

    for ci, census_i in enumerate(census_codes):
        if census_i not in xwalk.census_to_onet:
            continue
        onet_list_i = xwalk.census_to_onet[census_i]
        valid_idx_i = [onet_to_idx[o] for o in onet_list_i if o in onet_to_idx]

        for cj, census_j in enumerate(census_codes):
            if census_j not in xwalk.census_to_onet:
                continue
            onet_list_j = xwalk.census_to_onet[census_j]
            valid_idx_j = [onet_to_idx[o] for o in onet_list_j if o in onet_to_idx]

            if not valid_idx_i or not valid_idx_j:
                continue

            # Aggregate pairwise distances using mean
            distances = []
            for oi in valid_idx_i:
                for oj in valid_idx_j:
                    distances.append(d_inst_onet[oi, oj])

            if distances:
                d_inst_census[ci, cj] = np.mean(distances)

    return d_inst_census


def main():
    print("=" * 60)
    print("Phase 0.7a: Shock Integration Test")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load data and partition
    # =========================================================================
    print("\n[1] Loading data...")

    # Load transitions
    transitions = load_transitions()
    print(f"    Total transitions: {len(transitions):,}")
    print(f"    Year range: {transitions['year'].min()}-{transitions['year'].max()}")

    # Partition: train = 2015-2019, 2022-2023; holdout = 2024
    train_years = list(range(2015, 2020)) + [2022, 2023]
    holdout_years = [2024]

    train_df, holdout_df = partition_by_year(transitions, train_years, holdout_years)
    print(f"    Train: {len(train_df):,} transitions ({train_years[0]}-{train_years[-1]})")
    print(f"    Holdout: {len(holdout_df):,} transitions ({holdout_years})")

    # Load distance matrices
    d_sem, census_codes = load_wasserstein_census()
    print(f"    Census occupations: {len(census_codes)}")
    print("    Computing institutional distances (aggregating from O*NET)...")
    d_inst = load_institutional_census(census_codes)
    print(f"    Institutional matrix computed: {d_inst.shape}")

    # Load AIOE
    aioe_soc = get_aioe_by_soc_dataframe(use_lm=True)
    print(f"    AIOE scores (SOC-6): {len(aioe_soc)}")

    aioe_census = map_aioe_to_census(aioe_soc)
    print(f"    AIOE scores (Census): {len(aioe_census)}")

    # =========================================================================
    # Step 2: Preliminary checks
    # =========================================================================
    print("\n[2] Preliminary checks...")

    correlations = compute_aioe_geometry_correlations(
        aioe_census, d_sem, census_codes
    )
    print(f"    AIOE coverage: {correlations['aioe_coverage']:.1%}")
    print(f"    AIOE-Wasserstein correlation: {correlations['aioe_wasserstein_corr']:.3f}" if correlations['aioe_wasserstein_corr'] else "    AIOE-Wasserstein correlation: N/A")

    # Check if aggregation to SOC-3 is needed
    aggregation = None
    if correlations["aioe_coverage"] < 0.80:
        print("    WARNING: AIOE coverage < 80%. Would need SOC-3 aggregation.")
        aggregation = "soc3"

    # =========================================================================
    # Step 3: Partition by AIOE exposure
    # =========================================================================
    print("\n[3] Partitioning by AI exposure...")

    exposed_holdout, unexposed_holdout = partition_transitions_by_exposure(
        holdout_df, aioe_census, quartile=0.75
    )
    print(f"    Exposed (top quartile AIOE): {len(exposed_holdout):,}")
    print(f"    Unexposed: {len(unexposed_holdout):,}")

    # =========================================================================
    # Step 4: Fit conditional logit on training data
    # =========================================================================
    print("\n[4] Fitting conditional logit on training data...")

    # Build choice dataset
    choice_df = build_choice_dataset(
        train_df,
        d_sem,
        d_inst,
        census_codes,
        n_alternatives=10,
        random_seed=42,
    )
    print(f"    Choice dataset: {len(choice_df):,} rows")
    print(f"    Unique transitions: {choice_df['transition_id'].nunique():,}")

    # Fit model
    result = fit_conditional_logit(choice_df)
    print(f"    alpha (semantic): {result.alpha:.3f} (t={result.alpha_t:.1f})")
    print(f"    beta (institutional): {result.beta:.3f} (t={result.beta_t:.1f})")
    print(f"    Log-likelihood: {result.log_likelihood:,.0f}")

    # =========================================================================
    # Step 5: Compute baselines and model probabilities
    # =========================================================================
    print("\n[5] Computing baselines...")

    # Historical baseline from training data
    historical_probs = compute_historical_baseline(train_df, census_codes)
    print(f"    Historical baseline computed (sum check: {historical_probs.sum(axis=1).mean():.3f})")

    # Geometry-based probabilities
    model_probs = compute_model_probabilities(
        train_df, d_sem, d_inst, census_codes,
        alpha=result.alpha, beta=result.beta
    )
    print(f"    Geometry model computed (sum check: {model_probs.sum(axis=1).mean():.3f})")

    # =========================================================================
    # Step 6: Evaluate on holdout
    # =========================================================================
    print("\n[6] Evaluating on holdout...")

    # Full holdout
    metrics_full = evaluate_model_on_holdout(
        model_probs, historical_probs, holdout_df, census_codes
    )
    print(f"    Full holdout evaluated: {metrics_full['n_evaluated']:,} transitions")

    # Exposed subset only
    metrics_exposed = evaluate_model_on_holdout(
        model_probs, historical_probs, exposed_holdout, census_codes
    )
    print(f"    Exposed holdout evaluated: {metrics_exposed['n_evaluated']:,} transitions")

    # =========================================================================
    # Step 7: Compute verdict
    # =========================================================================
    print("\n[7] Computing verdict...")

    verdict = compute_verdict(
        metrics_full["geometry_ll"],
        metrics_full["baseline_historical_ll"],
        metrics_full["baseline_uniform_ll"],
    )
    print(f"    Verdict: {verdict}")

    # =========================================================================
    # Step 8: Assemble and save results
    # =========================================================================
    print("\n[8] Saving results...")

    output = ShockIntegrationResult(
        version="0.7a.0",
        preliminary_checks={
            "aioe_coverage": correlations["aioe_coverage"],
            "aioe_wasserstein_corr": correlations["aioe_wasserstein_corr"],
        },
        sample_sizes={
            "train_n": int(len(train_df)),
            "holdout_n": int(len(holdout_df)),
            "exposed_holdout_n": int(len(exposed_holdout)),
        },
        metrics={
            "geometry_ll": metrics_full["geometry_ll"],
            "baseline_historical_ll": metrics_full["baseline_historical_ll"],
            "baseline_uniform_ll": metrics_full["baseline_uniform_ll"],
            "geometry_top5_acc": metrics_full["geometry_top5_acc"],
            "baseline_historical_top5_acc": metrics_full["baseline_historical_top5_acc"],
        },
        deltas={
            "geometry_vs_historical": metrics_full["geometry_ll"] - metrics_full["baseline_historical_ll"],
            "geometry_vs_uniform": metrics_full["geometry_ll"] - metrics_full["baseline_uniform_ll"],
        },
        verdict=verdict,
        aggregation=aggregation,
    )

    output_path = Path("outputs/experiments/shock_integration_v070a.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(str(output_path))
    print(f"    Saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"AIOE Coverage: {correlations['aioe_coverage']:.1%}")
    print(f"AIOE-Wasserstein Corr: {correlations['aioe_wasserstein_corr']:.3f}" if correlations['aioe_wasserstein_corr'] else "AIOE-Wasserstein Corr: N/A")
    print(f"\nSample Sizes:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Holdout: {len(holdout_df):,}")
    print(f"  Exposed holdout: {len(exposed_holdout):,}")
    print(f"\nLog-Likelihoods:")
    print(f"  Geometry:   {metrics_full['geometry_ll']:,.0f}")
    print(f"  Historical: {metrics_full['baseline_historical_ll']:,.0f}")
    print(f"  Uniform:    {metrics_full['baseline_uniform_ll']:,.0f}")
    print(f"\nDeltas:")
    print(f"  Geometry vs Historical: {metrics_full['geometry_ll'] - metrics_full['baseline_historical_ll']:+,.0f}")
    print(f"  Geometry vs Uniform:    {metrics_full['geometry_ll'] - metrics_full['baseline_uniform_ll']:+,.0f}")
    print(f"\nTop-5 Accuracy:")
    print(f"  Geometry:   {metrics_full['geometry_top5_acc']:.1%}")
    print(f"  Historical: {metrics_full['baseline_historical_top5_acc']:.1%}")
    print(f"\nVERDICT: {verdict}")
    print("=" * 60)

    return output


if __name__ == "__main__":
    main()
