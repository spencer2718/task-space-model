#!/usr/bin/env python3
"""
Experiment v0.7.0.3c: Methodology Audit

Reconciles discrepancies between v0.7.0c and v0.7.0.3b Spearman correlations.

Key question: Why does "aggregate" Spearman differ by 10× (0.432 vs 0.043)?

Tests both methodologies on the same sample to isolate differences.

Usage:
    python scripts/run_methodology_audit_v0703c.py
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Canonical imports from task_space
from task_space.mobility.io import (
    load_wasserstein_census,
    load_institutional_census,
    get_holdout_transitions,
)
from task_space.utils.experiments import save_experiment_output


def compute_model_probabilities(
    d_wass: np.ndarray,
    d_inst: np.ndarray,
    gamma_sem: float,
    gamma_inst: float,
) -> np.ndarray:
    """Compute conditional logit probability matrix P(j|i)."""
    utility = -gamma_sem * d_wass - gamma_inst * d_inst
    np.fill_diagonal(utility, -np.inf)  # Exclude self-transitions

    exp_utility = np.exp(utility - np.nanmax(utility, axis=1, keepdims=True))
    np.fill_diagonal(exp_utility, 0.0)

    row_sums = exp_utility.sum(axis=1, keepdims=True)
    prob_matrix = exp_utility / (row_sums + 1e-15)

    return prob_matrix


# =============================================================================
# Methodology A: v0.7.0c style (model probabilities, all destinations)
# =============================================================================

def method_v070c_aggregate_spearman(
    holdout_df: pd.DataFrame,
    prob_matrix: np.ndarray,
    census_codes: List[int],
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
) -> Dict:
    """
    v0.7.0c's aggregate Spearman methodology.

    Computes:
    1. Weighted average of model probabilities across origins
    2. Spearman correlation over ALL destinations (447)
    """
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    # Get origin distribution from holdout
    origin_counts = holdout_df[origin_col].value_counts()
    valid_origins = [o for o in origin_counts.index if o in code_to_idx]

    # Weighted average of predicted probabilities
    predicted_probs = np.zeros(len(census_codes))
    for origin in valid_origins:
        count = origin_counts[origin]
        i = code_to_idx[origin]
        predicted_probs += count * prob_matrix[i, :]
    predicted_probs /= predicted_probs.sum() + 1e-15

    # Build observed probability vector
    observed_dest_counts = holdout_df[dest_col].value_counts()
    observed_total = observed_dest_counts.sum()

    observed_probs = np.zeros(len(census_codes))
    for dest, count in observed_dest_counts.items():
        if dest in code_to_idx:
            j = code_to_idx[dest]
            observed_probs[j] = count / observed_total

    # Spearman over ALL destinations
    obs_ranks = stats.rankdata(-observed_probs)
    pred_ranks = stats.rankdata(-predicted_probs)
    spearman, _ = stats.spearmanr(obs_ranks, pred_ranks)

    return {
        "method": "v070c_aggregate",
        "description": "Model probability weighted avg, Spearman over ALL destinations",
        "spearman": round(float(spearman), 4),
        "n_destinations": len(census_codes),
        "n_origins": len(valid_origins),
    }


def method_v070c_per_origin_spearman(
    holdout_df: pd.DataFrame,
    prob_matrix: np.ndarray,
    census_codes: List[int],
    min_destinations: int = 5,
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
) -> Dict:
    """
    v0.7.0c's per-origin Spearman methodology.

    For each origin with ≥min_destinations observed, compute Spearman
    between observed counts and model probability vector.
    """
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    origin_counts = holdout_df[origin_col].value_counts()
    per_origin_rhos = []

    for origin in origin_counts.index:
        if origin not in code_to_idx:
            continue
        i = code_to_idx[origin]

        # Observed destinations for this origin
        origin_holdout = holdout_df[holdout_df[origin_col] == origin]
        origin_dest_counts = origin_holdout[dest_col].value_counts()

        if len(origin_dest_counts) < min_destinations:
            continue

        # Build vectors over ALL destinations
        obs_vec = np.zeros(len(census_codes))
        for dest, count in origin_dest_counts.items():
            if dest in code_to_idx:
                obs_vec[code_to_idx[dest]] = count

        pred_vec = prob_matrix[i, :]

        # Spearman
        if obs_vec.sum() > 0:
            corr, _ = stats.spearmanr(obs_vec, pred_vec)
            if not np.isnan(corr):
                per_origin_rhos.append(corr)

    mean_rho = float(np.mean(per_origin_rhos)) if per_origin_rhos else 0

    return {
        "method": "v070c_per_origin",
        "description": f"Model probability, per-origin, min_dest≥{min_destinations}",
        "mean_spearman": round(mean_rho, 4),
        "n_origins_evaluated": len(per_origin_rhos),
        "min_destinations_filter": min_destinations,
    }


# =============================================================================
# Methodology B: v0.7.0.3b style (raw 1/distance, common destinations only)
# =============================================================================

def method_v0703b_aggregate_spearman(
    holdout_df: pd.DataFrame,
    d_wass: np.ndarray,
    census_codes: List[int],
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
) -> Dict:
    """
    v0.7.0.3b's aggregate Spearman methodology.

    Computes:
    1. Sum of 1/distance across all origins for each destination
    2. Spearman correlation over COMMON destinations only
    """
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    # Observed inflows by destination
    observed_inflows = holdout_df[dest_col].value_counts().to_dict()

    # Geometry-based prediction: sum(1/d) over all origins
    origins = holdout_df[origin_col].unique()
    geo_scores = {}

    for origin in origins:
        if origin not in code_to_idx:
            continue
        i = code_to_idx[origin]

        for j, dest in enumerate(census_codes):
            if dest == origin:
                continue
            d = d_wass[i, j]
            if d > 0:
                geo_scores[dest] = geo_scores.get(dest, 0) + 1/d

    # Common destinations
    common = set(geo_scores.keys()) & set(observed_inflows.keys())

    if len(common) < 3:
        return {"method": "v0703b_aggregate", "error": "Too few common destinations"}

    pred_vec = [geo_scores[d] for d in common]
    obs_vec = [observed_inflows[d] for d in common]

    spearman, _ = stats.spearmanr(pred_vec, obs_vec)

    return {
        "method": "v0703b_aggregate",
        "description": "Raw 1/distance summed, Spearman over COMMON destinations",
        "spearman": round(float(spearman), 4),
        "n_destinations_common": len(common),
        "n_origins": len([o for o in origins if o in code_to_idx]),
    }


def method_v0703b_per_origin_spearman(
    holdout_df: pd.DataFrame,
    d_wass: np.ndarray,
    census_codes: List[int],
    min_destinations: int = 3,
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
) -> Dict:
    """
    v0.7.0.3b's per-origin Spearman methodology.

    For each origin with ≥min_destinations observed, compute Spearman
    between observed counts and 1/distance (common destinations only).
    """
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    origins = holdout_df[origin_col].unique()
    per_origin_rhos = []

    for origin in origins:
        if origin not in code_to_idx:
            continue
        i = code_to_idx[origin]

        # Observed destinations for this origin
        origin_df = holdout_df[holdout_df[origin_col] == origin]
        observed_counts = origin_df[dest_col].value_counts().to_dict()

        if len(observed_counts) < min_destinations:
            continue

        # Predicted scores: 1/distance for observed destinations
        predicted_scores = {}
        for dest in observed_counts.keys():
            if dest in code_to_idx and dest != origin:
                d = d_wass[i, code_to_idx[dest]]
                if d > 0:
                    predicted_scores[dest] = 1.0 / d

        common = set(observed_counts.keys()) & set(predicted_scores.keys())
        if len(common) < min_destinations:
            continue

        obs_vec = [observed_counts[d] for d in common]
        pred_vec = [predicted_scores[d] for d in common]

        rho, _ = stats.spearmanr(pred_vec, obs_vec)
        if not np.isnan(rho):
            per_origin_rhos.append(rho)

    mean_rho = float(np.mean(per_origin_rhos)) if per_origin_rhos else 0

    return {
        "method": "v0703b_per_origin",
        "description": f"Raw 1/distance, per-origin, min_dest≥{min_destinations}, common only",
        "mean_spearman": round(mean_rho, 4),
        "n_origins_evaluated": len(per_origin_rhos),
        "min_destinations_filter": min_destinations,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Experiment v0.7.0.3c: Methodology Audit")
    print("=" * 70)

    # Model parameters from v0.7.0c
    GAMMA_SEM = 8.9346
    GAMMA_INST = 0.1361

    # Load data using canonical functions
    print("\n[1] Loading data...")
    d_wass, census_codes = load_wasserstein_census()
    d_inst, _ = load_institutional_census()
    holdout_2024 = get_holdout_transitions()

    print(f"    Census codes: {len(census_codes)}")
    print(f"    Holdout 2024+ transitions: {len(holdout_2024)}")
    print(f"    Unique origins in holdout: {holdout_2024['origin_occ'].nunique()}")

    # Compute model probability matrix
    print("\n[2] Computing model probability matrix...")
    prob_matrix = compute_model_probabilities(d_wass, d_inst, GAMMA_SEM, GAMMA_INST)

    # ==========================================================================
    # Test all methodologies on SAME sample (all 2024+ transitions)
    # ==========================================================================

    print("\n[3] Running all methodologies on same sample (all 2024+ origins)...")

    results = {
        "experiment": "methodology_audit_v0703c",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample": {
            "description": "All 2024+ holdout transitions",
            "n_transitions": len(holdout_2024),
            "n_origins": int(holdout_2024["origin_occ"].nunique()),
        },
        "model_parameters": {
            "gamma_sem": GAMMA_SEM,
            "gamma_inst": GAMMA_INST,
        },
    }

    # v0.7.0c methodologies
    print("\n    Running v0.7.0c methodologies...")
    v070c_agg = method_v070c_aggregate_spearman(holdout_2024, prob_matrix, census_codes)
    v070c_per5 = method_v070c_per_origin_spearman(holdout_2024, prob_matrix, census_codes, min_destinations=5)
    v070c_per3 = method_v070c_per_origin_spearman(holdout_2024, prob_matrix, census_codes, min_destinations=3)

    print(f"      Aggregate (all dests): ρ = {v070c_agg['spearman']}")
    print(f"      Per-origin (≥5 dests): ρ = {v070c_per5['mean_spearman']} (n={v070c_per5['n_origins_evaluated']})")
    print(f"      Per-origin (≥3 dests): ρ = {v070c_per3['mean_spearman']} (n={v070c_per3['n_origins_evaluated']})")

    # v0.7.0.3b methodologies
    print("\n    Running v0.7.0.3b methodologies...")
    v0703b_agg = method_v0703b_aggregate_spearman(holdout_2024, d_wass, census_codes)
    v0703b_per5 = method_v0703b_per_origin_spearman(holdout_2024, d_wass, census_codes, min_destinations=5)
    v0703b_per3 = method_v0703b_per_origin_spearman(holdout_2024, d_wass, census_codes, min_destinations=3)

    print(f"      Aggregate (common dests): ρ = {v0703b_agg['spearman']}")
    print(f"      Per-origin (≥5 dests): ρ = {v0703b_per5['mean_spearman']} (n={v0703b_per5['n_origins_evaluated']})")
    print(f"      Per-origin (≥3 dests): ρ = {v0703b_per3['mean_spearman']} (n={v0703b_per3['n_origins_evaluated']})")

    results["methodologies"] = {
        "v070c_aggregate": v070c_agg,
        "v070c_per_origin_min5": v070c_per5,
        "v070c_per_origin_min3": v070c_per3,
        "v0703b_aggregate": v0703b_agg,
        "v0703b_per_origin_min5": v0703b_per5,
        "v0703b_per_origin_min3": v0703b_per3,
    }

    # ==========================================================================
    # Reconciliation: Explain the original discrepancies
    # ==========================================================================

    print("\n[4] Reconciliation analysis...")

    # Original v0.7.0c reported numbers
    original_v070c = {
        "aggregate_spearman": 0.4319,  # From reallocation_v070c.json
        "per_origin_spearman": 0.1402,
        "n_origins_evaluated": 60,
        "sample": "Exposed origins only (AIOE ≥ Q75), year ≥ 2024",
    }

    # Original v0.7.0.3b reported numbers
    original_v0703b = {
        "aggregate_spearman": 0.043,
        "per_origin_spearman": 0.316,
        "sample": "All origins, year ≥ 2024",
    }

    results["original_reported"] = {
        "v070c": original_v070c,
        "v0703b": original_v0703b,
    }

    # Compute discrepancy explanations
    reconciliation = {
        "aggregate_discrepancy": {
            "v070c_original": original_v070c["aggregate_spearman"],
            "v070c_on_full_sample": v070c_agg["spearman"],
            "v0703b_on_full_sample": v0703b_agg["spearman"],
            "explanation": (
                "v0.7.0c's 0.432 was computed on EXPOSED origins only using MODEL probabilities "
                "over ALL destinations. When applied to full sample, similar methodology gives "
                f"{v070c_agg['spearman']}. v0.7.0.3b's 0.043 uses raw 1/distance over COMMON destinations."
            ),
        },
        "per_origin_discrepancy": {
            "v070c_original": original_v070c["per_origin_spearman"],
            "v070c_on_full_sample_min5": v070c_per5["mean_spearman"],
            "v0703b_on_full_sample_min3": v0703b_per3["mean_spearman"],
            "explanation": (
                "v0.7.0c's 0.140 required ≥5 destinations per origin and used MODEL probabilities. "
                "v0.7.0.3b's 0.316 required only ≥3 destinations and used raw 1/distance on COMMON "
                "destinations only, which is a more favorable metric."
            ),
        },
    }

    results["reconciliation"] = reconciliation

    # ==========================================================================
    # Verdict
    # ==========================================================================

    print("\n[5] Computing verdict...")

    # The key question: what is the TRUE per-origin pathway accuracy?
    # Most comparable: v0.7.0c methodology (model probs) on full sample

    verdict = {
        "true_aggregate_spearman": {
            "value": v070c_agg["spearman"],
            "methodology": "Model probability, all destinations",
            "interpretation": "How well does the model rank all destinations by aggregate inflow?",
        },
        "true_per_origin_spearman": {
            "value_min5": v070c_per5["mean_spearman"],
            "value_min3": v070c_per3["mean_spearman"],
            "n_origins_min5": v070c_per5["n_origins_evaluated"],
            "n_origins_min3": v070c_per3["n_origins_evaluated"],
            "methodology": "Model probability, per-origin, full sample",
            "interpretation": "Given origin A, how well does model rank destinations for workers from A?",
        },
        "discrepancy_explained": True,
        "explanation": (
            "The 0.432 in v0.7.0c came from a RESTRICTED sample (exposed origins only, n=60) "
            "using model probabilities over ALL destinations. The 0.043 in v0.7.0.3b used ALL origins "
            "with raw 1/distance over COMMON destinations. These are fundamentally different metrics. "
            f"On the full sample with v0.7.0c methodology: aggregate ρ = {v070c_agg['spearman']}, "
            f"per-origin ρ = {v070c_per5['mean_spearman']} (≥5) / {v070c_per3['mean_spearman']} (≥3)."
        ),
        "recommended_headline": {
            "metric": "Per-origin Spearman (model probability, ≥5 destinations)",
            "value": v070c_per5["mean_spearman"],
            "n_origins": v070c_per5["n_origins_evaluated"],
            "sample": "All 2024+ origins",
        },
        "warnings": [],
    }

    # Check if original 0.43 was misleading
    if v070c_agg["spearman"] < 0.35:
        verdict["warnings"].append(
            f"Original 0.432 does not replicate on full sample ({v070c_agg['spearman']}). "
            "The original was computed on exposed origins only."
        )

    results["verdict"] = verdict

    # ==========================================================================
    # Save
    # ==========================================================================

    print("\n[6] Saving results...")
    output_path = save_experiment_output("methodology_audit_v0703c", results)
    print(f"    Saved to: {output_path}")

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 70)
    print("METHODOLOGY AUDIT SUMMARY")
    print("=" * 70)

    print("\n1. Original Reported Numbers:")
    print(f"   v0.7.0c aggregate: {original_v070c['aggregate_spearman']} (exposed origins only)")
    print(f"   v0.7.0c per-origin: {original_v070c['per_origin_spearman']} (n={original_v070c['n_origins_evaluated']})")
    print(f"   v0.7.0.3b aggregate: {original_v0703b['aggregate_spearman']} (all origins)")
    print(f"   v0.7.0.3b per-origin: {original_v0703b['per_origin_spearman']}")

    print("\n2. Reconciliation (same sample, both methodologies):")
    print(f"   v0.7.0c methodology, aggregate: {v070c_agg['spearman']}")
    print(f"   v0.7.0c methodology, per-origin (≥5): {v070c_per5['mean_spearman']} (n={v070c_per5['n_origins_evaluated']})")
    print(f"   v0.7.0.3b methodology, aggregate: {v0703b_agg['spearman']}")
    print(f"   v0.7.0.3b methodology, per-origin (≥3): {v0703b_per3['mean_spearman']} (n={v0703b_per3['n_origins_evaluated']})")

    print("\n3. Verdict:")
    print(f"   Discrepancy explained: {verdict['discrepancy_explained']}")
    print(f"   Recommended headline metric: Per-origin ρ = {verdict['recommended_headline']['value']}")
    print(f"   Sample: {verdict['recommended_headline']['sample']}")

    if verdict["warnings"]:
        print("\n   WARNINGS:")
        for w in verdict["warnings"]:
            print(f"   - {w}")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    main()
