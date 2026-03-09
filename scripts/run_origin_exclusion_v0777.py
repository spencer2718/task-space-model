#!/usr/bin/env python3
"""
v0.7.7.7: Origin-Exclusion Robustness Test

Tests whether excluding the origin occupation from the sampled alternatives
changes the primary conditional logit results. In the baseline specification,
the origin can appear as an alternative with zero task distance. This script
compares baseline (origin included) vs. modified (origin excluded) results.

Stop-and-return:
- If excluding origin changes α by more than 10%, STOP — material finding.
- If modified sampling yields any choice sets with < 10 alternatives, STOP.
"""

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_space.mobility.io import load_distance_matrix, load_transitions
from task_space.mobility.choice_model import build_choice_dataset, fit_conditional_logit
from task_space.utils.experiments import save_experiment_output


def build_choice_dataset_exclude_origin(
    transitions_df: pd.DataFrame,
    d_sem_matrix: np.ndarray,
    d_inst_matrix: np.ndarray,
    occ_codes: List[int],
    n_alternatives: int = 10,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build choice dataset excluding both destination AND origin from alternatives."""
    np.random.seed(random_seed)

    occ_to_idx = {occ: i for i, occ in enumerate(occ_codes)}
    all_occs = set(occ_codes)

    rows = []
    transition_id = 0
    skipped_insufficient = 0

    for _, row in transitions_df.iterrows():
        origin = int(row["origin_occ"])
        dest = int(row["dest_occ"])

        if origin not in occ_to_idx or dest not in occ_to_idx:
            continue

        origin_idx = occ_to_idx[origin]
        dest_idx = occ_to_idx[dest]

        # Exclude BOTH destination AND origin from alternatives
        available = list(all_occs - {dest, origin})
        if len(available) < n_alternatives:
            skipped_insufficient += 1
            continue

        sampled_alts = np.random.choice(available, size=n_alternatives, replace=False)

        # Add chosen destination
        rows.append({
            "transition_id": transition_id,
            "occ": dest,
            "chosen": 1,
            "neg_d_sem": -d_sem_matrix[origin_idx, dest_idx],
            "neg_d_inst": -d_inst_matrix[origin_idx, dest_idx],
        })

        # Add sampled alternatives
        for alt in sampled_alts:
            alt_idx = occ_to_idx[alt]
            rows.append({
                "transition_id": transition_id,
                "occ": alt,
                "chosen": 0,
                "neg_d_sem": -d_sem_matrix[origin_idx, alt_idx],
                "neg_d_inst": -d_inst_matrix[origin_idx, alt_idx],
            })

        transition_id += 1

    if skipped_insufficient > 0:
        print(f"  WARNING: {skipped_insufficient} choice sets had < {n_alternatives} alternatives after origin exclusion")
        print("  STOP: This should not happen with 447 codes minus 2.")
        sys.exit(1)

    return pd.DataFrame(rows)


def compute_null_ll(n_transitions, n_alternatives=11):
    return n_transitions * np.log(1 / n_alternatives)


def main():
    print("=" * 70)
    print("v0.7.7.7: Origin-Exclusion Robustness Test")
    print("=" * 70)
    start_time = time.time()

    # Load data
    print("\nLoading data...")
    d_sem, census_codes = load_distance_matrix("cosine_embed")
    d_inst, _ = load_distance_matrix("institutional")
    transitions = load_transitions()

    valid = set(census_codes)
    mask = transitions["origin_occ"].isin(valid) & transitions["dest_occ"].isin(valid)
    transitions = transitions[mask].copy()
    print(f"  Transitions: {len(transitions):,}")
    print(f"  Occupations: {len(census_codes)}")

    # Count how often origin appears in baseline alternatives
    np.random.seed(42)
    origin_in_alts = 0
    total = 0
    occ_to_idx = {occ: i for i, occ in enumerate(census_codes)}
    all_occs = set(census_codes)
    for _, row in transitions.iterrows():
        origin = int(row["origin_occ"])
        dest = int(row["dest_occ"])
        if origin not in occ_to_idx or dest not in occ_to_idx:
            continue
        available = list(all_occs - {dest})
        sampled = set(np.random.choice(available, size=10, replace=False))
        if origin in sampled:
            origin_in_alts += 1
        total += 1
    pct_origin = 100 * origin_in_alts / total if total > 0 else 0
    print(f"  Origin in alternatives: {origin_in_alts:,}/{total:,} ({pct_origin:.1f}%)")

    # Baseline: origin included (standard build_choice_dataset)
    print("\n--- Baseline (origin included in alternatives) ---")
    choice_base = build_choice_dataset(
        transitions, d_sem_matrix=d_sem, d_inst_matrix=d_inst,
        occ_codes=census_codes, n_alternatives=10, random_seed=42,
    )
    n_base = choice_base["transition_id"].nunique()
    result_base = fit_conditional_logit(choice_base)
    ll_null_base = compute_null_ll(n_base)
    r2_base = 1 - (result_base.log_likelihood / ll_null_base)
    print(f"  α = {result_base.alpha:.4f} (SE = {result_base.alpha_se:.4f})")
    print(f"  β = {result_base.beta:.4f} (SE = {result_base.beta_se:.4f})")
    print(f"  pseudo-R² = {r2_base*100:.2f}%")

    # Modified: origin excluded
    print("\n--- Modified (origin excluded from alternatives) ---")
    choice_excl = build_choice_dataset_exclude_origin(
        transitions, d_sem, d_inst, census_codes,
        n_alternatives=10, random_seed=42,
    )
    n_excl = choice_excl["transition_id"].nunique()
    result_excl = fit_conditional_logit(choice_excl)
    ll_null_excl = compute_null_ll(n_excl)
    r2_excl = 1 - (result_excl.log_likelihood / ll_null_excl)
    print(f"  α = {result_excl.alpha:.4f} (SE = {result_excl.alpha_se:.4f})")
    print(f"  β = {result_excl.beta:.4f} (SE = {result_excl.beta_se:.4f})")
    print(f"  pseudo-R² = {r2_excl*100:.2f}%")

    # Comparison
    delta_alpha_pct = 100 * (result_excl.alpha - result_base.alpha) / abs(result_base.alpha)
    delta_r2_pp = (r2_excl - r2_base) * 100

    print(f"\n--- Comparison ---")
    print(f"  Δα = {delta_alpha_pct:+.2f}%")
    print(f"  ΔR² = {delta_r2_pp:+.2f}pp")

    # Stop-and-return: material change
    if abs(delta_alpha_pct) > 10:
        print(f"\nSTOP: α changed by {delta_alpha_pct:+.2f}% (>10%). Material finding.")
        sys.exit(1)

    print(f"\n  ✓ Origin exclusion does not materially affect results ({delta_alpha_pct:+.2f}%)")

    # Save output
    output = {
        "version": "0.7.7.7",
        "baseline": {
            "alpha": float(result_base.alpha),
            "alpha_se": float(result_base.alpha_se),
            "beta": float(result_base.beta),
            "beta_se": float(result_base.beta_se),
            "pseudo_r2": float(r2_base),
            "ll": float(result_base.log_likelihood),
            "n_transitions": int(n_base),
        },
        "origin_excluded": {
            "alpha": float(result_excl.alpha),
            "alpha_se": float(result_excl.alpha_se),
            "beta": float(result_excl.beta),
            "beta_se": float(result_excl.beta_se),
            "pseudo_r2": float(r2_excl),
            "ll": float(result_excl.log_likelihood),
            "n_transitions": int(n_excl),
        },
        "delta_alpha_pct": float(delta_alpha_pct),
        "delta_r2_pp": float(delta_r2_pp),
        "origin_in_alternatives_pct": float(pct_origin),
    }

    output_path = save_experiment_output("origin_exclusion_v0777", output)
    print(f"\nSaved: {output_path}")

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s")

    return output


if __name__ == "__main__":
    main()
