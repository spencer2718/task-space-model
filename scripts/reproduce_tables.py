#!/usr/bin/env python3
"""
Reproduce key paper tables from cached artifacts.

Loads pre-computed distance matrices and transition data, re-runs
conditional logit estimation, and prints results for comparison
against paper Tables 2, 3, and 5.

Usage: python scripts/reproduce_tables.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_space.mobility.io import load_distance_matrix, load_transitions
from task_space.mobility.choice_model import build_choice_dataset, fit_conditional_logit


def compute_null_ll(n_transitions, n_alternatives=11):
    return n_transitions * np.log(1 / n_alternatives)


def fit_and_report(name, d_sem, d_inst, census_codes, transitions_df):
    choice_df = build_choice_dataset(
        transitions_df, d_sem_matrix=d_sem, d_inst_matrix=d_inst,
        occ_codes=census_codes, n_alternatives=10, random_seed=42,
    )
    n_trans = choice_df["transition_id"].nunique()
    result = fit_conditional_logit(choice_df)
    ll_null = compute_null_ll(n_trans)
    pseudo_r2 = 1 - (result.log_likelihood / ll_null)
    return {
        "name": name, "alpha": result.alpha, "alpha_se": result.alpha_se,
        "beta": result.beta, "beta_se": result.beta_se,
        "ll": result.log_likelihood, "pseudo_r2": pseudo_r2, "n": n_trans,
    }


def main():
    print("=" * 70)
    print("Reproducing Paper Tables from Cached Artifacts")
    print("=" * 70)

    # Load data
    d_inst, _ = load_distance_matrix("institutional")
    transitions = load_transitions()

    # Get census codes from centroid matrix
    d_centroid, census_codes = load_distance_matrix("cosine_embed")
    valid = set(census_codes)
    mask = transitions["origin_occ"].isin(valid) & transitions["dest_occ"].isin(valid)
    transitions = transitions[mask].copy()
    print(f"Transitions: {len(transitions):,}")

    # Table 2: 2x2 comparison
    print("\n--- TABLE 2: Distance Metric Comparison ---")
    metrics = {
        "cosine_embed": "Embedding × Centroid",
        "wasserstein": "Embedding × Wasserstein",
        "cosine_onet": "O*NET × Cosine",
        "euclidean_dwa": "O*NET × Euclidean",
    }

    results = {}
    for kind, label in metrics.items():
        d_sem, _ = load_distance_matrix(kind)
        if kind == "wasserstein":
            np.fill_diagonal(d_sem, 0.0)  # Apply diagonal correction
        r = fit_and_report(label, d_sem, d_inst, census_codes, transitions)
        results[kind] = r
        print(f"  {label:<30} α={r['alpha']:.3f}  R²={r['pseudo_r2']*100:.2f}%")

    # Table 3: Ground metric comparison
    print("\n--- TABLE 3: Ground Metric Comparison ---")
    d_identity, _ = load_distance_matrix("wasserstein_identity")
    r_id = fit_and_report("Identity", d_identity, d_inst, census_codes, transitions)
    d_wass, _ = load_distance_matrix("wasserstein")
    np.fill_diagonal(d_wass, 0.0)
    r_em = fit_and_report("Embedding", d_wass, d_inst, census_codes, transitions)
    pct = (r_em["pseudo_r2"] / r_id["pseudo_r2"] - 1) * 100
    print(f"  Identity:  α={r_id['alpha']:.3f}  γ={r_id['beta']:.3f}  R²={r_id['pseudo_r2']*100:.2f}%")
    print(f"  Embedding: α={r_em['alpha']:.3f}  γ={r_em['beta']:.3f}  R²={r_em['pseudo_r2']*100:.2f}%")
    print(f"  Improvement: +{pct:.1f}%")

    # Table 5: Primary spec coefficients
    print("\n--- TABLE 5: Primary Specification (Centroid) ---")
    r = results["cosine_embed"]
    print(f"  α = {r['alpha']:.4f} (SE = {r['alpha_se']:.4f}, t = {r['alpha']/r['alpha_se']:.1f})")
    print(f"  β = {r['beta']:.4f} (SE = {r['beta_se']:.4f}, t = {r['beta']/r['beta_se']:.1f})")

    print("\n" + "=" * 70)
    print("Done. Compare against paper tables.")


if __name__ == "__main__":
    main()
