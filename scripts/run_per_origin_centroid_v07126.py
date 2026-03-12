#!/usr/bin/env python3
"""
D6: Per-Origin Model-Probability Spearman on Centroid (v0.7.12.6)

Computes the paper's per-origin ρ using model probabilities P(j|i)
over ALL 447 destinations, replacing the Wasserstein-based value (0.128).

Usage:
    python scripts/run_per_origin_centroid_v07126.py
"""

import json
from datetime import datetime, timezone

from task_space.mobility.io import (
    get_holdout_transitions,
    load_centroid_census,
    load_institutional_census,
)
from task_space.validation.shock_integration import compute_model_probabilities
from task_space.validation.spearman import per_origin_spearman_model_prob


# Centroid conditional logit coefficients (Table 5)
ALPHA = 7.404
BETA = 0.139


def main():
    print("=== D6: Per-Origin Model-Probability Spearman on Centroid (v0.7.12.6) ===")

    # Load data
    print("[1] Loading data...")
    d_cent, census_codes = load_centroid_census()
    d_inst, _ = load_institutional_census()
    holdout_df = get_holdout_transitions()
    print(f"  Centroid: {d_cent.shape}")
    print(f"  Institutional: {d_inst.shape}")
    print(f"  Holdout: {len(holdout_df):,}")

    # Compute model probability matrix
    print("[2] Computing model probabilities...")
    model_probs = compute_model_probabilities(
        holdout_df, d_cent, d_inst, census_codes,
        alpha=ALPHA, beta=BETA,
    )
    print(f"  Row sum check: {model_probs.sum(axis=1).mean():.3f}")

    # Per-origin Spearman over all destinations
    print("[3] Computing per-origin Spearman...")
    result = per_origin_spearman_model_prob(
        holdout_df, model_probs, census_codes, min_destinations=5,
    )

    mean_rho = result.mean_spearman
    median_rho = result.median_spearman
    std_rho = result.std_spearman
    n_origins = result.n_origins_evaluated

    print(f"  Mean ρ:   {mean_rho:.4f}")
    print(f"  Median ρ: {median_rho:.4f}")
    print(f"  Std ρ:    {std_rho:.4f}")
    print(f"  N origins: {n_origins}")
    print(f"Per-origin model-prob ρ = {mean_rho:.3f}, n = {n_origins} (Wasserstein was 0.128, n = 233)")

    # Save
    output = {
        "experiment": "per_origin_model_prob_centroid_v07126",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "distance_metric": "cosine_embed (centroid)",
        "alpha": ALPHA,
        "beta": BETA,
        "methodology": "model_probability_per_origin_all_destinations",
        "mean_spearman": round(float(mean_rho), 6),
        "median_spearman": round(float(median_rho), 6),
        "std_spearman": round(float(std_rho), 6),
        "n_origins_evaluated": n_origins,
        "comparison_to_wasserstein": {
            "wasserstein_mean_spearman": 0.128,
            "source": "run_methodology_audit_v0703c.py / paper Table",
        },
    }

    out_path = "outputs/experiments/per_origin_model_prob_centroid_v07126.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
