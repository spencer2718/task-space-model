#!/usr/bin/env python3
"""
D4: Out-of-Period Comparison on Centroid (v0.7.12.4)

Recomputes ΔLL = geometry vs historical baseline using centroid-based
model probabilities on the 2024 holdout.

Usage:
    python scripts/run_oop_centroid_v07124.py
"""

import json
from datetime import datetime, timezone

import numpy as np

from task_space.mobility.io import (
    load_transitions,
    get_training_transitions,
    get_holdout_transitions,
    load_centroid_census,
    load_institutional_census,
)
from task_space.validation.shock_integration import (
    compute_historical_baseline,
    compute_model_probabilities,
    evaluate_model_on_holdout,
)


# Centroid conditional logit coefficients (from Table 5 — centroid spec)
ALPHA = 7.404
BETA = 0.139


def main():
    print("=== D4: Out-of-Period Comparison on Centroid (v0.7.12.4) ===")

    # Load data
    print("[1] Loading data...")
    transitions = load_transitions()
    train_df = get_training_transitions(transitions)
    holdout_df = get_holdout_transitions()
    print(f"  Train: {len(train_df):,}")
    print(f"  Holdout: {len(holdout_df):,}")

    d_cent, census_codes = load_centroid_census()
    d_inst, _ = load_institutional_census()
    print(f"  Centroid matrix: {d_cent.shape}")
    print(f"  Institutional matrix: {d_inst.shape}")

    # Historical baseline from training data
    print("[2] Computing historical baseline...")
    historical_probs = compute_historical_baseline(train_df, census_codes)
    print(f"  Row sum check: {historical_probs.sum(axis=1).mean():.3f}")

    # Model probabilities using centroid
    # compute_model_probabilities accepts the semantic matrix via its
    # 'wasserstein_matrix' parameter name, but it just multiplies by alpha —
    # the function doesn't care what the matrix is.
    print("[3] Computing model probabilities...")
    model_probs = compute_model_probabilities(
        train_df, d_cent, d_inst, census_codes,
        alpha=ALPHA, beta=BETA,
    )
    print(f"  Row sum check: {model_probs.sum(axis=1).mean():.3f}")

    # Evaluate on holdout
    print("[4] Evaluating on holdout...")
    metrics = evaluate_model_on_holdout(
        model_probs, historical_probs, holdout_df, census_codes,
    )

    geometry_ll = metrics["geometry_ll"]
    historical_ll = metrics["baseline_historical_ll"]
    delta_ll = geometry_ll - historical_ll
    n_evaluated = metrics["n_evaluated"]

    print(f"  Geometry LL:   {geometry_ll:,.0f}")
    print(f"  Historical LL: {historical_ll:,.0f}")
    print(f"  ΔLL = +{delta_ll:,.0f} (Wasserstein was +23,119)")
    print(f"  N evaluated: {n_evaluated:,}")

    # Save
    result = {
        "experiment": "oop_centroid_v07124",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "distance_metric": "cosine_embed (centroid)",
        "alpha": ALPHA,
        "beta": BETA,
        "geometry_ll": round(float(geometry_ll), 2),
        "historical_ll": round(float(historical_ll), 2),
        "delta_ll": round(float(delta_ll), 2),
        "n_evaluated": n_evaluated,
        "comparison_to_wasserstein": {
            "wasserstein_delta_ll": 23119,
            "source": "S module validation",
        },
    }

    out_path = "outputs/experiments/oop_centroid_v07124.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
