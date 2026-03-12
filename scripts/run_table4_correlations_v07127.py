#!/usr/bin/env python3
"""
v0.7.12.7: Table 4 Centroid Correlations

Compute Spearman ρ between centroid (reference) and each other distance
matrix on the upper triangle (excluding diagonal).

Usage:
    python scripts/run_table4_correlations_v07127.py
"""

import json
from datetime import datetime, timezone

import numpy as np
from scipy.stats import spearmanr

from task_space.mobility.io import load_distance_matrix


def main():
    print("=== Table 4 Centroid Correlations (v0.7.12.7) ===")

    # Load all matrices
    print("[1] Loading distance matrices...")
    d_cent, codes = load_distance_matrix("cosine_embed")
    d_wass, _ = load_distance_matrix("wasserstein")
    d_euc, _ = load_distance_matrix("euclidean_dwa")
    d_ident, _ = load_distance_matrix("wasserstein_identity")
    d_onet_cos, _ = load_distance_matrix("cosine_onet")
    print(f"  All matrices loaded: {d_cent.shape[0]} occupations")

    # Upper triangle indices
    iu = np.triu_indices(447, k=1)
    n_pairs = len(iu[0])
    cent_upper = d_cent[iu]

    # Compute correlations
    print("[2] Computing Spearman correlations...")
    comparisons = {
        "centroid_vs_wasserstein": d_wass,
        "centroid_vs_wasserstein_identity": d_ident,
        "centroid_vs_euclidean_onet": d_euc,
        "centroid_vs_cosine_onet": d_onet_cos,
    }

    correlations = {}
    for name, d_other in comparisons.items():
        rho, _ = spearmanr(cent_upper, d_other[iu])
        correlations[name] = round(float(rho), 6)
        print(f"  {name}: ρ = {rho:.4f}")

    # Save
    result = {
        "experiment": "table4_correlations_v07127",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reference_matrix": "cosine_embed (centroid)",
        "correlations": correlations,
        "n_pairs": n_pairs,
    }

    out_path = "outputs/experiments/table4_correlations_v07127.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
