#!/usr/bin/env python3
"""
D1: AIOE-Centroid Correlation (v0.7.12.1)

Recomputes AIOE-distance correlation using centroid (cosine embedding)
instead of Wasserstein distances.

Usage:
    python scripts/run_aioe_centroid_v07121.py
"""

import json
from datetime import datetime, timezone

import numpy as np
from scipy.stats import pearsonr, spearmanr

from task_space.mobility.io import load_centroid_census
from task_space.validation.shock_integration import (
    get_aioe_by_soc_dataframe,
    map_aioe_to_census,
)


def main():
    print("=== D1: AIOE-Centroid Correlation (v0.7.12.1) ===")

    # Load centroid matrix
    d_cent, census_codes = load_centroid_census()
    print(f"Centroid matrix: {d_cent.shape}")

    # Load AIOE scores mapped to Census codes
    aioe_soc = get_aioe_by_soc_dataframe(use_lm=True)
    aioe_census = map_aioe_to_census(aioe_soc)
    print(f"AIOE scores (Census): {len(aioe_census)}")

    # Build lookup
    aioe_dict = dict(zip(aioe_census["census_code"], aioe_census["aioe_score"]))
    code_to_idx = {c: i for i, c in enumerate(census_codes)}
    census_set = set(census_codes)

    # Matched codes
    matched_codes = set(aioe_dict.keys()) & census_set
    coverage = len(matched_codes) / len(census_set) if census_set else 0.0
    print(f"Matched occupations: {len(matched_codes)} ({coverage:.1%})")

    # For each matched occupation, compute mean centroid distance to all others
    aioe_scores = []
    mean_distances = []

    for code in matched_codes:
        if code in code_to_idx:
            idx = code_to_idx[code]
            mean_dist = np.mean(d_cent[idx, :])
            aioe_scores.append(aioe_dict[code])
            mean_distances.append(mean_dist)

    aioe_arr = np.array(aioe_scores)
    dist_arr = np.array(mean_distances)

    # Pearson r
    r_pearson, p_pearson = pearsonr(aioe_arr, dist_arr)

    # Spearman rho
    rho_spearman, p_spearman = spearmanr(aioe_arr, dist_arr)

    print(f"AIOE-centroid Pearson r  = {r_pearson:.4f} (p={p_pearson:.4f})")
    print(f"AIOE-centroid Spearman ρ = {rho_spearman:.4f} (p={p_spearman:.4f})")
    print(f"AIOE-centroid r = {r_pearson:.3f} (Wasserstein was 0.020)")

    # Save
    result = {
        "experiment": "aioe_centroid_v07121",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "distance_metric": "cosine_embed (centroid)",
        "aioe_centroid_pearson_r": round(float(r_pearson), 6),
        "aioe_centroid_pearson_p": float(p_pearson),
        "aioe_centroid_spearman_rho": round(float(rho_spearman), 6),
        "n_matched_occupations": len(matched_codes),
        "aioe_coverage": round(float(coverage), 4),
        "comparison_to_wasserstein": {
            "wasserstein_r": 0.020,
            "source": "outputs/experiments/shock_integration_v070a.json",
        },
    }

    out_path = "outputs/experiments/aioe_centroid_v07121.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
