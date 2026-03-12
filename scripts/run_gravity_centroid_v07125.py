#!/usr/bin/env python3
"""
D5: Gravity Model on Centroid (v0.7.12.5)

Reports centroid partial R² alongside Wasserstein verification.
Adapts run_gravity_model_v0734.py with only centroid + wasserstein metrics.

Usage:
    python scripts/run_gravity_centroid_v07125.py
"""

import json
import time
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm

from task_space.mobility.io import load_distance_matrix, load_transitions


def build_gravity_dataset(
    transitions_df: pd.DataFrame,
    distance_matrices: Dict[str, np.ndarray],
    census_codes: List[int],
) -> pd.DataFrame:
    """Build gravity model dataset with all occupation pairs."""
    n_occ = len(census_codes)
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    bilateral = transitions_df.groupby(["origin_occ", "dest_occ"]).size()
    bilateral = bilateral.reset_index(name="flow")

    emp_origin = transitions_df.groupby("origin_occ").size()
    emp_dest = transitions_df.groupby("dest_occ").size()
    emp_origin_dict = emp_origin.to_dict()
    emp_dest_dict = emp_dest.to_dict()

    # Build flow lookup for speed
    flow_lookup = {}
    for _, row in bilateral.iterrows():
        flow_lookup[(row["origin_occ"], row["dest_occ"])] = row["flow"]

    rows = []
    for i, orig in enumerate(census_codes):
        for j, dest in enumerate(census_codes):
            if orig == dest:
                continue

            flow = flow_lookup.get((orig, dest), 0)
            e_orig = emp_origin_dict.get(orig, 1)
            e_dest = emp_dest_dict.get(dest, 1)

            row = {
                "flow": flow,
                "emp_origin": e_orig,
                "emp_dest": e_dest,
            }

            for metric, d_matrix in distance_matrices.items():
                row[f"d_{metric}"] = d_matrix[i, j]

            rows.append(row)

    df = pd.DataFrame(rows)
    df["ln_flow"] = np.log(df["flow"] + 1)
    df["ln_emp_origin"] = np.log(df["emp_origin"] + 1)
    df["ln_emp_dest"] = np.log(df["emp_dest"] + 1)

    return df


def main():
    print("=== D5: Gravity Model on Centroid (v0.7.12.5) ===")
    start = time.time()

    # Load distance matrices
    print("[1] Loading distance matrices...")
    d_cent, census_codes = load_distance_matrix(kind="cosine_embed")
    d_wass, _ = load_distance_matrix(kind="wasserstein")
    distance_matrices = {"centroid": d_cent, "wasserstein": d_wass}
    print(f"  Centroid range: [{d_cent.min():.4f}, {d_cent.max():.4f}]")
    print(f"  Wasserstein range: [{d_wass.min():.4f}, {d_wass.max():.4f}]")

    # Load transitions
    print("[2] Loading CPS transitions...")
    transitions = load_transitions()
    valid = set(census_codes)
    mask = transitions["origin_occ"].isin(valid) & transitions["dest_occ"].isin(valid)
    trans = transitions[mask].copy()
    print(f"  Transitions: {len(trans):,}")

    # Build gravity dataset
    print("[3] Building gravity dataset...")
    df = build_gravity_dataset(trans, distance_matrices, census_codes)
    n_positive = int((df["flow"] > 0).sum())
    n_total = len(df)
    print(f"  Total pairs: {n_total:,}")
    print(f"  Positive flow: {n_positive:,} ({100 * n_positive / n_total:.1f}%)")

    # Mass-only model
    print("[4] Fitting mass-only model...")
    X_mass = sm.add_constant(df[["ln_emp_origin", "ln_emp_dest"]])
    model_mass = sm.OLS(df["ln_flow"], X_mass).fit()
    r2_mass = model_mass.rsquared
    print(f"  R² (mass only): {r2_mass:.4f} ({100 * r2_mass:.2f}%)")

    # Fit gravity models
    print("[5] Fitting gravity models...")
    results = {}

    for metric in ["centroid", "wasserstein"]:
        col = f"d_{metric}"
        X = sm.add_constant(df[["ln_emp_origin", "ln_emp_dest", col]])
        model = sm.OLS(df["ln_flow"], X).fit()

        partial_r2 = model.rsquared - r2_mass

        results[metric] = {
            "beta_distance": round(float(model.params[col]), 6),
            "beta_distance_se": round(float(model.bse[col]), 6),
            "t_stat": round(float(model.tvalues[col]), 2),
            "r2_full": round(float(model.rsquared), 6),
            "partial_r2": round(100 * float(partial_r2), 4),
            "mass_only_r2": round(100 * float(r2_mass), 4),
        }

        print(f"  {metric}: β={model.params[col]:.4f}, partial R²={100 * partial_r2:.2f}%")

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Centroid partial R²: {results['centroid']['partial_r2']:.2f}%")
    print(f"Wasserstein partial R²: {results['wasserstein']['partial_r2']:.2f}% (was 3.46%)")

    # Save
    output = {
        "experiment": "gravity_centroid_v07125",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_pairs": n_total,
        "results": {
            "centroid": results["centroid"],
            "wasserstein_verify": {
                **results["wasserstein"],
                "note": "Should match 3.46% from original",
            },
        },
    }

    out_path = "outputs/experiments/gravity_centroid_v07125.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
