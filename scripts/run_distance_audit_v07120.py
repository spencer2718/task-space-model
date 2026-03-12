"""
Distance Matrix Audit — v0.7.12.0

Verify centroid matrix integrity, compute inter-matrix Spearman ρ
between centroid and Wasserstein, and report median centroid distance
for switching cost recalibration (D6).

Usage:
    python scripts/run_distance_audit_v07120.py
"""

import json
from datetime import datetime, timezone

import numpy as np
from scipy.stats import spearmanr

from task_space.mobility.io import load_wasserstein_census, load_centroid_census


def main():
    # Load both matrices via canonical IO
    d_cent, codes_c = load_centroid_census()
    d_wass, codes_w = load_wasserstein_census()

    # --- Centroid matrix checks ---
    shape = list(d_cent.shape)
    diag_all_zero = bool(np.all(d_cent.diagonal() == 0.0))
    has_nan = bool(np.any(np.isnan(d_cent)))
    has_negative = bool(np.any(d_cent < 0))
    n_codes = len(codes_c)

    iu = np.triu_indices(447, k=1)
    off_diag_vals = d_cent[iu]
    off_diag_mean = float(np.mean(off_diag_vals))
    off_diag_std = float(np.std(off_diag_vals))
    median_centroid = float(np.median(off_diag_vals))

    centroid_info = {
        "shape": shape,
        "diagonal_all_zero": diag_all_zero,
        "has_nan": has_nan,
        "has_negative": has_negative,
        "n_codes": n_codes,
        "off_diag_mean": round(off_diag_mean, 6),
        "off_diag_std": round(off_diag_std, 6),
        "median_distance": round(median_centroid, 6),
    }

    # --- Wasserstein matrix checks ---
    wass_diag = d_wass.diagonal()
    nonzero_diag_count = int(np.count_nonzero(wass_diag))
    max_diag = float(np.max(wass_diag))

    wasserstein_info = {
        "nonzero_diagonal_count": nonzero_diag_count,
        "max_diagonal": round(max_diag, 6),
        "note": "Diagonal correction applied at runtime, not in cached file",
    }

    # --- Codes match ---
    codes_match = bool(np.array_equal(codes_c, codes_w))

    # --- Inter-matrix Spearman ρ ---
    cent_upper = d_cent[iu]
    wass_upper = d_wass[iu]
    rho, p = spearmanr(cent_upper, wass_upper)

    # Also compute with Wasserstein diagonal zeroed (sanity check)
    d_wass_corrected = d_wass.copy()
    np.fill_diagonal(d_wass_corrected, 0.0)
    rho_corrected, p_corrected = spearmanr(cent_upper, d_wass_corrected[iu])

    n_pairs = len(cent_upper)

    correlation_info = {
        "spearman_rho": round(float(rho), 6),
        "spearman_p": float(p),
        "spearman_rho_diag_corrected": round(float(rho_corrected), 6),
        "sanity_check_match": bool(abs(rho - rho_corrected) < 1e-10),
        "note": f"Upper triangle excluding diagonal, {n_pairs} pairs",
    }

    # --- Verdict ---
    problems = []
    if not diag_all_zero:
        problems.append("centroid diagonal not all zero")
    if has_nan:
        problems.append("centroid has NaN")
    if has_negative:
        problems.append("centroid has negative values")
    if not codes_match:
        problems.append("codes mismatch between matrices")
    if rho < 0.90:
        problems.append(f"Spearman rho = {rho:.4f} < 0.90")

    verdict = "PASS" if not problems else f"FAIL: {'; '.join(problems)}"

    # --- Assemble output ---
    result = {
        "experiment": "distance_audit_v07120",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "centroid_matrix": centroid_info,
        "wasserstein_matrix": wasserstein_info,
        "codes_match": codes_match,
        "inter_matrix_correlation": correlation_info,
        "verdict": verdict,
    }

    # Print summary
    print("=== Distance Matrix Audit v0.7.12.0 ===")
    print(f"Centroid shape: {shape}")
    print(f"Centroid diagonal all zero: {diag_all_zero}")
    print(f"Centroid has NaN: {has_nan}")
    print(f"Centroid has negatives: {has_negative}")
    print(f"Centroid off-diag mean: {off_diag_mean:.6f}")
    print(f"Centroid median distance: {median_centroid:.6f}")
    print(f"Wasserstein nonzero diagonal: {nonzero_diag_count}")
    print(f"Wasserstein max diagonal: {max_diag:.6f}")
    print(f"Codes match: {codes_match}")
    print(f"Spearman rho: {rho:.6f} (p={p:.2e})")
    print(f"Spearman rho (diag-corrected): {rho_corrected:.6f}")
    print(f"Sanity check (rho == rho_corrected): {abs(rho - rho_corrected) < 1e-10}")
    print(f"Verdict: {verdict}")

    # Save
    out_path = "outputs/experiments/distance_audit_v07120.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
