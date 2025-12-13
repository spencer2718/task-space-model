#!/usr/bin/env python3
"""
Phase D: SAE Validation

Tests whether SAE-derived sparse features outperform raw binary overlap
for predicting wage comovement.

Key question: Does soft feature structure add value beyond discrete activity matching?

Usage:
    PYTHONPATH=src python tests/run_phase_d.py
"""

from pathlib import Path
import sys
import json

import numpy as np
from scipy import stats as sp_stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from task_space import (
    build_dwa_occupation_measures,
    load_oes_panel,
    compute_wage_comovement,
    onet_to_soc,
)


def compute_sae_overlap(
    occupation_features: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Compute Jaccard overlap from SAE-derived occupation features.

    Args:
        occupation_features: (n_occ, n_features) sparse occupation features
        threshold: Binarization threshold

    Returns:
        (n_occ, n_occ) Jaccard overlap matrix
    """
    # Binarize
    B = (occupation_features > threshold).astype(np.float64)

    # Jaccard
    intersection = B @ B.T
    row_sums = B.sum(axis=1, keepdims=True)
    union = row_sums + row_sums.T - intersection
    union[union == 0] = 1

    return intersection / union


def run_regression(
    overlap_matrix: np.ndarray,
    occupation_codes: list[str],
    comovement_matrix: np.ndarray,
    comovement_codes: list[str],
    crosswalk_map: dict[str, str],
) -> dict:
    """
    Run validation regression with clustered SEs.

    Returns dict with regression results.
    """
    # Map O*NET to SOC
    onet_to_soc_map = {}
    for onet_code in occupation_codes:
        soc = crosswalk_map.get(onet_code)
        if soc:
            onet_to_soc_map[onet_code] = soc

    # Get common SOC codes
    soc_codes = list(set(onet_to_soc_map.values()))
    soc_codes = [soc for soc in soc_codes if soc in comovement_codes]
    soc_codes = sorted(soc_codes)

    n_soc = len(soc_codes)
    soc_to_idx = {soc: i for i, soc in enumerate(soc_codes)}
    comovement_idx = {soc: i for i, soc in enumerate(comovement_codes)}
    onet_to_overlap_idx = {code: i for i, code in enumerate(occupation_codes)}

    # Aggregate overlap to SOC level
    soc_overlap = np.zeros((n_soc, n_soc))
    soc_counts = np.zeros((n_soc, n_soc))

    for onet_i, soc_i in onet_to_soc_map.items():
        if soc_i not in soc_to_idx:
            continue
        for onet_j, soc_j in onet_to_soc_map.items():
            if soc_j not in soc_to_idx:
                continue
            if onet_i >= onet_j:
                continue

            i_idx = onet_to_overlap_idx[onet_i]
            j_idx = onet_to_overlap_idx[onet_j]
            overlap_val = overlap_matrix[i_idx, j_idx]

            si = soc_to_idx[soc_i]
            sj = soc_to_idx[soc_j]
            if si > sj:
                si, sj = sj, si

            soc_overlap[si, sj] += overlap_val
            soc_counts[si, sj] += 1

    soc_counts[soc_counts == 0] = 1
    soc_overlap = soc_overlap / soc_counts

    # Build pairs
    pairs_overlap = []
    pairs_y = []
    pairs_cluster = []

    for i in range(n_soc):
        for j in range(i + 1, n_soc):
            soc_i = soc_codes[i]
            soc_j = soc_codes[j]

            ci = comovement_idx[soc_i]
            cj = comovement_idx[soc_j]
            comove_val = comovement_matrix[ci, cj]

            if np.isnan(comove_val):
                continue

            pairs_overlap.append(soc_overlap[i, j])
            pairs_y.append(comove_val)
            pairs_cluster.append(soc_i)

    X = np.array(pairs_overlap)
    y = np.array(pairs_y)
    clusters = np.array(pairs_cluster)

    # OLS with clustered SEs
    n = len(y)
    X_mat = np.column_stack([np.ones(n), X])

    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    beta_vec = XtX_inv @ (X_mat.T @ y)
    resid = y - X_mat @ beta_vec

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Clustered SEs
    unique_clusters, cluster_ids = np.unique(clusters, return_inverse=True)
    n_clusters = len(unique_clusters)

    meat = np.zeros((2, 2))
    for c_idx in range(n_clusters):
        mask = cluster_ids == c_idx
        cluster_resid = resid[mask]
        cluster_X = X_mat[mask]
        score = cluster_X.T @ cluster_resid
        meat += np.outer(score, score)

    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - 2))
    var_beta = correction * XtX_inv @ meat @ XtX_inv
    se_beta = np.sqrt(np.diag(var_beta))

    beta = beta_vec[1]
    se = se_beta[1]
    t_stat = beta / se
    pvalue = 2 * (1 - sp_stats.t.cdf(np.abs(t_stat), df=n_clusters - 1))

    t_crit = sp_stats.t.ppf(0.975, df=n_clusters - 1)
    ci_lower = beta - t_crit * se
    ci_upper = beta + t_crit * se

    return {
        "beta": float(beta),
        "se": float(se),
        "t_stat": float(t_stat),
        "pvalue": float(pvalue),
        "r_squared": float(r_squared),
        "n_pairs": int(n),
        "n_clusters": int(n_clusters),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "passes": bool(beta > 0 and pvalue < 0.10),
        "overlap_values": X.tolist(),  # For correlation analysis
    }


def main():
    print("=" * 60)
    print("Phase D: SAE Validation")
    print("=" * 60)

    output_dir = Path("outputs/phase_d")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    print("\n[1/5] Loading data...")

    # DWA sparse features from SAE
    dwa_features = np.load("outputs/phase_b/dwa_sparse_features.npy")
    print(f"      DWA sparse features: {dwa_features.shape}")

    # Occupation measures (for aggregation weights)
    measures = build_dwa_occupation_measures()
    print(f"      Occupations: {len(measures.occupation_codes)}")

    # Binary overlap from Phase A
    binary_overlap = np.load("outputs/phase_a/binary_overlap.npy")
    print(f"      Binary overlap: {binary_overlap.shape}")

    # Wage comovement
    oes_path = Path("data/external/oes")
    panel = load_oes_panel(years=[2019, 2020, 2021, 2022, 2023], data_dir=oes_path)
    comovement = compute_wage_comovement(panel, min_years=4)
    print(f"      Comovement: {comovement.n_occupations} occupations")

    # Step 2: Aggregate SAE features to occupation level
    print("\n[2/5] Aggregating SAE features to occupation level...")

    # W: (n_occ, n_dwa) occupation-activity weights
    # F: (n_dwa, n_features) sparse DWA features
    # W_sparse: (n_occ, n_features) occupation-level features

    W = measures.raw_matrix  # Use raw weights for aggregation
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-10)  # Normalize rows

    occupation_features = W_norm @ dwa_features
    print(f"      Occupation features: {occupation_features.shape}")

    # Feature statistics
    active_per_occ = (occupation_features > 0.01).sum(axis=1)
    print(f"      Mean active features per occupation: {active_per_occ.mean():.1f}")

    # Step 3: Compute SAE overlap
    print("\n[3/5] Computing SAE Jaccard overlap...")

    # Find good threshold (median non-zero activation)
    nonzero_vals = occupation_features[occupation_features > 0]
    threshold = np.median(nonzero_vals)
    print(f"      Binarization threshold: {threshold:.4f}")

    sae_overlap = compute_sae_overlap(occupation_features, threshold=threshold)
    print(f"      SAE overlap matrix: {sae_overlap.shape}")

    # Step 4: Compare Binary vs SAE overlap
    print("\n[4/5] Comparing Binary vs SAE overlap...")

    # Extract upper triangular (excluding diagonal)
    n = binary_overlap.shape[0]
    triu_idx = np.triu_indices(n, k=1)

    binary_flat = binary_overlap[triu_idx]
    sae_flat = sae_overlap[triu_idx]

    # Correlation
    # Filter to pairs where at least one has non-zero overlap
    valid_mask = (binary_flat > 0) | (sae_flat > 0)
    binary_valid = binary_flat[valid_mask]
    sae_valid = sae_flat[valid_mask]

    pearson_r, pearson_p = sp_stats.pearsonr(binary_valid, sae_valid)
    spearman_r, spearman_p = sp_stats.spearmanr(binary_valid, sae_valid)

    print(f"      Valid pairs (non-zero overlap): {valid_mask.sum()}")
    print(f"      Pearson r: {pearson_r:.3f} (p={pearson_p:.2e})")
    print(f"      Spearman r: {spearman_r:.3f} (p={spearman_p:.2e})")

    # Interpretation
    if pearson_r > 0.9:
        corr_interpretation = "HIGH (>0.9) - Measures are nearly identical, SAE unlikely to add value"
    elif pearson_r > 0.7:
        corr_interpretation = "MODERATE (0.7-0.9) - Significant overlap but room for SAE contribution"
    else:
        corr_interpretation = "LOW (<0.7) - Measures capture different structure"
    print(f"      Interpretation: {corr_interpretation}")

    # Step 5: Run validation regressions
    print("\n[5/5] Running validation regressions...")

    crosswalk_map = {code: onet_to_soc(code) for code in measures.occupation_codes}

    # Binary baseline (re-run to get comparable overlap values)
    print("      Running Binary baseline...")
    binary_result = run_regression(
        binary_overlap,
        measures.occupation_codes,
        comovement.comovement_matrix,
        comovement.occupation_codes,
        crosswalk_map,
    )

    # SAE
    print("      Running SAE regression...")
    sae_result = run_regression(
        sae_overlap,
        measures.occupation_codes,
        comovement.comovement_matrix,
        comovement.occupation_codes,
        crosswalk_map,
    )

    # Comparison
    print("\n" + "-" * 60)
    print("REGRESSION RESULTS")
    print("-" * 60)

    print(f"\n{'Metric':<20} {'Binary':<15} {'SAE':<15} {'Δ':<15}")
    print("-" * 60)
    print(f"{'β':<20} {binary_result['beta']:<15.4f} {sae_result['beta']:<15.4f} {sae_result['beta'] - binary_result['beta']:<+15.4f}")
    print(f"{'SE':<20} {binary_result['se']:<15.4f} {sae_result['se']:<15.4f}")
    print(f"{'t-stat':<20} {binary_result['t_stat']:<15.2f} {sae_result['t_stat']:<15.2f}")
    print(f"{'p-value':<20} {binary_result['pvalue']:<15.4f} {sae_result['pvalue']:<15.4f}")
    print(f"{'R²':<20} {binary_result['r_squared']:<15.6f} {sae_result['r_squared']:<15.6f} {sae_result['r_squared'] - binary_result['r_squared']:<+15.6f}")
    print(f"{'n_pairs':<20} {binary_result['n_pairs']:<15} {sae_result['n_pairs']:<15}")

    # Success criteria
    beta_improvement = (sae_result['beta'] - binary_result['beta']) / binary_result['beta']
    r2_improvement = sae_result['r_squared'] - binary_result['r_squared']

    print("\n" + "-" * 60)
    print("SUCCESS CRITERIA")
    print("-" * 60)
    print(f"SAE β > 0, p < 0.05:        {'✓ PASS' if sae_result['beta'] > 0 and sae_result['pvalue'] < 0.05 else '✗ FAIL'}")
    print(f"β improvement > 20%:        {'✓ PASS' if beta_improvement > 0.20 else '✗ FAIL'} ({beta_improvement:+.1%})")
    print(f"R² improvement > 0.0005:    {'✓ PASS' if r2_improvement > 0.0005 else '✗ FAIL'} ({r2_improvement:+.6f})")

    # Overall verdict
    sae_wins = (
        sae_result['beta'] > 0 and
        sae_result['pvalue'] < 0.05 and
        beta_improvement > 0.20 and
        r2_improvement > 0.0005
    )

    # Save results
    results = {
        "overlap_correlation": {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "interpretation": corr_interpretation,
        },
        "binary_baseline": {k: v for k, v in binary_result.items() if k != "overlap_values"},
        "sae_result": {k: v for k, v in sae_result.items() if k != "overlap_values"},
        "comparison": {
            "beta_improvement_pct": float(beta_improvement * 100),
            "r2_improvement": float(r2_improvement),
            "sae_beats_baseline": bool(sae_wins),
        },
        "verdict": "SAE adds value" if sae_wins else "Binary baseline sufficient",
    }

    with open(output_dir / "validation_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save overlap matrices for further analysis
    np.save(output_dir / "sae_overlap.npy", sae_overlap)
    np.save(output_dir / "occupation_features.npy", occupation_features)

    print(f"\n      Results saved to {output_dir}")

    # Final verdict
    print("\n" + "=" * 60)
    print("PHASE D VERDICT")
    print("=" * 60)

    if sae_wins:
        print("✓ SAE WINS: Sparse features outperform raw binary overlap")
        print(f"  β improvement: {beta_improvement:+.1%}")
        print(f"  R² improvement: {r2_improvement:+.6f}")
        print("  → Sparsity hypothesis validated")
        print("  → Proceed to Phase II with SAE-based overlap")
    else:
        print("✗ BINARY WINS: Raw activity overlap is sufficient")
        if sae_result['beta'] > 0 and sae_result['pvalue'] < 0.05:
            print(f"  SAE is significant (β={sae_result['beta']:.4f}, p={sae_result['pvalue']:.4f})")
            print(f"  But improvement is marginal: β +{beta_improvement:.1%}, R² +{r2_improvement:.6f}")
        else:
            print(f"  SAE failed significance test")
        print("  → Binary overlap captures the economic structure")
        print("  → Consider: supervised probing, different validation target")

    return 0 if sae_wins else 1


if __name__ == "__main__":
    sys.exit(main())
