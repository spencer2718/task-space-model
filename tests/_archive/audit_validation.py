"""
Audit the v0.4.2 validation results.

Three checks for the "Suspiciously Perfect" result:
1. Histogram of pairwise distances - is the distribution collapsed?
2. Overlap collinearity - are p10 and p90 overlaps perfectly correlated?
3. Diagonal dominance - does the result vanish without self-overlap?
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    print("=" * 70)
    print("VALIDATION AUDIT: Checking for 'Suspiciously Perfect' Results")
    print("=" * 70)

    from task_space import (
        build_dwa_activity_domain,
        build_dwa_occupation_measures,
        compute_text_embedding_distances,
        build_kernel_matrix,
        compute_overlap,
        distance_percentiles,
        SIGMA_PERCENTILES,
    )
    from task_space.crosswalk import (
        build_onet_oes_crosswalk,
        load_oes_panel,
        compute_wage_comovement,
        aggregate_occupation_measures,
    )
    from task_space.domain import OccupationMeasures
    from task_space.validation import _cluster_se
    from scipy.stats import t as t_dist, pearsonr, spearmanr
    import pandas as pd

    # Build manifold
    print("\n[Setup] Building DWA manifold...")
    domain = build_dwa_activity_domain()
    measures = build_dwa_occupation_measures()
    titles = list(domain.activity_names.values())
    distances = compute_text_embedding_distances(titles, domain.activity_ids)
    pcts = distance_percentiles(distances)

    # =========================================================================
    # CHECK 1: Histogram of pairwise distances
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 1: Distance Distribution (Curse of Dimensionality?)")
    print("=" * 70)

    n = len(distances.activity_ids)
    triu_idx = np.triu_indices(n, k=1)
    pairwise_dists = distances.distance_matrix[triu_idx]

    print(f"\nPairwise distance statistics (n={len(pairwise_dists):,} pairs):")
    print(f"  Min:    {pairwise_dists.min():.4f}")
    print(f"  p10:    {np.percentile(pairwise_dists, 10):.4f}")
    print(f"  p25:    {np.percentile(pairwise_dists, 25):.4f}")
    print(f"  Median: {np.percentile(pairwise_dists, 50):.4f}")
    print(f"  p75:    {np.percentile(pairwise_dists, 75):.4f}")
    print(f"  p90:    {np.percentile(pairwise_dists, 90):.4f}")
    print(f"  Max:    {pairwise_dists.max():.4f}")
    print(f"  Mean:   {pairwise_dists.mean():.4f}")
    print(f"  Std:    {pairwise_dists.std():.4f}")

    # Coefficient of variation
    cv = pairwise_dists.std() / pairwise_dists.mean()
    print(f"\n  Coefficient of Variation: {cv:.4f}")

    # IQR / Median ratio
    iqr = np.percentile(pairwise_dists, 75) - np.percentile(pairwise_dists, 25)
    iqr_ratio = iqr / np.percentile(pairwise_dists, 50)
    print(f"  IQR/Median ratio: {iqr_ratio:.4f}")

    if cv < 0.15:
        print("\n  ⚠️  WARNING: Low variance in distances (CV < 0.15)")
        print("     This suggests 'curse of dimensionality' - all pairs ~equally distant")
    else:
        print("\n  ✓  Distance variance looks reasonable (CV >= 0.15)")

    # Save histogram
    plt.figure(figsize=(10, 6))
    plt.hist(pairwise_dists, bins=100, density=True, alpha=0.7, edgecolor='black')
    plt.axvline(pcts['p10'], color='red', linestyle='--', label=f"p10={pcts['p10']:.3f}")
    plt.axvline(pcts['p50'], color='green', linestyle='--', label=f"p50={pcts['p50']:.3f}")
    plt.axvline(pcts['p90'], color='blue', linestyle='--', label=f"p90={pcts['p90']:.3f}")
    plt.xlabel('Cosine Distance')
    plt.ylabel('Density')
    plt.title('Distribution of Pairwise DWA Distances (Recipe Y)')
    plt.legend()
    plt.savefig('outputs/phase_i_dwa/distance_histogram.png', dpi=150, bbox_inches='tight')
    print(f"\n  Histogram saved to outputs/phase_i_dwa/distance_histogram.png")

    # =========================================================================
    # CHECK 2: Overlap collinearity across sigma values
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 2: Overlap Collinearity (Are different σ values redundant?)")
    print("=" * 70)

    # Load validation data
    print("\n  Loading validation data...")
    oes_panel = load_oes_panel([2019, 2020, 2021, 2022, 2023])
    comovement = compute_wage_comovement(oes_panel, min_years=4)
    crosswalk = build_onet_oes_crosswalk(measures.occupation_codes, comovement.occupation_codes)

    # Aggregate measures to SOC level
    agg_matrix, agg_codes = aggregate_occupation_measures(
        measures.occupation_matrix,
        measures.occupation_codes,
        crosswalk,
    )

    # Find common occupations
    common_codes = sorted(set(agg_codes) & set(comovement.occupation_codes))
    agg_idx = [agg_codes.index(c) for c in common_codes]

    agg_measures = OccupationMeasures(
        occupation_codes=[agg_codes[i] for i in agg_idx],
        occupation_matrix=agg_matrix[agg_idx],
        activity_ids=measures.activity_ids,
        raw_matrix=agg_matrix[agg_idx],
    )

    # Compute overlaps at p10 and p90
    print("\n  Computing overlaps at p10 and p90...")

    kernel_p10 = build_kernel_matrix(distances, sigma=pcts['p10'])
    kernel_p90 = build_kernel_matrix(distances, sigma=pcts['p90'])

    overlap_p10 = compute_overlap(agg_measures, kernel_p10)
    overlap_p90 = compute_overlap(agg_measures, kernel_p90)

    # Extract upper triangle (off-diagonal pairs)
    n_occ = len(agg_measures.occupation_codes)
    triu_occ = np.triu_indices(n_occ, k=1)

    overlap_p10_flat = overlap_p10[triu_occ]
    overlap_p90_flat = overlap_p90[triu_occ]

    # Compute correlations
    pearson_r, _ = pearsonr(overlap_p10_flat, overlap_p90_flat)
    spearman_r, _ = spearmanr(overlap_p10_flat, overlap_p90_flat)

    print(f"\n  Correlation between Overlap(p10) and Overlap(p90):")
    print(f"    Pearson r:  {pearson_r:.6f}")
    print(f"    Spearman ρ: {spearman_r:.6f}")

    if pearson_r > 0.99:
        print("\n  ⚠️  WARNING: Overlaps are nearly perfectly correlated (r > 0.99)")
        print("     Changing σ is NOT changing the structure - just rescaling!")
    elif pearson_r > 0.95:
        print("\n  ⚠️  CAUTION: Very high correlation (r > 0.95)")
        print("     Limited variation in overlap structure across σ values")
    else:
        print("\n  ✓  Correlation is acceptable (r < 0.95)")

    # =========================================================================
    # CHECK 3: Diagonal Dominance (The Smoking Gun)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 3: Diagonal Dominance (Does result vanish without self-overlap?)")
    print("=" * 70)

    # Get comovement for common occupations
    com_idx = [comovement.occupation_codes.index(c) for c in common_codes]
    comovement_common = comovement.comovement_matrix[np.ix_(com_idx, com_idx)]

    print("\n  Running regression WITH diagonal (original)...")

    # Original regression (with diagonal in kernel)
    sigma = pcts['p50']
    kernel = build_kernel_matrix(distances, sigma=sigma)
    overlap = compute_overlap(agg_measures, kernel)

    # Build pair dataset (upper triangle, excluding diagonal)
    pairs_orig = []
    for i in range(n_occ):
        for j in range(i + 1, n_occ):
            pairs_orig.append({
                "occ_i": common_codes[i],
                "occ_j": common_codes[j],
                "overlap": overlap[i, j],
                "y": comovement_common[i, j],
            })

    df_orig = pd.DataFrame(pairs_orig).dropna()

    X_orig = df_orig[["overlap"]].values
    y_orig = df_orig["y"].values
    clusters_orig = df_orig["occ_i"].values

    beta_orig, se_orig, r2_orig, n_clusters = _cluster_se(X_orig, y_orig, clusters_orig)
    t_orig = beta_orig / se_orig
    p_orig = 2 * (1 - t_dist.cdf(abs(t_orig), n_clusters - 1))

    print(f"    β = {beta_orig:.4f}, SE = {se_orig:.4f}")
    print(f"    t = {t_orig:.4f}, p = {p_orig:.6f}")
    print(f"    R² = {r2_orig:.6f}")

    # Now: Zero out diagonal of kernel and re-run
    print("\n  Running regression WITHOUT diagonal (zeroed kernel diagonal)...")

    # Build kernel with zeroed diagonal
    kernel_raw = np.exp(-distances.distance_matrix / sigma)
    np.fill_diagonal(kernel_raw, 0)  # Zero out self-similarity

    # Row-normalize
    row_sums = kernel_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    kernel_no_diag = kernel_raw / row_sums

    # Compute overlap with modified kernel
    rho = agg_measures.occupation_matrix
    overlap_no_diag = rho @ kernel_no_diag @ rho.T

    # Symmetrize
    overlap_no_diag = (overlap_no_diag + overlap_no_diag.T) / 2

    # Build pair dataset
    pairs_no_diag = []
    for i in range(n_occ):
        for j in range(i + 1, n_occ):
            pairs_no_diag.append({
                "occ_i": common_codes[i],
                "occ_j": common_codes[j],
                "overlap": overlap_no_diag[i, j],
                "y": comovement_common[i, j],
            })

    df_no_diag = pd.DataFrame(pairs_no_diag).dropna()

    X_no_diag = df_no_diag[["overlap"]].values
    y_no_diag = df_no_diag["y"].values
    clusters_no_diag = df_no_diag["occ_i"].values

    beta_no_diag, se_no_diag, r2_no_diag, n_clusters = _cluster_se(X_no_diag, y_no_diag, clusters_no_diag)
    t_no_diag = beta_no_diag / se_no_diag
    p_no_diag = 2 * (1 - t_dist.cdf(abs(t_no_diag), n_clusters - 1))

    print(f"    β = {beta_no_diag:.4f}, SE = {se_no_diag:.4f}")
    print(f"    t = {t_no_diag:.4f}, p = {p_no_diag:.6f}")
    print(f"    R² = {r2_no_diag:.6f}")

    # Interpret
    print("\n  COMPARISON:")
    print(f"    t-stat WITH diagonal:    {t_orig:.4f}")
    print(f"    t-stat WITHOUT diagonal: {t_no_diag:.4f}")

    if abs(t_no_diag) < 2.0:
        print("\n  ⚠️  SMOKING GUN: Result VANISHES without diagonal!")
        print("     The 'geometric spillover' is a mirage.")
        print("     You're just rediscovering 'self-similarity predicts self-similarity'")
    elif abs(t_no_diag) < abs(t_orig) * 0.5:
        print("\n  ⚠️  WARNING: Result is substantially weaker without diagonal")
        print("     Diagonal (self-overlap) is dominating the signal")
    else:
        print("\n  ✓  Result SURVIVES without diagonal!")
        print("     Off-diagonal spillovers are driving the effect")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    issues = []
    if cv < 0.15:
        issues.append("Distance distribution is collapsed (curse of dimensionality)")
    if pearson_r > 0.99:
        issues.append("Overlaps perfectly correlated across σ (no real bandwidth variation)")
    if abs(t_no_diag) < 2.0:
        issues.append("Result vanishes without diagonal (diagonal dominance)")

    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print("\n  The validation result may be spurious.")
    else:
        print("\n✓  No major issues found. Result appears robust.")

    return {
        "cv": cv,
        "pearson_r": pearson_r,
        "t_with_diag": t_orig,
        "t_without_diag": t_no_diag,
    }


if __name__ == "__main__":
    main()
