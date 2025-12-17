"""
Robustness checks for v0.4.2.1 validation audit.

Three main checks:
1. Permutation test - does the effect depend on actual occupation-activity structure?
2. Random distance placebo - does the effect require meaningful distances?
3. Jackknife stability - is the effect stable to dropping occupations?
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _compute_overlap_from_measures(measures: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Compute overlap matrix from measures and kernel."""
    overlap = measures @ kernel @ measures.T
    return (overlap + overlap.T) / 2  # Symmetrize


def _run_regression(overlap: np.ndarray, comovement: np.ndarray, clusters: np.ndarray) -> dict:
    """Run OLS regression with clustered standard errors."""
    from task_space.validation import _cluster_se

    n = overlap.shape[0]
    triu_idx = np.triu_indices(n, k=1)

    X = overlap[triu_idx].reshape(-1, 1)
    y = comovement[triu_idx]

    # Create cluster labels (use first occupation in each pair)
    cluster_labels = clusters[triu_idx[0]]

    # Remove NaN
    valid = ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    cluster_labels = cluster_labels[valid]

    beta, se, r2, n_clusters = _cluster_se(X, y, cluster_labels)
    t_stat = beta / se
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), n_clusters - 1))

    return {
        "beta": float(beta),
        "se": float(se),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "r2": float(r2),
        "n_clusters": int(n_clusters),
    }


def run_permutation_test(
    measures: np.ndarray,
    kernel: np.ndarray,
    comovement: np.ndarray,
    clusters: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
    verbose: bool = True
) -> dict:
    """
    Permutation test for validation regression.

    Shuffles activity weights within each occupation to break
    the activity-occupation structure while preserving marginals.

    Args:
        measures: (n_occ, n_act) occupation-activity matrix
        kernel: (n_act, n_act) kernel matrix
        comovement: (n_occ, n_occ) wage comovement matrix
        clusters: (n_occ,) cluster labels
        n_permutations: Number of permutations
        seed: Random seed
        verbose: Print progress

    Returns:
        Dict with observed_beta, null distribution stats, p_value
    """
    np.random.seed(seed)

    # Observed statistic
    overlap = _compute_overlap_from_measures(measures, kernel)
    observed = _run_regression(overlap, comovement, clusters)
    observed_beta = observed["beta"]

    if verbose:
        print(f"Observed β = {observed_beta:.4f}")
        print(f"Running {n_permutations} permutations...")

    # Null distribution
    null_betas = []
    for i in range(n_permutations):
        # Permute within rows (within occupations)
        perm_measures = measures.copy()
        for j in range(perm_measures.shape[0]):
            np.random.shuffle(perm_measures[j, :])

        # Re-normalize to probability measures
        row_sums = perm_measures.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        perm_measures = perm_measures / row_sums

        # Compute overlap and regression
        perm_overlap = _compute_overlap_from_measures(perm_measures, kernel)
        perm_result = _run_regression(perm_overlap, comovement, clusters)
        null_betas.append(perm_result["beta"])

        if verbose and (i + 1) % 100 == 0:
            print(f"  Permutation {i+1}/{n_permutations}")

    null_betas = np.array(null_betas)

    # P-value (proportion of null >= observed)
    p_value = (np.sum(null_betas >= observed_beta) + 1) / (n_permutations + 1)

    # Z-score
    null_std = np.std(null_betas)
    z_score = (observed_beta - np.mean(null_betas)) / null_std if null_std > 0 else np.inf

    return {
        "observed_beta": float(observed_beta),
        "null_mean": float(np.mean(null_betas)),
        "null_std": float(null_std),
        "null_percentiles": {
            "p5": float(np.percentile(null_betas, 5)),
            "p25": float(np.percentile(null_betas, 25)),
            "p50": float(np.percentile(null_betas, 50)),
            "p75": float(np.percentile(null_betas, 75)),
            "p95": float(np.percentile(null_betas, 95)),
            "p99": float(np.percentile(null_betas, 99)),
        },
        "p_value": float(p_value),
        "z_score": float(z_score),
        "n_permutations": n_permutations,
    }


def run_placebo_test(
    measures: np.ndarray,
    true_distances: np.ndarray,
    comovement: np.ndarray,
    clusters: np.ndarray,
    sigma: float,
    n_trials: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> dict:
    """
    Placebo test: replace true distances with shuffled distances.

    Shuffles the distance values while preserving the marginal distribution.

    Args:
        measures: (n_occ, n_act) occupation-activity matrix
        true_distances: (n_act, n_act) true distance matrix
        comovement: (n_occ, n_occ) wage comovement matrix
        clusters: (n_occ,) cluster labels
        sigma: Kernel bandwidth
        n_trials: Number of placebo trials
        seed: Random seed
        verbose: Print progress

    Returns:
        Dict with true_beta, placebo distribution stats
    """
    np.random.seed(seed)
    n_activities = measures.shape[1]

    # Build kernel from true distances
    true_kernel_raw = np.exp(-true_distances / sigma)
    row_sums = true_kernel_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    true_kernel = true_kernel_raw / row_sums

    # Observed with true distances
    true_overlap = _compute_overlap_from_measures(measures, true_kernel)
    true_result = _run_regression(true_overlap, comovement, clusters)

    if verbose:
        print(f"True β = {true_result['beta']:.4f}")
        print(f"Running {n_trials} placebo trials...")

    # Placebo distribution
    placebo_betas = []
    for i in range(n_trials):
        # Get upper triangle values and shuffle them
        triu_idx = np.triu_indices(n_activities, k=1)
        dist_values = true_distances[triu_idx].copy()
        np.random.shuffle(dist_values)

        # Rebuild symmetric distance matrix
        random_dist = np.zeros((n_activities, n_activities))
        random_dist[triu_idx] = dist_values
        random_dist = random_dist + random_dist.T

        # Build kernel
        random_kernel_raw = np.exp(-random_dist / sigma)
        row_sums = random_kernel_raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        random_kernel = random_kernel_raw / row_sums

        # Compute overlap and regression
        random_overlap = _compute_overlap_from_measures(measures, random_kernel)
        random_result = _run_regression(random_overlap, comovement, clusters)
        placebo_betas.append(random_result["beta"])

        if verbose and (i + 1) % 20 == 0:
            print(f"  Placebo {i+1}/{n_trials}")

    placebo_betas = np.array(placebo_betas)

    # Compute ratio and p-value
    placebo_mean = np.mean(placebo_betas)
    ratio = true_result["beta"] / placebo_mean if placebo_mean != 0 else np.inf
    p_value = (np.sum(placebo_betas >= true_result["beta"]) + 1) / (n_trials + 1)

    return {
        "true_beta": float(true_result["beta"]),
        "placebo_mean": float(placebo_mean),
        "placebo_std": float(np.std(placebo_betas)),
        "placebo_percentiles": {
            "p5": float(np.percentile(placebo_betas, 5)),
            "p50": float(np.percentile(placebo_betas, 50)),
            "p95": float(np.percentile(placebo_betas, 95)),
        },
        "ratio": float(ratio),
        "p_value": float(p_value),
        "n_trials": n_trials,
    }


def run_jackknife_stability(
    overlap: np.ndarray,
    comovement: np.ndarray,
    clusters: np.ndarray,
    drop_fraction: float = 0.10,
    n_trials: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> dict:
    """
    Jackknife stability test.

    Repeatedly drop random subsets of occupations and re-run validation.

    Args:
        overlap: (n_occ, n_occ) overlap matrix
        comovement: (n_occ, n_occ) wage comovement matrix
        clusters: (n_occ,) cluster labels
        drop_fraction: Fraction of occupations to drop
        n_trials: Number of jackknife trials
        seed: Random seed
        verbose: Print progress

    Returns:
        Dict with beta distribution stats
    """
    np.random.seed(seed)
    n_occ = overlap.shape[0]
    n_drop = int(n_occ * drop_fraction)

    if verbose:
        print(f"Running {n_trials} jackknife trials (dropping {n_drop}/{n_occ} occupations)...")

    betas = []
    for i in range(n_trials):
        # Sample indices to keep
        keep_idx = np.random.choice(n_occ, n_occ - n_drop, replace=False)
        keep_idx = np.sort(keep_idx)

        # Subset matrices
        subset_overlap = overlap[np.ix_(keep_idx, keep_idx)]
        subset_comovement = comovement[np.ix_(keep_idx, keep_idx)]
        subset_clusters = clusters[keep_idx]

        # Run regression
        result = _run_regression(subset_overlap, subset_comovement, subset_clusters)
        betas.append(result["beta"])

        if verbose and (i + 1) % 20 == 0:
            print(f"  Jackknife {i+1}/{n_trials}")

    betas = np.array(betas)

    return {
        "mean_beta": float(np.mean(betas)),
        "std_beta": float(np.std(betas)),
        "cv": float(np.std(betas) / np.mean(betas)) if np.mean(betas) != 0 else np.inf,
        "min_beta": float(np.min(betas)),
        "max_beta": float(np.max(betas)),
        "percentiles": {
            "p5": float(np.percentile(betas, 5)),
            "p25": float(np.percentile(betas, 25)),
            "p50": float(np.percentile(betas, 50)),
            "p75": float(np.percentile(betas, 75)),
            "p95": float(np.percentile(betas, 95)),
        },
        "all_positive": bool(np.all(betas > 0)),
        "n_positive": int(np.sum(betas > 0)),
        "n_trials": n_trials,
        "drop_fraction": drop_fraction,
    }


def main():
    """Run all robustness checks and save results."""
    print("=" * 70)
    print("v0.4.2.1 ROBUSTNESS CHECKS")
    print("=" * 70)

    from task_space import (
        build_dwa_activity_domain,
        build_dwa_occupation_measures,
        compute_text_embedding_distances,
        build_kernel_matrix,
        compute_overlap,
        distance_percentiles,
    )
    from task_space.crosswalk import (
        build_onet_oes_crosswalk,
        load_oes_panel,
        compute_wage_comovement,
        aggregate_occupation_measures,
    )
    from task_space.domain import OccupationMeasures

    # =========================================================================
    # Setup: Build manifold and load data
    # =========================================================================
    print("\n[Setup] Building DWA manifold...")
    domain = build_dwa_activity_domain()
    measures = build_dwa_occupation_measures()
    titles = list(domain.activity_names.values())
    distances = compute_text_embedding_distances(titles, domain.activity_ids)
    pcts = distance_percentiles(distances)
    sigma = pcts['p50']

    print(f"  Domain: {domain.n_activities} DWAs")
    print(f"  Sigma (p50): {sigma:.4f}")

    print("\n[Setup] Loading validation data...")
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
    com_idx = [comovement.occupation_codes.index(c) for c in common_codes]

    # Extract aligned matrices
    agg_measures_matrix = agg_matrix[agg_idx]
    comovement_matrix = comovement.comovement_matrix[np.ix_(com_idx, com_idx)]
    clusters = np.array(common_codes)

    print(f"  Common occupations: {len(common_codes)}")

    # Build kernel
    kernel = build_kernel_matrix(distances, sigma=sigma)

    results = {
        "setup": {
            "n_activities": domain.n_activities,
            "n_occupations": len(common_codes),
            "sigma": sigma,
        }
    }

    # =========================================================================
    # Check 1: Permutation Test
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 1: Permutation Test (n=1000)")
    print("=" * 70)

    perm_result = run_permutation_test(
        measures=agg_measures_matrix,
        kernel=kernel.matrix,
        comovement=comovement_matrix,
        clusters=clusters,
        n_permutations=1000,
        seed=42,
        verbose=True
    )

    print(f"\nResults:")
    print(f"  Observed β: {perm_result['observed_beta']:.4f}")
    print(f"  Null mean:  {perm_result['null_mean']:.4f}")
    print(f"  Null std:   {perm_result['null_std']:.4f}")
    print(f"  Z-score:    {perm_result['z_score']:.2f}")
    print(f"  p-value:    {perm_result['p_value']:.6f}")

    if perm_result['p_value'] < 0.01:
        print("  ✓  PASS: Effect is significantly different from null (p < 0.01)")
    else:
        print("  ⚠️  FAIL: Effect is not distinguishable from permutation null")

    results["permutation_test"] = perm_result

    # =========================================================================
    # Check 2: Random Distance Placebo
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 2: Random Distance Placebo (n=100)")
    print("=" * 70)

    placebo_result = run_placebo_test(
        measures=agg_measures_matrix,
        true_distances=distances.distance_matrix,
        comovement=comovement_matrix,
        clusters=clusters,
        sigma=sigma,
        n_trials=100,
        seed=42,
        verbose=True
    )

    print(f"\nResults:")
    print(f"  True β:     {placebo_result['true_beta']:.4f}")
    print(f"  Placebo mean: {placebo_result['placebo_mean']:.4f}")
    print(f"  Placebo std:  {placebo_result['placebo_std']:.4f}")
    print(f"  Ratio:      {placebo_result['ratio']:.2f}x")
    print(f"  p-value:    {placebo_result['p_value']:.6f}")

    if placebo_result['ratio'] > 2.0 and placebo_result['p_value'] < 0.05:
        print("  ✓  PASS: True distances give much stronger effect than random")
    else:
        print("  ⚠️  Concern: True distances not clearly better than random")

    results["placebo_test"] = placebo_result

    # =========================================================================
    # Check 3: Jackknife Stability
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 3: Jackknife Stability (n=100, drop 10%)")
    print("=" * 70)

    # First compute the overlap matrix
    agg_measures = OccupationMeasures(
        occupation_codes=[agg_codes[i] for i in agg_idx],
        occupation_matrix=agg_measures_matrix,
        activity_ids=measures.activity_ids,
        raw_matrix=agg_measures_matrix,
    )
    overlap = compute_overlap(agg_measures, kernel)

    jackknife_result = run_jackknife_stability(
        overlap=overlap,
        comovement=comovement_matrix,
        clusters=clusters,
        drop_fraction=0.10,
        n_trials=100,
        seed=42,
        verbose=True
    )

    print(f"\nResults:")
    print(f"  Mean β:     {jackknife_result['mean_beta']:.4f}")
    print(f"  Std β:      {jackknife_result['std_beta']:.4f}")
    print(f"  CV:         {jackknife_result['cv']:.4f}")
    print(f"  Range:      [{jackknife_result['min_beta']:.4f}, {jackknife_result['max_beta']:.4f}]")
    print(f"  All positive: {jackknife_result['all_positive']} ({jackknife_result['n_positive']}/100)")

    if jackknife_result['cv'] < 0.3 and jackknife_result['all_positive']:
        print("  ✓  PASS: Effect is stable (CV < 0.3, all positive)")
    elif jackknife_result['all_positive']:
        print("  ⚠️  Partial: All positive but CV >= 0.3")
    else:
        print("  ⚠️  FAIL: Effect is fragile")

    results["jackknife_stability"] = jackknife_result

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("ROBUSTNESS CHECK SUMMARY")
    print("=" * 70)

    checks_passed = 0
    checks_total = 3

    # Permutation test
    if perm_result['p_value'] < 0.01:
        print("  1. Permutation Test:     ✓ PASS")
        checks_passed += 1
    else:
        print("  1. Permutation Test:     ⚠️ FAIL")

    # Placebo test
    if placebo_result['ratio'] > 2.0 and placebo_result['p_value'] < 0.05:
        print("  2. Placebo Test:         ✓ PASS")
        checks_passed += 1
    else:
        print("  2. Placebo Test:         ⚠️ FAIL")

    # Jackknife stability
    if jackknife_result['cv'] < 0.3 and jackknife_result['all_positive']:
        print("  3. Jackknife Stability:  ✓ PASS")
        checks_passed += 1
    else:
        print("  3. Jackknife Stability:  ⚠️ FAIL")

    print(f"\n  Overall: {checks_passed}/{checks_total} checks passed")

    results["summary"] = {
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "permutation_pass": perm_result['p_value'] < 0.01,
        "placebo_pass": placebo_result['ratio'] > 2.0 and placebo_result['p_value'] < 0.05,
        "jackknife_pass": jackknife_result['cv'] < 0.3 and jackknife_result['all_positive'],
    }

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs" / "phase_i_dwa"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_dir / 'robustness_results.json'}")

    return results


if __name__ == "__main__":
    main()
