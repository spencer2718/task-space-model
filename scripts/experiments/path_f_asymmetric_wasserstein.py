#!/usr/bin/env python3
"""
Path F: Asymmetric Barriers Test with Wasserstein Distance (v0.6.8.2)

Re-runs the directional barriers test using Wasserstein distance instead of
kernel overlap to verify whether the symmetric barrier finding (beta_up/beta_down = 1.04)
is geometry-robust.

Context:
- Original asymmetric test (v0.6.6.0) used kernel-based d_sem_census.npz
- Wasserstein is now validated primary geometry (ΔLL = +9,576)
- Need to verify symmetric finding holds under Wasserstein

Output: outputs/experiments/mobility_asymmetric_wasserstein_v0682.json

Usage:
    PYTHONPATH=src python scripts/experiments/path_f_asymmetric_wasserstein.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from task_space.mobility import (
    build_asymmetric_institutional_distance,
    verify_asymmetric_decomposition,
    load_census_onet_crosswalk,
    aggregate_distances_to_census,
    load_verified_transitions,
    build_choice_dataset,
    fit_conditional_logit,
    build_asymmetric_choice_dataset,
    fit_asymmetric_conditional_logit,
    compute_odds_ratios,
    compute_asymmetric_odds_ratios,
)


# Paths
ONET_PATH = Path("data/onet/db_30_0_excel")
CACHE_PATH = Path(".cache/artifacts/v1")
OUTPUT_PATH = Path("outputs/experiments/mobility_asymmetric_wasserstein_v0682.json")


def load_wasserstein_distance_onet() -> tuple[np.ndarray, list]:
    """Load Wasserstein distance matrix at O*NET level (894x894)."""
    cached = CACHE_PATH / "wasserstein" / "d_wasserstein_onet.npz"
    if not cached.exists():
        raise FileNotFoundError(
            f"Wasserstein distance not found at {cached}. "
            "Run Wasserstein computation first."
        )

    data = np.load(cached, allow_pickle=True)
    d_wass = data["distance_matrix"]
    occ_codes = data["occupation_codes"].tolist()

    print(f"  Loaded Wasserstein O*NET: shape={d_wass.shape}")
    print(f"    mean={d_wass.mean():.4f}, std={d_wass.std():.4f}")
    print(f"    min={d_wass.min():.4f}, max={d_wass.max():.4f}")

    return d_wass, occ_codes


def load_kernel_distance_onet() -> tuple[np.ndarray, list]:
    """Load kernel distance matrix for comparison."""
    cached = CACHE_PATH / "mobility" / "d_sem_census.npz"
    if not cached.exists():
        raise FileNotFoundError(f"Kernel distance not found at {cached}")

    data = np.load(cached, allow_pickle=True)
    d_kern = data["d_sem"]
    occ_codes = data["occ_codes"].tolist()

    return d_kern, occ_codes


def compute_distance_stats(d: np.ndarray, name: str) -> dict:
    """Compute summary statistics for a distance matrix."""
    n = d.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = d[mask]

    return {
        "name": name,
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "frac_zero": float(np.mean(vals == 0)),
    }


def compute_distance_correlations(
    d_sem: np.ndarray,
    d_up: np.ndarray,
    d_down: np.ndarray,
    d_sym: np.ndarray,
) -> dict:
    """Compute correlations between distance measures (off-diagonal only)."""
    n = d_sem.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    sem_flat = d_sem[mask]
    up_flat = d_up[mask]
    down_flat = d_down[mask]
    sym_flat = d_sym[mask]

    return {
        "sem_up": float(np.corrcoef(sem_flat, up_flat)[0, 1]),
        "sem_down": float(np.corrcoef(sem_flat, down_flat)[0, 1]),
        "sem_sym": float(np.corrcoef(sem_flat, sym_flat)[0, 1]),
        "up_down": float(np.corrcoef(up_flat, down_flat)[0, 1]),
        "up_sym": float(np.corrcoef(up_flat, sym_flat)[0, 1]),
        "down_sym": float(np.corrcoef(down_flat, sym_flat)[0, 1]),
    }


def main():
    print("=" * 70)
    print("Path F: Asymmetric Barriers Test with WASSERSTEIN Distance")
    print("Version: 0.6.8.2")
    print("=" * 70)
    print()
    print("Objective: Verify symmetric barrier finding (ratio=1.04) under Wasserstein")
    print()

    # Load crosswalk
    print("Loading Census-O*NET crosswalk...")
    crosswalk = load_census_onet_crosswalk()
    print(f"  Coverage: {crosswalk.coverage:.1%} of O*NET codes")
    print(f"  Census occupations: {crosswalk.n_census}")
    print()

    # Load Wasserstein distance (O*NET level)
    print("Loading Wasserstein distance (O*NET level)...")
    d_wass_onet, wass_onet_codes = load_wasserstein_distance_onet()
    print()

    # Load kernel distance for comparison
    print("Loading kernel distance for comparison...")
    d_kern_onet, kern_onet_codes = load_kernel_distance_onet()
    print(f"  Kernel O*NET: mean={d_kern_onet.mean():.4f}, std={d_kern_onet.std():.4f}")
    print()

    # Aggregate Wasserstein to Census level
    print("Aggregating Wasserstein to Census level...")
    d_wass_census, wass_census_codes = aggregate_distances_to_census(
        d_wass_onet,
        wass_onet_codes,
        crosswalk,
        aggregation="mean",
    )
    print(f"  Census matrix shape: {d_wass_census.shape}")

    # Save aggregated Wasserstein matrix
    wass_census_path = CACHE_PATH / "mobility" / "d_wasserstein_census.npz"
    np.savez(
        wass_census_path,
        distances=d_wass_census,
        occupation_codes=np.array(wass_census_codes),
        aggregation_method="mean",
        source="d_wasserstein_onet.npz",
    )
    print(f"  Saved to: {wass_census_path}")
    print()

    # Also aggregate kernel for fair comparison
    print("Aggregating kernel to Census level...")
    d_kern_census, kern_census_codes = aggregate_distances_to_census(
        d_kern_onet,
        kern_onet_codes,
        crosswalk,
        aggregation="mean",
    )
    print(f"  Census matrix shape: {d_kern_census.shape}")
    print()

    # Build asymmetric institutional distances
    print("Building asymmetric institutional distances...")
    asym_result = build_asymmetric_institutional_distance(ONET_PATH)

    # Verify decomposition
    verification = verify_asymmetric_decomposition(asym_result)
    print(f"  Decomposition valid: {verification['all_properties_pass']}")

    # Aggregate institutional distances to Census level
    print("  Aggregating to Census level...")
    d_up_census, up_census_codes = aggregate_distances_to_census(
        asym_result.d_up,
        asym_result.occupations,
        crosswalk,
        aggregation="mean",
    )

    d_down_census, _ = aggregate_distances_to_census(
        asym_result.d_down,
        asym_result.occupations,
        crosswalk,
        aggregation="mean",
    )

    d_sym_census, _ = aggregate_distances_to_census(
        asym_result.d_symmetric,
        asym_result.occupations,
        crosswalk,
        aggregation="mean",
    )
    print(f"  Institutional matrix shape: {d_up_census.shape}")
    print()

    # Align census codes between semantic and institutional
    if wass_census_codes != up_census_codes:
        common_codes = sorted(set(wass_census_codes) & set(up_census_codes))
        print(f"  Warning: Census codes differ. Using {len(common_codes)} common codes.")

        wass_idx = [wass_census_codes.index(c) for c in common_codes]
        inst_idx = [up_census_codes.index(c) for c in common_codes]
        kern_idx = [kern_census_codes.index(c) for c in common_codes]

        d_wass_census = d_wass_census[np.ix_(wass_idx, wass_idx)]
        d_kern_census = d_kern_census[np.ix_(kern_idx, kern_idx)]
        d_up_census = d_up_census[np.ix_(inst_idx, inst_idx)]
        d_down_census = d_down_census[np.ix_(inst_idx, inst_idx)]
        d_sym_census = d_sym_census[np.ix_(inst_idx, inst_idx)]
        census_codes = common_codes
    else:
        census_codes = wass_census_codes

    print(f"Final matrix dimension: {len(census_codes)} occupations")
    print()

    # Wasserstein distance statistics
    print("=" * 70)
    print("DISTANCE MATRIX STATISTICS")
    print("=" * 70)
    stats = {
        "d_wasserstein": compute_distance_stats(d_wass_census, "wasserstein"),
        "d_kernel": compute_distance_stats(d_kern_census, "kernel"),
        "d_up": compute_distance_stats(d_up_census, "upward"),
        "d_down": compute_distance_stats(d_down_census, "downward"),
        "d_sym": compute_distance_stats(d_sym_census, "symmetric_inst"),
    }

    print("\nWasserstein vs Kernel comparison:")
    print(f"  Wasserstein: mean={stats['d_wasserstein']['mean']:.4f}, "
          f"std={stats['d_wasserstein']['std']:.4f}, "
          f"range=[{stats['d_wasserstein']['min']:.4f}, {stats['d_wasserstein']['max']:.4f}]")
    print(f"  Kernel:      mean={stats['d_kernel']['mean']:.4f}, "
          f"std={stats['d_kernel']['std']:.4f}, "
          f"range=[{stats['d_kernel']['min']:.4f}, {stats['d_kernel']['max']:.4f}]")
    print()

    # Correlation between Wasserstein and institutional
    print("Correlations (Wasserstein-based):")
    corr_wass = compute_distance_correlations(d_wass_census, d_up_census, d_down_census, d_sym_census)
    for k, v in corr_wass.items():
        print(f"  {k}: {v:.3f}")

    print("\nCorrelations (Kernel-based for comparison):")
    corr_kern = compute_distance_correlations(d_kern_census, d_up_census, d_down_census, d_sym_census)
    for k, v in corr_kern.items():
        print(f"  {k}: {v:.3f}")
    print()

    # Load verified transitions
    print("Loading verified transitions...")
    transitions = load_verified_transitions()
    print(f"  Transitions: {len(transitions):,}")
    print()

    # Build choice datasets
    print("Building choice datasets...")

    # Wasserstein asymmetric dataset
    choice_wass_asym = build_asymmetric_choice_dataset(
        transitions,
        d_wass_census,
        d_up_census,
        d_down_census,
        census_codes,
        n_alternatives=10,
        random_seed=42,
    )
    print(f"  Wasserstein asymmetric: {len(choice_wass_asym):,} rows")

    # Wasserstein symmetric dataset (for comparison)
    choice_wass_sym = build_choice_dataset(
        transitions,
        d_wass_census,
        d_sym_census,
        census_codes,
        n_alternatives=10,
        random_seed=42,
    )
    print(f"  Wasserstein symmetric: {len(choice_wass_sym):,} rows")

    # Kernel asymmetric dataset (for comparison)
    choice_kern_asym = build_asymmetric_choice_dataset(
        transitions,
        d_kern_census,
        d_up_census,
        d_down_census,
        census_codes,
        n_alternatives=10,
        random_seed=42,
    )
    print(f"  Kernel asymmetric: {len(choice_kern_asym):,} rows")
    print()

    # Fit models
    print("=" * 70)
    print("MODEL ESTIMATION")
    print("=" * 70)
    print()

    # Model 1: Wasserstein Symmetric
    print("Model 1: Wasserstein Symmetric (baseline)")
    result_wass_sym = fit_conditional_logit(choice_wass_sym)
    print(f"  α (semantic):       {result_wass_sym.alpha:.4f} (t={result_wass_sym.alpha_t:.1f})")
    print(f"  β (institutional):  {result_wass_sym.beta:.4f} (t={result_wass_sym.beta_t:.1f})")
    print(f"  Log-likelihood:     {result_wass_sym.log_likelihood:.1f}")
    print()

    # Model 2: Wasserstein Asymmetric (PRIMARY)
    print("Model 2: Wasserstein Asymmetric (PRIMARY)")
    result_wass_asym = fit_asymmetric_conditional_logit(choice_wass_asym)
    print(f"  α (semantic):       {result_wass_asym.alpha:.4f} (t={result_wass_asym.alpha_t:.1f})")
    print(f"  β_up (upward):      {result_wass_asym.beta_up:.4f} (t={result_wass_asym.beta_up_t:.1f})")
    print(f"  β_down (downward):  {result_wass_asym.beta_down:.4f} (t={result_wass_asym.beta_down_t:.1f})")
    print(f"  Log-likelihood:     {result_wass_asym.log_likelihood:.1f}")
    print()

    # Model 3: Kernel Asymmetric (for comparison)
    print("Model 3: Kernel Asymmetric (comparison)")
    result_kern_asym = fit_asymmetric_conditional_logit(choice_kern_asym)
    print(f"  α (semantic):       {result_kern_asym.alpha:.4f} (t={result_kern_asym.alpha_t:.1f})")
    print(f"  β_up (upward):      {result_kern_asym.beta_up:.4f} (t={result_kern_asym.beta_up_t:.1f})")
    print(f"  β_down (downward):  {result_kern_asym.beta_down:.4f} (t={result_kern_asym.beta_down_t:.1f})")
    print(f"  Log-likelihood:     {result_kern_asym.log_likelihood:.1f}")
    print()

    # Hypothesis tests
    print("=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)
    print()

    print("WASSERSTEIN (Primary):")
    print(f"  Asymmetry ratio: |β_up| / |β_down| = {result_wass_asym.asymmetry_ratio:.4f}")
    print(f"  LR test (H0: β_up = β_down):")
    print(f"    LR statistic:  {result_wass_asym.lr_test_statistic:.2f}")
    print(f"    p-value:       {result_wass_asym.lr_test_pvalue:.4f}")
    print()

    print("KERNEL (Comparison):")
    print(f"  Asymmetry ratio: |β_up| / |β_down| = {result_kern_asym.asymmetry_ratio:.4f}")
    print(f"  LR test (H0: β_up = β_down):")
    print(f"    LR statistic:  {result_kern_asym.lr_test_statistic:.2f}")
    print(f"    p-value:       {result_kern_asym.lr_test_pvalue:.4f}")
    print()

    # Comparison table
    print("=" * 70)
    print("GEOMETRY COMPARISON: Kernel vs Wasserstein")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'Kernel':<15} {'Wasserstein':<15} {'Δ':<10}")
    print("-" * 65)
    print(f"{'α (semantic)':<25} {result_kern_asym.alpha:<15.4f} {result_wass_asym.alpha:<15.4f} {result_wass_asym.alpha - result_kern_asym.alpha:+.4f}")
    print(f"{'β_up (upward)':<25} {result_kern_asym.beta_up:<15.4f} {result_wass_asym.beta_up:<15.4f} {result_wass_asym.beta_up - result_kern_asym.beta_up:+.4f}")
    print(f"{'β_down (downward)':<25} {result_kern_asym.beta_down:<15.4f} {result_wass_asym.beta_down:<15.4f} {result_wass_asym.beta_down - result_kern_asym.beta_down:+.4f}")
    print(f"{'Asymmetry ratio':<25} {result_kern_asym.asymmetry_ratio:<15.4f} {result_wass_asym.asymmetry_ratio:<15.4f} {result_wass_asym.asymmetry_ratio - result_kern_asym.asymmetry_ratio:+.4f}")
    print(f"{'Log-likelihood':<25} {result_kern_asym.log_likelihood:<15.1f} {result_wass_asym.log_likelihood:<15.1f} {result_wass_asym.log_likelihood - result_kern_asym.log_likelihood:+.1f}")
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    ratio_wass = result_wass_asym.asymmetry_ratio
    ratio_kern = result_kern_asym.asymmetry_ratio
    ratio_diff = abs(ratio_wass - ratio_kern)

    if ratio_wass > 2 and result_wass_asym.lr_test_pvalue < 0.05:
        verdict = "ASYMMETRIC: Wasserstein reveals asymmetry kernel missed"
        robust = False
    elif ratio_wass < 1.5 or result_wass_asym.lr_test_pvalue >= 0.05:
        verdict = "SYMMETRIC FINDING ROBUST: Barriers remain symmetric under Wasserstein"
        robust = True
    elif ratio_diff > 0.5:
        verdict = "GEOMETRY-SENSITIVE: Results differ between kernel and Wasserstein"
        robust = False
    else:
        verdict = "SYMMETRIC FINDING ROBUST: Similar ratios under both geometries"
        robust = True

    print(f"Kernel asymmetry ratio:      {ratio_kern:.4f}")
    print(f"Wasserstein asymmetry ratio: {ratio_wass:.4f}")
    print(f"Ratio difference:            {ratio_diff:.4f}")
    print()
    print(f"VERDICT: {verdict}")
    print()

    # Build output
    output = {
        "version": "0.6.8.2",
        "timestamp": datetime.now().isoformat(),
        "task": "Path F: Asymmetric barriers test with Wasserstein",
        "hypothesis": "Verify symmetric barrier finding is geometry-robust",
        "sample": {
            "n_transitions": result_wass_asym.n_transitions,
            "n_alternatives": 10,
            "n_census_occupations": len(census_codes),
        },
        "distance_statistics": {
            "wasserstein": stats["d_wasserstein"],
            "kernel": stats["d_kernel"],
            "institutional_up": stats["d_up"],
            "institutional_down": stats["d_down"],
            "institutional_sym": stats["d_sym"],
        },
        "distance_correlations": {
            "wasserstein_based": corr_wass,
            "kernel_based": corr_kern,
        },
        "models": {
            "wasserstein_symmetric": {
                "alpha": result_wass_sym.alpha,
                "alpha_se": result_wass_sym.alpha_se,
                "alpha_t": result_wass_sym.alpha_t,
                "beta": result_wass_sym.beta,
                "beta_se": result_wass_sym.beta_se,
                "beta_t": result_wass_sym.beta_t,
                "log_likelihood": result_wass_sym.log_likelihood,
            },
            "wasserstein_asymmetric": result_wass_asym.to_dict(),
            "kernel_asymmetric": result_kern_asym.to_dict(),
        },
        "comparison": {
            "alpha_diff": result_wass_asym.alpha - result_kern_asym.alpha,
            "beta_up_diff": result_wass_asym.beta_up - result_kern_asym.beta_up,
            "beta_down_diff": result_wass_asym.beta_down - result_kern_asym.beta_down,
            "asymmetry_ratio_kernel": ratio_kern,
            "asymmetry_ratio_wasserstein": ratio_wass,
            "ratio_difference": ratio_diff,
            "log_likelihood_diff": result_wass_asym.log_likelihood - result_kern_asym.log_likelihood,
        },
        "interpretation": {
            "symmetric_robust": robust,
            "verdict": verdict,
        },
        "provenance": {
            "semantic_distance_source": "d_wasserstein_onet.npz (aggregated to census)",
            "institutional_source": "build_asymmetric_institutional_distance()",
            "transitions_source": "verified_transitions.parquet",
            "aggregation_method": "mean",
        },
    }

    # Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {OUTPUT_PATH}")
    print()

    return output


if __name__ == "__main__":
    output = main()
