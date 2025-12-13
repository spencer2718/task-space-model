#!/usr/bin/env python3
"""
Phase 1 Fix Runner: Test kernel-weighted overlap with NN-based sigma.

This script:
1. Computes nearest-neighbor distance statistics
2. Tests sigma discrimination with NN-based candidates
3. Runs validation regressions with multiple sigma values
4. Compares to Jaccard baseline (t=8.00, β=0.471, R²=0.00167)

Usage:
    PYTHONPATH=src python tests/run_phase1_fix.py
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from task_space.domain import build_dwa_activity_domain, build_dwa_occupation_measures
from task_space.crosswalk import build_onet_oes_crosswalk, compute_wage_comovement
from task_space.diagnostics_v061 import (
    diagnose_nearest_neighbor_distances,
    test_sigma_discrimination,
    compute_kernel_overlap,
    run_kernel_validation,
)


def main():
    output_dir = Path("outputs/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1 Fix: NN-Based Sigma Selection")
    print("=" * 60)

    # Jaccard baseline from v0.5.0
    JACCARD_BASELINE = {
        "beta": 0.471,
        "t": 8.00,
        "r2": 0.00167,
    }

    # Step 1: Load data
    print("\n[1/6] Loading data...")
    domain = build_dwa_activity_domain()
    measures = build_dwa_occupation_measures()
    print(f"  - {domain.n_activities} activities")
    print(f"  - {len(measures.occupation_codes)} occupations")

    # Load distance matrix from Phase 1 embeddings
    embeddings = np.load(output_dir / "activity_embeddings.npy")
    print(f"  - Loaded embeddings: {embeddings.shape}")

    # Compute cosine distance matrix
    from sklearn.metrics.pairwise import cosine_distances
    dist_matrix = cosine_distances(embeddings)
    print(f"  - Distance matrix: {dist_matrix.shape}")

    # Step 2: Load wage comovement data
    print("\n[2/6] Loading wage comovement data...")
    from task_space.crosswalk import load_oes_panel, onet_to_soc

    # Load OES panel (years 2019-2023)
    oes_panel = load_oes_panel(years=[2019, 2020, 2021, 2022, 2023])
    oes_codes = oes_panel["OCC_CODE"].unique().tolist()
    print(f"  - Loaded OES data for {len(oes_codes)} SOC codes")

    # Build crosswalk
    crosswalk = build_onet_oes_crosswalk(
        onet_codes=measures.occupation_codes,
        oes_codes=oes_codes,
    )
    print(f"  - Crosswalk coverage: {crosswalk.coverage:.1%}")

    # Compute wage comovement
    comovement = compute_wage_comovement(oes_panel, min_years=4)
    print(f"  - {len(comovement.occupation_codes)} SOC occupations with sufficient data")
    print(f"  - Years: {comovement.years}")

    # Build crosswalk map (O*NET code -> SOC code)
    crosswalk_map = {}
    for onet_code in measures.occupation_codes:
        soc = onet_to_soc(onet_code)
        if soc in comovement.occupation_codes:
            crosswalk_map[onet_code] = soc

    # Step 3: Compute nearest-neighbor distance statistics
    print("\n[3/6] Computing nearest-neighbor distances...")
    nn_stats = diagnose_nearest_neighbor_distances(dist_matrix)
    print(f"  - NN min: {nn_stats.min:.4f}")
    print(f"  - NN median: {nn_stats.median:.4f}")
    print(f"  - NN max: {nn_stats.max:.4f}")
    print(f"  - NN p10: {nn_stats.p10:.4f}")
    print(f"  - NN p25: {nn_stats.p25:.4f}")
    print(f"  - Sigma candidates:")
    for name, val in nn_stats.sigma_candidates.items():
        print(f"    {name}: {val:.4f}")

    # Step 4: Test sigma discrimination
    print("\n[4/6] Testing sigma discrimination...")
    sigma_values = list(nn_stats.sigma_candidates.values())

    # Also test the old sigma (median of all distances) for comparison
    overall_median = float(np.median(dist_matrix[dist_matrix > 0]))
    sigma_values.append(overall_median)

    discrimination_results = test_sigma_discrimination(dist_matrix, sigma_values)

    print(f"\n  {'Sigma':>10} | {'Ratio':>8} | {'Range':>10} | Status")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*10}-+--------")
    for key, result in discrimination_results.items():
        status = "COLLAPSED" if result.collapsed else "OK"
        print(f"  {result.sigma:>10.4f} | {result.discrimination_ratio:>8.1f}x | {result.normalized_weight_range:>10.6f} | {status}")

    # Step 5: Run validation regressions
    print("\n[5/6] Running validation regressions...")

    validation_results = {}

    # Test grid from spec
    test_configs = [
        ("sigma_nn_p10_norm", nn_stats.sigma_candidates['nn_p10'], True),
        ("sigma_nn_p25_norm", nn_stats.sigma_candidates['nn_p25'], True),
        ("sigma_nn_med_norm", nn_stats.sigma_candidates['nn_median'], True),
        ("sigma_nn_p10_unnorm", nn_stats.sigma_candidates['nn_p10'], False),
        ("sigma_nn_med_unnorm", nn_stats.sigma_candidates['nn_median'], False),
        ("sigma_old_norm", overall_median, True),  # Old approach for comparison
    ]

    print(f"\n  {'Label':<22} | {'Sigma':>8} | {'Norm':>5} | {'Beta':>8} | {'t_ols':>8} | {'R²':>10}")
    print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

    def compute_ols_tstat(overlap_matrix, comovement_matrix, overlap_codes, comovement_codes, crosswalk_map):
        """Compute simple OLS t-stat (not clustered) for comparison."""
        from scipy import stats as sp_stats

        # Build pairs (same aggregation as in run_kernel_validation)
        onet_to_soc = {k: v for k, v in crosswalk_map.items()}
        soc_codes = list(set(onet_to_soc.values()))
        soc_codes = [soc for soc in soc_codes if soc in comovement_codes]
        soc_codes = sorted(soc_codes)

        n_soc = len(soc_codes)
        soc_to_idx = {soc: i for i, soc in enumerate(soc_codes)}
        comovement_idx = {soc: i for i, soc in enumerate(comovement_codes)}
        onet_to_overlap_idx = {code: i for i, code in enumerate(overlap_codes)}

        soc_overlap = np.zeros((n_soc, n_soc))
        soc_counts = np.zeros((n_soc, n_soc))

        for onet_i, soc_i in onet_to_soc.items():
            if soc_i not in soc_to_idx:
                continue
            for onet_j, soc_j in onet_to_soc.items():
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

        # Build pair arrays
        pairs_x, pairs_y = [], []
        for i in range(n_soc):
            for j in range(i + 1, n_soc):
                ci = comovement_idx[soc_codes[i]]
                cj = comovement_idx[soc_codes[j]]
                comove_val = comovement_matrix[ci, cj]
                if np.isnan(comove_val):
                    continue
                pairs_x.append(soc_overlap[i, j])
                pairs_y.append(comove_val)

        x = np.array(pairs_x)
        y = np.array(pairs_y)

        # Simple Pearson correlation -> t-stat
        r, p = sp_stats.pearsonr(x, y)
        n = len(x)
        t_stat = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else float('inf')

        return t_stat, r

    for label, sigma, normalized in test_configs:
        # Compute kernel overlap
        overlap = compute_kernel_overlap(
            occ_measures=measures.occupation_matrix,
            dist_matrix=dist_matrix,
            sigma=sigma,
            normalize_kernel=normalized,
        )

        # Run validation
        result = run_kernel_validation(
            overlap_matrix=overlap,
            comovement_matrix=comovement.comovement_matrix,
            overlap_codes=measures.occupation_codes,
            comovement_codes=comovement.occupation_codes,
            crosswalk_map=crosswalk_map,
            sigma=sigma,
            normalized=normalized,
            label=label,
        )

        # Compute simple OLS t-stat for comparison
        t_ols, r_pearson = compute_ols_tstat(
            overlap, comovement.comovement_matrix,
            measures.occupation_codes, comovement.occupation_codes,
            crosswalk_map
        )

        validation_results[label] = result
        # Store OLS t-stat in results for output
        validation_results[label] = result
        validation_results[label + "_t_ols"] = t_ols

        norm_str = "Yes" if normalized else "No"
        print(f"  {label:<22} | {sigma:>8.4f} | {norm_str:>5} | {result.beta:>8.4f} | {t_ols:>8.2f} | {result.r_squared:>10.6f}")

    # Step 6: Compare to Jaccard baseline
    print("\n[6/6] Comparing to Jaccard baseline...")
    print(f"\n  Jaccard Baseline: β={JACCARD_BASELINE['beta']:.3f}, t={JACCARD_BASELINE['t']:.2f}, R²={JACCARD_BASELINE['r2']:.5f}")

    # Find best kernel result by OLS t-stat
    kernel_labels = [k for k in validation_results.keys() if not k.endswith("_t_ols")]
    best_label = max(kernel_labels, key=lambda k: validation_results.get(k + "_t_ols", 0))
    best = validation_results[best_label]
    best_t_ols = validation_results.get(best_label + "_t_ols", 0)

    print(f"  Best Kernel:      β={best.beta:.3f}, t_ols={best_t_ols:.2f}, R²={best.r_squared:.5f} ({best_label})")

    # Determine if kernel improves over Jaccard
    improves = best_t_ols > JACCARD_BASELINE['t'] and best.r_squared > JACCARD_BASELINE['r2']
    print(f"\n  Kernel improves over Jaccard: {improves}")

    # Generate output
    output = {
        "nn_distance_stats": asdict(nn_stats),
        "sigma_discrimination_tests": {
            k: asdict(v) for k, v in discrimination_results.items()
        },
        "validation_results": {
            k: (asdict(v) if hasattr(v, '__dataclass_fields__') else v)
            for k, v in validation_results.items()
        },
        "comparison_to_jaccard": {
            "jaccard_beta": JACCARD_BASELINE['beta'],
            "jaccard_t": JACCARD_BASELINE['t'],
            "jaccard_r2": JACCARD_BASELINE['r2'],
            "best_kernel_label": best_label,
            "best_kernel_beta": best.beta,
            "best_kernel_t_ols": float(best_t_ols),
            "best_kernel_r2": best.r_squared,
            "kernel_improves_over_jaccard": improves,
        },
    }

    # Save results
    with open(output_dir / "phase1_fix_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_dir}/phase1_fix_results.json")

    # Print decision
    print("\n" + "=" * 60)
    print("DECISION")
    print("=" * 60)

    if best_t_ols > JACCARD_BASELINE['t']:
        print(f"\n✓ Kernel (t_ols={best_t_ols:.2f}) BEATS Jaccard (t={JACCARD_BASELINE['t']:.2f})")
        print(f"  Use kernel-weighted overlap for Phase II with sigma={best.sigma:.4f}")
        print(f"  R² improvement: {best.r_squared:.5f} vs {JACCARD_BASELINE['r2']:.5f} (+{100*(best.r_squared/JACCARD_BASELINE['r2']-1):.1f}%)")
    elif best_t_ols > 2.0:
        print(f"\n~ Kernel works (t_ols={best_t_ols:.2f}) but doesn't dominate Jaccard (t={JACCARD_BASELINE['t']:.2f})")
        print(f"  Report as alternative measure")
    else:
        print(f"\n✗ Kernel still fails (t_ols={best_t_ols:.2f}) after fix")
        print(f"  Proceed to Phase 2 (alternative distances)")


if __name__ == "__main__":
    main()
