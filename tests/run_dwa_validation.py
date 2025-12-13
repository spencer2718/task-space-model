"""
Run Phase I validation with DWA domain + Recipe Y geometry.

v0.4.2: Post-validation pivot from GWA + Recipe X (which failed).
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    print("=" * 70)
    print("v0.4.2 VALIDATION: DWA Domain + Recipe Y (Text Embeddings)")
    print("=" * 70)

    from task_space import (
        # Domain
        build_dwa_activity_domain,
        build_dwa_occupation_measures,
        # Distances
        compute_text_embedding_distances,
        distance_percentiles,
        # Kernel
        build_kernel_matrix,
        compute_overlap,
        # Validation
        SIGMA_PERCENTILES,
        compute_overlap_stats,
        build_validation_dataset,
        run_validation_regression,
        check_monotonicity,
        # Crosswalk
        build_onet_oes_crosswalk,
        load_oes_panel,
        compute_wage_comovement,
        aggregate_occupation_measures,
    )
    from task_space.diagnostics import diagnose_dwa_sparsity

    # =========================================================================
    # Step 1: Build DWA Domain and Occupation Measures
    # =========================================================================
    print("\n[1/6] Building DWA domain and occupation measures...")

    domain = build_dwa_activity_domain()
    measures = build_dwa_occupation_measures()

    print(f"  Domain: {domain.n_activities} DWAs")
    print(f"  Measures: {len(measures.occupation_codes)} occupations × {len(measures.activity_ids)} activities")

    # Sparsity diagnostic
    sparsity = diagnose_dwa_sparsity(measures)
    print(f"  Effective support (median): {sparsity.effective_support_percentiles['p50']:.1f}")
    print(f"  DWA coverage: {sparsity.dwa_coverage:.1%}")

    # =========================================================================
    # Step 2: Compute Recipe Y Text Embedding Distances
    # =========================================================================
    print("\n[2/6] Computing Recipe Y (text embedding) distances...")

    titles = list(domain.activity_names.values())
    distances = compute_text_embedding_distances(titles, domain.activity_ids)

    pcts = distance_percentiles(distances)
    print(f"  Distance percentiles:")
    print(f"    p10: {pcts['p10']:.4f}")
    print(f"    p50: {pcts['p50']:.4f}")
    print(f"    p90: {pcts['p90']:.4f}")

    # =========================================================================
    # Step 3: Load OES Wage Comovement Data
    # =========================================================================
    print("\n[3/6] Loading OES wage comovement data...")

    try:
        oes_panel = load_oes_panel([2019, 2020, 2021, 2022, 2023])
        comovement = compute_wage_comovement(oes_panel, min_years=4)
        print(f"  Comovement matrix: {comovement.n_occupations} SOC codes")
        print(f"  Years: {comovement.years}")
        print(f"  Coverage: {comovement.coverage:.1%}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Cannot run validation without OES data.")
        print("  Download OES data manually from https://www.bls.gov/oes/tables.htm")
        return

    # =========================================================================
    # Step 4: Build Crosswalk and Aggregate Measures
    # =========================================================================
    print("\n[4/6] Building crosswalk and aggregating measures...")

    crosswalk = build_onet_oes_crosswalk(
        measures.occupation_codes,
        comovement.occupation_codes,
    )
    print(f"  O*NET occupations: {crosswalk.n_onet}")
    print(f"  Matched SOC codes: {crosswalk.n_matched}")
    print(f"  Coverage: {crosswalk.coverage:.1%}")

    # Aggregate O*NET measures to SOC level
    agg_matrix, agg_codes = aggregate_occupation_measures(
        measures.occupation_matrix,
        measures.occupation_codes,
        crosswalk,
    )
    print(f"  Aggregated measures: {len(agg_codes)} SOC codes × {agg_matrix.shape[1]} activities")

    # =========================================================================
    # Step 5: Compute Overlap Grid (5 sigma values)
    # =========================================================================
    print("\n[5/6] Computing overlap grid for 5 sigma values...")

    results = {}
    overlap_grids = {}

    for sigma_pct in SIGMA_PERCENTILES:
        sigma = pcts[sigma_pct]
        print(f"\n  {sigma_pct}: σ = {sigma:.4f}")

        # Build kernel
        kernel = build_kernel_matrix(distances, sigma=sigma)

        # Compute overlap at aggregated (SOC) level
        from task_space.domain import OccupationMeasures

        agg_measures = OccupationMeasures(
            occupation_codes=agg_codes,
            occupation_matrix=agg_matrix,
            activity_ids=measures.activity_ids,
            raw_matrix=agg_matrix,  # Already normalized
        )
        overlap = compute_overlap(agg_measures, kernel)

        # Find common occupations between overlap and comovement
        common_codes = sorted(set(agg_codes) & set(comovement.occupation_codes))
        print(f"    Common occupations: {len(common_codes)}")

        if len(common_codes) < 100:
            print(f"    WARNING: Too few common occupations, skipping")
            continue

        # Extract submatrices for common codes
        agg_idx = [agg_codes.index(c) for c in common_codes]
        com_idx = [comovement.occupation_codes.index(c) for c in common_codes]

        overlap_common = overlap[np.ix_(agg_idx, agg_idx)]
        comovement_common = comovement.comovement_matrix[np.ix_(com_idx, com_idx)]

        # Build pair dataset
        n = len(common_codes)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append({
                    "occ_i": common_codes[i],
                    "occ_j": common_codes[j],
                    "overlap": overlap_common[i, j],
                    "y": comovement_common[i, j],
                })

        import pandas as pd
        pair_df = pd.DataFrame(pairs)
        pair_df = pair_df.dropna()

        print(f"    Pairs: {len(pair_df)}")
        print(f"    Overlap: mean={pair_df['overlap'].mean():.6f}, std={pair_df['overlap'].std():.6f}")
        print(f"    Comovement: mean={pair_df['y'].mean():.4f}, std={pair_df['y'].std():.4f}")

        # Run regression
        from task_space.validation import _cluster_se

        X = pair_df[["overlap"]].values
        y = pair_df["y"].values
        clusters = pair_df["occ_i"].values

        beta, se, r2, n_clusters = _cluster_se(X, y, clusters)

        # Compute t-stat and p-value
        from scipy.stats import t as t_dist
        t_stat = beta / se
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), n_clusters - 1))

        passes = bool(beta > 0 and p_value < 0.10)

        results[sigma_pct] = {
            "sigma": sigma,
            "beta": float(beta),
            "se": float(se),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "r_squared": float(r2),
            "n_pairs": len(pair_df),
            "n_clusters": n_clusters,
            "passes": passes,
        }

        print(f"    β = {beta:.4f} (SE = {se:.4f})")
        print(f"    t = {t_stat:.2f}, p = {p_value:.4f}")
        print(f"    R² = {r2:.6f}")
        print(f"    RESULT: {'PASS' if passes else 'FAIL'}")

        overlap_grids[sigma_pct] = {
            "overlap_mean": float(pair_df['overlap'].mean()),
            "overlap_std": float(pair_df['overlap'].std()),
        }

    # =========================================================================
    # Step 6: Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY: DWA + Recipe Y")
    print("=" * 70)

    print("\n| σ Percentile | β | SE | p-value | R² | Result |")
    print("|--------------|------|------|---------|-----|--------|")

    n_passing = 0
    for pct in SIGMA_PERCENTILES:
        if pct in results:
            r = results[pct]
            status = "PASS" if r["passes"] else "FAIL"
            if r["passes"]:
                n_passing += 1
            print(f"| {pct} | {r['beta']:.4f} | {r['se']:.4f} | {r['p_value']:.4f} | {r['r_squared']:.6f} | {status} |")
        else:
            print(f"| {pct} | - | - | - | - | SKIP |")

    print(f"\nPassing: {n_passing}/{len(SIGMA_PERCENTILES)}")

    if n_passing == len(SIGMA_PERCENTILES):
        decision = "PASS"
    elif n_passing == 0:
        decision = "FAIL"
    else:
        decision = "PARTIAL"

    print(f"Overall Decision: {decision}")

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs" / "phase_i_dwa"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "validation_results.json", "w") as f:
        json.dump({
            "domain": "DWA",
            "recipe": "Y (text embeddings)",
            "model": "all-mpnet-base-v2",
            "n_activities": domain.n_activities,
            "n_occupations": len(measures.occupation_codes),
            "results": results,
            "n_passing": n_passing,
            "decision": decision,
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'validation_results.json'}")


if __name__ == "__main__":
    main()
