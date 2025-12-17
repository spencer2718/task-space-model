#!/usr/bin/env python3
"""
Phase A: Raw Binary Overlap Baseline

Tests whether counting shared activities predicts wage comovement,
establishing the floor for SAE comparison.

Usage:
    PYTHONPATH=src python tests/run_phase_a.py
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from task_space import (
    build_dwa_occupation_measures,
    compute_binary_overlap,
    run_baseline_regression,
    save_baseline_results,
    build_onet_oes_crosswalk,
    load_oes_panel,
    compute_wage_comovement,
    onet_to_soc,
)


def main():
    print("=" * 60)
    print("Phase A: Raw Binary Overlap Baseline")
    print("=" * 60)

    # Output directory
    output_dir = Path("outputs/phase_a")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load DWA occupation measures
    print("\n[1/5] Loading DWA occupation measures...")
    measures = build_dwa_occupation_measures()
    print(f"      Loaded {len(measures.occupation_codes)} occupations × {len(measures.activity_ids)} activities")
    print(f"      Raw matrix shape: {measures.raw_matrix.shape}")

    # Step 2: Compute binary overlap
    print("\n[2/5] Computing binary Jaccard overlap...")
    overlap_result = compute_binary_overlap(measures, threshold=0.0)
    print(f"      Overlap matrix shape: {overlap_result.overlap_matrix.shape}")
    print(f"      Sparsity stats:")
    print(f"        - Mean activities per occupation: {overlap_result.sparsity_stats['mean_activities_per_occupation']:.1f}")
    print(f"        - Matrix sparsity: {overlap_result.sparsity_stats['sparsity']:.1%}")
    print(f"      Overlap stats:")
    print(f"        - Mean off-diagonal: {overlap_result.stats['off_diag_mean']:.4f}")
    print(f"        - Median off-diagonal: {overlap_result.stats['off_diag_p50']:.4f}")

    # Step 3: Load wage comovement data
    print("\n[3/5] Loading wage comovement data...")
    oes_path = Path("data/external/oes")
    if not oes_path.exists():
        print(f"      ERROR: OES data not found at {oes_path}")
        print("      Please download OES data manually from https://www.bls.gov/oes/tables.htm")
        return 1

    panel = load_oes_panel(years=[2019, 2020, 2021, 2022, 2023], data_dir=oes_path)
    print(f"      Loaded {len(panel)} occupation-year observations")

    # min_years=4 because we have 5 years of data = 4 year-over-year changes
    comovement = compute_wage_comovement(panel, min_years=4)
    print(f"      Comovement matrix: {comovement.comovement_matrix.shape[0]} occupations")
    n_pairs = comovement.n_occupations * (comovement.n_occupations - 1) // 2
    print(f"      Total pairs: {n_pairs}, coverage: {comovement.coverage:.1%}")

    # Step 4: Build crosswalk and run regression
    print("\n[4/5] Running validation regression...")

    # Build crosswalk map
    crosswalk_map = {code: onet_to_soc(code) for code in measures.occupation_codes}

    regression_result = run_baseline_regression(
        overlap_result=overlap_result,
        comovement_matrix=comovement.comovement_matrix,
        comovement_codes=comovement.occupation_codes,
        crosswalk_map=crosswalk_map,
    )

    # Step 5: Save results
    print("\n[5/5] Saving results...")
    save_baseline_results(overlap_result, regression_result, output_dir)
    print(f"      Results saved to {output_dir}")

    # Decision gate
    print("\n" + "=" * 60)
    print("PHASE A DECISION GATE")
    print("=" * 60)
    if regression_result.passes:
        print("✓ PASS: Binary overlap predicts wage comovement")
        print("  → Baseline established. SAE must beat R² = {:.6f}".format(regression_result.r_squared))
        print("  → Proceed to Phase B (SAE Training)")
    else:
        print("✗ FAIL: Binary overlap does not predict wage comovement")
        print("  → Investigate data quality before proceeding")
        print("  → Check: Are occupations with shared activities earning similar wages?")

    return 0 if regression_result.passes else 1


if __name__ == "__main__":
    sys.exit(main())
