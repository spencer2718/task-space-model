#!/usr/bin/env python3
"""
Experiment v0.7.0.2: Switching Cost Sensitivity Analysis

Computes switching costs across the full literature calibration range
to verify qualitative findings hold.

Calibration points (wage-years):
- Lee & Wolpin (2006): 0.75
- Dix-Carneiro mid (adopted): 2.0
- Dix-Carneiro upper: 2.7
- Artuc et al. (2010): 6.5

Usage:
    python scripts/run_sensitivity_v0702.py
"""

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Canonical imports from task_space
from task_space.mobility.io import (
    load_transitions,
    get_training_transitions,
    load_wasserstein_census,
    load_institutional_census,
)
from task_space.utils.experiments import save_experiment_output
from task_space.validation.scaled_costs import (
    compute_externally_calibrated_costs,
    compute_example_transition_costs,
    compute_median_distances,
)


# =============================================================================
# Calibration Points
# =============================================================================

CALIBRATION_SOURCES = [
    {"source": "Lee & Wolpin (2006)", "anchor_wage_years": 0.75},
    {"source": "Dix-Carneiro mid", "anchor_wage_years": 2.0},
    {"source": "Dix-Carneiro upper", "anchor_wage_years": 2.7},
    {"source": "Artuc et al. (2010)", "anchor_wage_years": 6.5},
]

# Example occupation pairs for sensitivity check
# Census codes from OES/crosswalk lookup
EXAMPLE_PAIRS = [
    {"from": "Cashiers", "to": "Retail Salespersons", "census_from": 4720, "census_to": 4760},
    {"from": "Software Developers", "to": "Data Scientists", "census_from": 1020, "census_to": 1200},
    {"from": "Accountants", "to": "Software Developers", "census_from": 800, "census_to": 1020},
    {"from": "Cashiers", "to": "Registered Nurses", "census_from": 4720, "census_to": 3255},
]


# =============================================================================
# Core Analysis
# =============================================================================

def lookup_example_distances(
    examples: List[Dict],
    d_sem: np.ndarray,
    census_codes: List[int],
) -> List[Dict]:
    """Look up Wasserstein distances for example pairs."""
    code_to_idx = {c: i for i, c in enumerate(census_codes)}
    result = []

    for ex in examples:
        if ex["census_from"] in code_to_idx and ex["census_to"] in code_to_idx:
            i = code_to_idx[ex["census_from"]]
            j = code_to_idx[ex["census_to"]]
            d_wass = float(d_sem[i, j])
            result.append({
                "from": ex["from"],
                "to": ex["to"],
                "d_wass": round(d_wass, 4),
            })
        else:
            print(f"    WARNING: Could not find codes for {ex['from']} -> {ex['to']}")

    return result


def run_sensitivity_analysis(
    d_wasserstein_median: float,
    mean_annual_wage: float,
    examples_with_distances: List[Dict],
) -> List[Dict]:
    """
    Run sensitivity analysis across all calibration points.

    Returns list of calibration results.
    """
    results = []

    for calib in CALIBRATION_SOURCES:
        # Compute calibrated costs
        calibration = compute_externally_calibrated_costs(
            d_wasserstein_median=d_wasserstein_median,
            benchmark_cost_wage_years=calib["anchor_wage_years"],
            mean_annual_wage=mean_annual_wage,
        )

        sc_per_unit = calibration["sc_per_unit_wasserstein_wage_years"]
        median_transition_cost = sc_per_unit * d_wasserstein_median

        # Compute example transition costs
        examples_with_costs = compute_example_transition_costs(
            examples_with_distances,
            sc_per_unit,
            mean_annual_wage,
        )

        results.append({
            "source": calib["source"],
            "anchor_wage_years": calib["anchor_wage_years"],
            "sc_per_unit_wasserstein": round(sc_per_unit, 3),
            "median_transition_cost": round(median_transition_cost, 3),
            "example_transitions": examples_with_costs,
        })

    return results


def check_orderings_preserved(calibration_results: List[Dict]) -> Tuple[bool, str]:
    """
    Check if qualitative orderings are preserved across calibration points.

    For any two calibration points, the ordering of transition costs
    should be identical (since d_wass is constant, only scale changes).
    """
    # Extract example transition costs for each calibration
    orderings = []

    for calib in calibration_results:
        # Get list of (from, to, cost) tuples
        costs = [(ex["from"], ex["to"], ex["cost_wage_years"])
                 for ex in calib["example_transitions"]]
        # Sort by cost
        sorted_costs = sorted(costs, key=lambda x: x[2])
        # Extract ordering (just the pairs)
        ordering = [(c[0], c[1]) for c in sorted_costs]
        orderings.append(ordering)

    # Check if all orderings are identical
    first_ordering = orderings[0]
    all_same = all(o == first_ordering for o in orderings)

    if all_same:
        notes = "All orderings identical across calibration points (scale-invariant as expected)"
    else:
        # Find where they differ
        notes = "UNEXPECTED: Orderings differ across calibration points"
        for i, o in enumerate(orderings):
            if o != first_ordering:
                notes += f"\n  - {calibration_results[i]['source']} differs"

    return all_same, notes


def validate_examples(examples_with_costs: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Validate that example transition costs pass smell tests.

    Returns (passed, list of issues).
    """
    issues = []

    for ex in examples_with_costs:
        cost = ex["cost_wage_years"]

        # Cashier -> Retail should be relatively low (similar jobs)
        if ex["from"] == "Cashiers" and ex["to"] == "Retail Salespersons":
            if cost < 0:
                issues.append(f"Cashiers -> Retail has negative cost: {cost}")

        # All costs should be positive
        if cost < 0:
            issues.append(f"{ex['from']} -> {ex['to']} has negative cost: {cost}")

        # Costs should be finite
        if not np.isfinite(cost):
            issues.append(f"{ex['from']} -> {ex['to']} has non-finite cost: {cost}")

    return len(issues) == 0, issues


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Experiment v0.7.0.2: Switching Cost Sensitivity Analysis")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n[1] Loading data...")

    # Load distance matrices
    d_sem, census_codes = load_wasserstein_census()
    print(f"    Wasserstein matrix: {d_sem.shape}")

    # Load transitions and compute median distance
    transitions = load_transitions()
    train_df = get_training_transitions(transitions)
    print(f"    Training transitions: {len(train_df):,}")

    # Load institutional distances for median computation
    d_inst, _ = load_institutional_census()

    # Compute median distances
    d_sem_median, d_inst_median = compute_median_distances(
        train_df, d_sem, d_inst, census_codes
    )
    print(f"    d_wasserstein_median: {d_sem_median:.4f}")

    # Use OES 2023 mean wage (from v0.7b)
    mean_annual_wage = 71992.0
    print(f"    Mean annual wage: ${mean_annual_wage:,.0f}")

    # =========================================================================
    # Step 2: Look up example distances
    # =========================================================================
    print("\n[2] Looking up example transition distances...")

    examples_with_distances = lookup_example_distances(
        EXAMPLE_PAIRS, d_sem, census_codes
    )

    for ex in examples_with_distances:
        print(f"    {ex['from']:25} -> {ex['to']:20}: d_wass = {ex['d_wass']:.4f}")

    if len(examples_with_distances) != len(EXAMPLE_PAIRS):
        print(f"\n    WARNING: Only {len(examples_with_distances)}/{len(EXAMPLE_PAIRS)} examples found")
        if len(examples_with_distances) == 0:
            print("    STOP: No example distances available")
            return None

    # =========================================================================
    # Step 3: Run sensitivity analysis
    # =========================================================================
    print("\n[3] Running sensitivity analysis...")

    calibration_results = run_sensitivity_analysis(
        d_wasserstein_median=d_sem_median,
        mean_annual_wage=mean_annual_wage,
        examples_with_distances=examples_with_distances,
    )

    # Print results table
    print("\n    Calibration Results:")
    print("    " + "-" * 70)
    print(f"    {'Source':<25} {'Anchor':>10} {'SC/unit':>12} {'Median Cost':>12}")
    print("    " + "-" * 70)

    for r in calibration_results:
        print(f"    {r['source']:<25} {r['anchor_wage_years']:>10.2f} "
              f"{r['sc_per_unit_wasserstein']:>12.3f} {r['median_transition_cost']:>12.3f}")

    # =========================================================================
    # Step 4: Check orderings
    # =========================================================================
    print("\n[4] Checking qualitative orderings...")

    orderings_preserved, ordering_notes = check_orderings_preserved(calibration_results)
    print(f"    Orderings preserved: {orderings_preserved}")
    print(f"    Notes: {ordering_notes}")

    # =========================================================================
    # Step 5: Validate examples
    # =========================================================================
    print("\n[5] Validating example costs...")

    all_valid = True
    for r in calibration_results:
        valid, issues = validate_examples(r["example_transitions"])
        if not valid:
            all_valid = False
            print(f"    {r['source']}: ISSUES")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print(f"    {r['source']}: OK")

    # =========================================================================
    # Step 6: Assemble output
    # =========================================================================
    print("\n[6] Assembling output...")

    output = {
        "experiment": "sensitivity_switching_costs_v0702",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "d_wasserstein_median": round(d_sem_median, 4),
        "mean_annual_wage": mean_annual_wage,
        "calibration_results": calibration_results,
        "qualitative_check": {
            "orderings_preserved": orderings_preserved,
            "all_examples_valid": all_valid,
            "notes": ordering_notes,
        },
    }

    # =========================================================================
    # Step 7: Save results
    # =========================================================================
    output_path = save_experiment_output("sensitivity_switching_costs_v0702", output)
    print(f"    Saved to: {output_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nMedian Wasserstein distance: {d_sem_median:.4f}")
    print(f"\nCalibration Range:")

    for r in calibration_results:
        print(f"\n  {r['source']}:")
        print(f"    Anchor: {r['anchor_wage_years']} wage-years")
        print(f"    SC per unit Wasserstein: {r['sc_per_unit_wasserstein']:.3f} wage-years")
        print(f"    Median transition cost: {r['median_transition_cost']:.3f} wage-years")
        print(f"    Example transitions:")
        for ex in r["example_transitions"]:
            print(f"      {ex['from']:25} -> {ex['to']:20}: {ex['cost_wage_years']:.2f} yrs")

    print(f"\nQualitative Check:")
    print(f"  Orderings preserved: {orderings_preserved}")
    print(f"  All examples valid: {all_valid}")

    print("\n" + "=" * 60)

    return output


if __name__ == "__main__":
    main()
