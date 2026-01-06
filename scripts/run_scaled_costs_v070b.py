#!/usr/bin/env python3
"""
Phase 0.7b: Scaled Cost Estimation

Adds destination wages to conditional logit to enable switching cost
estimation in wage-equivalent units.

Steps:
1. Load CPS transitions (training set from 0.7a: 2015-2019, 2022-2023)
2. Load OES wages, map to Census codes
3. Build choice dataset with wages
4. Estimate three model variants (M1, M2, M3)
5. Compute switching costs from M1 (primary)
6. Compare to Dix-Carneiro benchmark (1.4–2.7× annual wages)
7. Write to outputs/experiments/scaled_costs_v070b.json

Usage:
    python scripts/run_scaled_costs_v070b.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Canonical imports from task_space
from task_space.mobility.io import (
    load_transitions,
    get_training_transitions,
    load_wasserstein_census,
    load_institutional_census,
)
from task_space.validation.scaled_costs import (
    load_oes_wages_by_census,
    get_wage_coverage,
    build_choice_dataset_with_wages,
    estimate_scaled_model,
    compute_switching_costs,
    compute_median_distances,
    compute_verdict,
    compute_externally_calibrated_costs,
    compute_example_transition_costs,
    ScaledCostsResult,
)


def main():
    print("=" * 60)
    print("Phase 0.7b: Scaled Cost Estimation")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n[1] Loading data...")

    # Load transitions using canonical functions
    transitions = load_transitions()
    train_df = get_training_transitions(transitions)
    print(f"    Training transitions: {len(train_df):,}")

    # Load distance matrices using canonical functions
    d_sem, census_codes = load_wasserstein_census()
    print(f"    Census occupations: {len(census_codes)}")
    print("    Loading institutional distances...")
    d_inst, _ = load_institutional_census()

    # =========================================================================
    # Step 2: Load OES wages
    # =========================================================================
    print("\n[2] Loading OES wages (2023)...")

    oes_year = 2023
    wages_df = load_oes_wages_by_census(year=oes_year)
    print(f"    OES occupations with wages: {len(wages_df)}")

    coverage = get_wage_coverage(wages_df, census_codes)
    print(f"    Coverage of Census codes: {coverage:.1%}")

    mean_annual_wage = wages_df["mean_annual_wage"].mean()
    print(f"    Mean annual wage: ${mean_annual_wage:,.0f}")

    if coverage < 0.80:
        print("    WARNING: Coverage < 80%, proceed with caution")

    # =========================================================================
    # Step 3: Build choice dataset with wages
    # =========================================================================
    print("\n[3] Building choice dataset with wages...")

    choice_df = build_choice_dataset_with_wages(
        train_df,
        d_sem,
        d_inst,
        wages_df,
        census_codes,
        n_alternatives=10,
        random_seed=42,
    )
    print(f"    Choice dataset: {len(choice_df):,} rows")
    print(f"    Unique cases: {choice_df['case_id'].nunique():,}")

    # Compute median distances from observed transitions
    d_sem_median, d_inst_median = compute_median_distances(
        train_df, d_sem, d_inst, census_codes
    )
    print(f"    Median Wasserstein distance: {d_sem_median:.4f}")
    print(f"    Median institutional distance: {d_inst_median:.4f}")

    # =========================================================================
    # Step 4: Estimate model variants
    # =========================================================================
    print("\n[4] Estimating model variants...")

    # M1: Log destination wage (primary)
    print("\n    Model M1: log(wage_dest)...")
    result_m1 = estimate_scaled_model(choice_df, model_variant="M1")
    print(f"      gamma_sem: {result_m1.gamma_sem:.4f} (t={result_m1.gamma_sem_t:.1f})")
    print(f"      gamma_inst: {result_m1.gamma_inst:.4f} (t={result_m1.gamma_inst_t:.1f})")
    print(f"      beta_wage: {result_m1.beta_wage:.4f} (t={result_m1.beta_wage_t:.1f})")
    print(f"      Log-likelihood: {result_m1.log_likelihood:,.0f}")

    # M2: Level destination wage (in $10k)
    print("\n    Model M2: wage_dest (in $10k)...")
    result_m2 = estimate_scaled_model(choice_df, model_variant="M2")
    print(f"      gamma_sem: {result_m2.gamma_sem:.4f} (t={result_m2.gamma_sem_t:.1f})")
    print(f"      gamma_inst: {result_m2.gamma_inst:.4f} (t={result_m2.gamma_inst_t:.1f})")
    print(f"      beta_wage: {result_m2.beta_wage:.4f} (t={result_m2.beta_wage_t:.1f})")
    print(f"      Log-likelihood: {result_m2.log_likelihood:,.0f}")

    # M3: Log wage ratio
    print("\n    Model M3: log(wage_dest / wage_origin)...")
    result_m3 = estimate_scaled_model(choice_df, model_variant="M3")
    print(f"      gamma_sem: {result_m3.gamma_sem:.4f} (t={result_m3.gamma_sem_t:.1f})")
    print(f"      gamma_inst: {result_m3.gamma_inst:.4f} (t={result_m3.gamma_inst_t:.1f})")
    print(f"      beta_wage: {result_m3.beta_wage:.4f} (t={result_m3.beta_wage_t:.1f})")
    print(f"      Log-likelihood: {result_m3.log_likelihood:,.0f}")

    # =========================================================================
    # Step 5: Compute switching costs from M1
    # =========================================================================
    print("\n[5] Computing switching costs from M1...")

    switching_costs = compute_switching_costs(
        result_m1,
        d_wasserstein_median=d_sem_median,
        d_inst_median=d_inst_median,
        mean_annual_wage=mean_annual_wage,
    )
    print(f"    SC per unit Wasserstein: {switching_costs['sc_sem_per_unit']:.4f} log-wage")
    print(f"    SC per unit institutional: {switching_costs['sc_inst_per_unit']:.4f} log-wage")
    print(f"    Typical switching cost: {switching_costs['sc_typical_wage_years']:.2f} wage-years")
    print(f"    Typical switching cost: ${switching_costs['sc_typical_dollars']:,.0f}")

    # =========================================================================
    # Step 6: Compare to Dix-Carneiro benchmark
    # =========================================================================
    print("\n[6] Benchmark comparison...")

    dix_carneiro_range = [1.4, 2.7]
    within_range = dix_carneiro_range[0] <= switching_costs["sc_typical_wage_years"] <= dix_carneiro_range[1]
    print(f"    Dix-Carneiro (2014) range: {dix_carneiro_range[0]}-{dix_carneiro_range[1]} wage-years")
    print(f"    Our estimate: {switching_costs['sc_typical_wage_years']:.2f} wage-years")
    print(f"    Within benchmark range: {within_range}")

    # =========================================================================
    # Step 7: External Calibration (0.7b-revised)
    # =========================================================================
    print("\n[7] External calibration (Dix-Carneiro benchmark)...")

    # Since wage identification fails, calibrate externally
    external_calibration = compute_externally_calibrated_costs(
        d_wasserstein_median=d_sem_median,
        benchmark_cost_wage_years=2.0,  # Dix-Carneiro midpoint
        mean_annual_wage=mean_annual_wage,
    )

    sc_per_unit = external_calibration["sc_per_unit_wasserstein_wage_years"]
    print(f"    Benchmark: 2.0 wage-years for typical transition")
    print(f"    Calibrated SC per unit Wasserstein: {sc_per_unit:.2f} wage-years")
    print(f"    Calibrated SC per unit Wasserstein: ${external_calibration['sc_per_unit_wasserstein_dollars']:,.0f}")

    # Compute example transition costs with ACTUAL Wasserstein distances
    # Census codes from OES/crosswalk lookup:
    # Cashiers: 4720, Retail Salespersons: 4760, Registered Nurses: 3255
    # Software Developers: 1020, Data Scientists: 1200, Truck Drivers: 9130
    # Laborers/Material Movers: 9620, Accountants: 800
    example_pairs = [
        {"from": "Cashiers", "to": "Retail Salespersons", "census_from": 4720, "census_to": 4760},
        {"from": "Cashiers", "to": "Registered Nurses", "census_from": 4720, "census_to": 3255},
        {"from": "Software Developers", "to": "Data Scientists", "census_from": 1020, "census_to": 1200},
        {"from": "Truck Drivers", "to": "Material Movers", "census_from": 9130, "census_to": 9620},
        {"from": "Accountants", "to": "Software Developers", "census_from": 800, "census_to": 1020},
        {"from": "Retail Salespersons", "to": "Accountants", "census_from": 4760, "census_to": 800},
    ]

    # Look up actual distances
    code_to_idx = {c: i for i, c in enumerate(census_codes)}
    examples_with_distances = []
    for ex in example_pairs:
        if ex["census_from"] in code_to_idx and ex["census_to"] in code_to_idx:
            i = code_to_idx[ex["census_from"]]
            j = code_to_idx[ex["census_to"]]
            d_wass = float(d_sem[i, j])
            examples_with_distances.append({
                "from": ex["from"],
                "to": ex["to"],
                "d_wass": round(d_wass, 4),
            })

    examples_with_costs = compute_example_transition_costs(
        examples_with_distances, sc_per_unit, mean_annual_wage
    )
    external_calibration["examples"] = examples_with_costs

    print("\n    Example transition costs (externally calibrated):")
    for ex in examples_with_costs:
        print(f"      {ex['from']:25} -> {ex['to']:20}: d={ex['d_wass']:.2f}, cost={ex['cost_wage_years']:.2f} yrs (${ex['cost_dollars']:,.0f})")

    # =========================================================================
    # Step 8: Compute verdict and diagnostics
    # =========================================================================
    print("\n[8] Computing verdict...")

    # Original verdict (for wage identification)
    wage_verdict = compute_verdict(
        beta_wage=result_m1.beta_wage,
        gamma_sem=result_m1.gamma_sem,
        gamma_inst=result_m1.gamma_inst,
        sc_typical_wage_years=switching_costs["sc_typical_wage_years"],
    )
    print(f"    Wage identification verdict: {wage_verdict}")

    # Revised verdict after external calibration
    # Check if examples pass smell test
    smell_test_passed = True
    for ex in examples_with_costs:
        # Cashier -> Retail should be < 2 years (similar jobs)
        if "Cashiers" in ex["from"] and "Retail" in ex["to"]:
            if ex["cost_wage_years"] > 2.0:
                smell_test_passed = False
        # Cashier -> RN should be > 1 year (requires training)
        if "Cashiers" in ex["from"] and "Nurse" in ex["to"]:
            if ex["cost_wage_years"] < 1.0:
                smell_test_passed = False

    revised_verdict = "calibrated" if smell_test_passed else "investigate"
    print(f"    Revised verdict (calibrated): {revised_verdict}")
    print(f"    Smell test passed: {smell_test_passed}")

    diagnostics = {
        "beta_wage_positive": result_m1.beta_wage > 0,
        "gamma_sem_positive": result_m1.gamma_sem > 0,
        "gamma_inst_positive": result_m1.gamma_inst > 0,
    }

    wage_identification_failure = {
        "diagnosed": True,
        "reason": "Occupation-mean wages do not capture entry wages switchers receive",
        "m1_beta_wage": result_m1.beta_wage,
        "m2_beta_wage": result_m2.beta_wage,
        "data_requirement": "Individual-level wages at transition (LEHD, admin data)",
    }

    # =========================================================================
    # Step 9: Assemble and save results
    # =========================================================================
    print("\n[9] Saving results...")

    output = ScaledCostsResult(
        version="0.7b.0",
        wage_data={
            "oes_year": oes_year,
            "coverage": coverage,
            "mean_annual_wage": mean_annual_wage,
            "n_occupations_with_wages": len(wages_df),
        },
        model_m1=result_m1.to_dict(),
        model_m2=result_m2.to_dict(),
        model_m3=result_m3.to_dict(),
        switching_costs=switching_costs,
        benchmark_comparison={
            "dix_carneiro_range": dix_carneiro_range,
            "our_estimate_wage_years": switching_costs["sc_typical_wage_years"],
            "within_range": within_range,
        },
        diagnostics=diagnostics,
        verdict=wage_verdict,
        external_calibration=external_calibration,
        wage_identification_failure=wage_identification_failure,
        revised_verdict=revised_verdict,
    )

    output_path = Path("outputs/experiments/scaled_costs_v070b.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(str(output_path))
    print(f"    Saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nWage Data:")
    print(f"  OES Year: {oes_year}")
    print(f"  Coverage: {coverage:.1%}")
    print(f"  Mean Annual Wage: ${mean_annual_wage:,.0f}")

    print(f"\nModel M1 (Primary - Log Wages):")
    print(f"  gamma_sem: {result_m1.gamma_sem:.4f} (t={result_m1.gamma_sem_t:.1f})")
    print(f"  gamma_inst: {result_m1.gamma_inst:.4f} (t={result_m1.gamma_inst_t:.1f})")
    print(f"  beta_wage: {result_m1.beta_wage:.4f} (t={result_m1.beta_wage_t:.1f})")

    print(f"\nSwitching Costs:")
    print(f"  Median Wasserstein: {d_sem_median:.4f}")
    print(f"  Median Institutional: {d_inst_median:.4f}")
    print(f"  Typical Cost: {switching_costs['sc_typical_wage_years']:.2f} wage-years")
    print(f"  Typical Cost: ${switching_costs['sc_typical_dollars']:,.0f}")

    print(f"\nBenchmark:")
    print(f"  Dix-Carneiro: {dix_carneiro_range[0]}-{dix_carneiro_range[1]} wage-years")
    print(f"  Our estimate: {switching_costs['sc_typical_wage_years']:.2f} wage-years")
    print(f"  Within range: {within_range}")

    print(f"\nDiagnostics:")
    print(f"  β_wage > 0: {diagnostics['beta_wage_positive']}")
    print(f"  γ_sem > 0: {diagnostics['gamma_sem_positive']}")
    print(f"  γ_inst > 0: {diagnostics['gamma_inst_positive']}")

    print(f"\nWage Identification: {wage_verdict}")

    print(f"\nExternal Calibration (Dix-Carneiro benchmark):")
    print(f"  SC per unit Wasserstein: {sc_per_unit:.2f} wage-years")
    print(f"  SC per unit Wasserstein: ${external_calibration['sc_per_unit_wasserstein_dollars']:,.0f}")

    print(f"\nExample Transition Costs:")
    for ex in examples_with_costs:
        print(f"  {ex['from']:25} -> {ex['to']:20}: {ex['cost_wage_years']:.2f} yrs (${ex['cost_dollars']:,.0f})")

    print(f"\nREVISED VERDICT: {revised_verdict}")
    print("=" * 60)

    return output


if __name__ == "__main__":
    main()
