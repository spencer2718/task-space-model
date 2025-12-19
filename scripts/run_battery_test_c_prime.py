"""
Test C': Robot Exposure and Employment Decline

Tests whether occupations with higher robot exposure experienced larger
employment declines during the robot adoption period (1990-2005).

Based on Webb (2020) and Acemoglu-Restrepo (2020) findings that robots
automate routine manual tasks like welding, assembling, and material handling.

Specification:
    Δ ln(emp_share)_k = β₀ + β₁ · RobotExp_k + γ · X_k + ε_k

Where:
- RobotExp_k: Fraction of occupation's DWAs that are robot-exposed
- X_k: Controls (baseline wage, baseline employment share)

Expected: β₁ < 0 (robot-exposed occupations declined)

Interpretation:
- "+": p < 0.05 AND ΔR² ≥ 0.01 AND correct sign (negative)
- "−": p < 0.05 AND wrong sign (positive)
- "0": Otherwise
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from task_space.battery.robot_exposure import RobotExposure


def load_occ1990dd_to_onet() -> pd.DataFrame:
    """Load pre-built occ1990dd -> O*NET-SOC crosswalk."""
    path = Path("data/processed/crosswalks/occ1990dd_to_onet_soc.csv")
    return pd.read_csv(path)


def load_dorn_occ_data() -> pd.DataFrame:
    """Load Dorn occupation-level employment data."""
    path = Path("data/external/dorn_replication/dorn_extracted")
    path = path / "Autor-Dorn-LowSkillServices-FileArchive.zip Folder/dta/occ1990dd_data2012.dta"
    return pd.read_stata(path)


def compute_employment_change(df: pd.DataFrame) -> pd.DataFrame:
    """Compute employment share changes 1990-2005."""
    df = df.copy()

    # Employment share change (log difference)
    df["delta_ln_emp"] = np.log(df["sh_empl2005"] + 1e-10) - np.log(df["sh_empl1990"] + 1e-10)

    # Baseline controls
    df["ln_wage_1990"] = np.log(df["avg_hrwage1990"] + 1e-10)
    df["ln_emp_1990"] = np.log(df["sh_empl1990"] + 1e-10)

    return df


def merge_robot_exposure(
    dorn_df: pd.DataFrame,
    robot: RobotExposure,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """Merge robot exposure onto Dorn occupation data via crosswalk."""
    # Get robot exposures for all O*NET occupations
    exposures = robot.compute_all_exposures()

    # Merge crosswalk to get occ1990dd -> O*NET-SOC mapping
    # The crosswalk has occ1990dd and onet_soc columns
    merged = dorn_df.merge(
        crosswalk[["occ1990dd", "onet_soc"]],
        on="occ1990dd",
        how="left"
    )

    # Merge robot exposure
    merged = merged.merge(
        exposures[["onet_soc", "robot_exposure", "robot_exposed_binary"]],
        on="onet_soc",
        how="left"
    )

    return merged


def run_regression(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: list[str],
) -> dict:
    """Run OLS regression and return results."""
    # Drop missing
    cols = [outcome, treatment] + controls
    data = df[cols].dropna()

    # Model 1: Treatment only
    X1 = sm.add_constant(data[treatment])
    model1 = sm.OLS(data[outcome], X1).fit(cov_type="HC1")

    # Model 2: Treatment + controls
    X2 = sm.add_constant(data[[treatment] + controls])
    model2 = sm.OLS(data[outcome], X2).fit(cov_type="HC1")

    return {
        "model1": {
            "beta": float(model1.params[treatment]),
            "se": float(model1.bse[treatment]),
            "pvalue": float(model1.pvalues[treatment]),
            "r2": float(model1.rsquared),
            "n": int(model1.nobs),
        },
        "model2": {
            "beta": float(model2.params[treatment]),
            "se": float(model2.bse[treatment]),
            "pvalue": float(model2.pvalues[treatment]),
            "r2": float(model2.rsquared),
            "delta_r2": float(model2.rsquared - model1.rsquared),
            "n": int(model2.nobs),
        },
    }


def interpret_result(beta: float, pvalue: float, delta_r2: float, expected_sign: str) -> str:
    """
    Interpret regression result.

    Args:
        beta: Coefficient estimate
        pvalue: p-value
        delta_r2: Change in R² from adding treatment
        expected_sign: "negative" or "positive"

    Returns:
        "+", "−", or "0"
    """
    correct_sign = (expected_sign == "negative" and beta < 0) or \
                   (expected_sign == "positive" and beta > 0)

    if pvalue < 0.05:
        if correct_sign and delta_r2 >= 0.01:
            return "+"
        elif not correct_sign:
            return "-"
    return "0"


def main():
    print("=" * 60)
    print("Test C': Robot Exposure and Employment Decline")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    dorn_df = load_dorn_occ_data()
    print(f"   Dorn occupations: {len(dorn_df)}")

    # Load crosswalk
    crosswalk = load_occ1990dd_to_onet()
    print(f"   Crosswalk entries: {len(crosswalk)}")

    # Initialize robot exposure
    print("\n2. Computing robot exposure...")
    robot = RobotExposure()
    print(f"   Robot-exposed DWAs: {robot.n_robot_dwas}")

    stats = robot.get_exposure_stats()
    print(f"   Occupations with exposure data: {stats['n_occupations']}")
    print(f"   High-exposure occupations: {stats['n_high_exposure']}")
    print(f"   Mean exposure: {stats['mean']:.4f}")

    # Compute employment change
    print("\n3. Computing employment changes (1990-2005)...")
    dorn_df = compute_employment_change(dorn_df)

    # Merge robot exposure
    print("\n4. Merging robot exposure via crosswalk...")
    merged = merge_robot_exposure(dorn_df, robot, crosswalk)

    valid = merged["robot_exposure"].notna()
    print(f"   Valid matches: {valid.sum()} / {len(merged)} ({100*valid.sum()/len(merged):.1f}%)")

    # Filter to valid observations
    analysis_df = merged[valid].copy()

    # Drop occupations with zero employment in either period
    analysis_df = analysis_df[
        (analysis_df["sh_empl1990"] > 0) &
        (analysis_df["sh_empl2005"] > 0)
    ].copy()
    print(f"   Final sample: {len(analysis_df)} occupations")

    # Run regressions
    print("\n5. Running regressions...")
    print("-" * 60)

    outcomes = {
        "delta_ln_emp": "Δ ln(employment share) 1990-2005",
    }

    results = {}
    for outcome_var, outcome_label in outcomes.items():
        print(f"\n   Outcome: {outcome_label}")

        # Continuous exposure
        result_cont = run_regression(
            analysis_df,
            outcome=outcome_var,
            treatment="robot_exposure",
            controls=["ln_wage_1990", "ln_emp_1990"],
        )

        # Binary exposure
        result_binary = run_regression(
            analysis_df,
            outcome=outcome_var,
            treatment="robot_exposed_binary",
            controls=["ln_wage_1990", "ln_emp_1990"],
        )

        results[outcome_var] = {
            "continuous": result_cont,
            "binary": result_binary,
        }

        # Print continuous results
        print(f"\n   Continuous robot exposure:")
        print(f"     β = {result_cont['model2']['beta']:.4f}")
        print(f"     SE = {result_cont['model2']['se']:.4f}")
        print(f"     p-value = {result_cont['model2']['pvalue']:.4f}")
        print(f"     R² = {result_cont['model2']['r2']:.4f}")
        print(f"     n = {result_cont['model2']['n']}")

        # Interpret
        verdict = interpret_result(
            beta=result_cont["model2"]["beta"],
            pvalue=result_cont["model2"]["pvalue"],
            delta_r2=result_cont["model2"]["delta_r2"],
            expected_sign="negative",
        )
        print(f"     Verdict: {verdict}")
        results[outcome_var]["verdict"] = verdict

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for outcome_var, outcome_label in outcomes.items():
        r = results[outcome_var]
        print(f"\n{outcome_label}:")
        print(f"  Continuous: β={r['continuous']['model2']['beta']:.4f}, p={r['continuous']['model2']['pvalue']:.4f}")
        print(f"  Binary: β={r['binary']['model2']['beta']:.4f}, p={r['binary']['model2']['pvalue']:.4f}")
        print(f"  Verdict: {r['verdict']}")

    # Save results
    output = {
        "version": "0.7.2.5",
        "test": "C'",
        "description": "Robot exposure and employment decline (1990-2005)",
        "timestamp": datetime.now().isoformat(),
        "sample": {
            "n_occupations": len(analysis_df),
            "n_robot_exposed_dwas": robot.n_robot_dwas,
            "n_high_exposure_occs": stats["n_high_exposure"],
            "crosswalk_coverage": f"{100*valid.sum()/len(merged):.1f}%",
        },
        "robot_exposure_stats": stats,
        "results": results,
        "interpretation": {
            "delta_ln_emp": results["delta_ln_emp"]["verdict"],
        },
        "summary": {
            "verdict": results["delta_ln_emp"]["verdict"],
            "beta_robot_exposure": results["delta_ln_emp"]["continuous"]["model2"]["beta"],
            "p_value": results["delta_ln_emp"]["continuous"]["model2"]["pvalue"],
            "interpretation": (
                "Robot-exposed occupations experienced significantly larger employment declines"
                if results["delta_ln_emp"]["verdict"] == "+"
                else "No significant relationship found between robot exposure and employment decline"
                if results["delta_ln_emp"]["verdict"] == "0"
                else "Unexpected: Robot-exposed occupations grew faster"
            ),
        },
    }

    output_path = Path("outputs/experiments/battery_test_c_prime_v0725.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    main()
