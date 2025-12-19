"""
Test B (Proper): Autor-Dorn Polarization with Independent CZ-level CSH

Tests whether continuous semantic height (CSH) adds explanatory power
beyond discrete routine share hypothesis (RSH) for employment polarization.

Key difference from v0.7.2.3 proxy: CSH_cz is computed from actual
CZ × occupation employment data (from IPUMS Census), not as a linear
transformation of RSH_cz.

Specification (matches Autor-Dorn 2013 Table 5):
    ΔY_czt = β₁·RSH_cz + β₃·CSH_resid_cz + X'γ + α_s + δ_t + ε_czt

Where:
- RSH_cz: Routine Share Hypothesis (binary: top tercile = 1)
- CSH_resid_cz: CSH residualized on RSH bands
- CSH_cz = Σ_k (emp_share_cz_k × CSH_k) using occupation-level CSH

Interpretation matrix threshold: p < 0.05 AND ΔR² ≥ 0.01
- "+": Significant with correct sign (negative for routine outcomes)
- "−": Significant with wrong sign
- "0": Not significant or small effect
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from task_space.battery.exposure import RSHExposure


def load_dorn_workfile() -> pd.DataFrame:
    """Load Dorn CZ-level workfile."""
    path = Path("data/external/dorn_replication/dorn_extracted")
    path = path / "Autor-Dorn-LowSkillServices-FileArchive.zip Folder/dta/workfile2012.dta"
    return pd.read_stata(path)


def load_cz_occ_employment(year: int) -> pd.DataFrame:
    """Load CZ × occupation employment from IPUMS processing."""
    path = Path(f"data/processed/cz_employment/cz_occ_employment_{year}.csv")
    return pd.read_csv(path)


def load_occ1990dd_to_onet() -> pd.DataFrame:
    """Load occ1990dd → O*NET crosswalk."""
    path = Path("data/processed/crosswalks/occ1990dd_to_onet_soc.csv")
    return pd.read_csv(path)


def compute_occ_level_csh(exposure: RSHExposure) -> pd.DataFrame:
    """
    Extract occupation-level CSH from RSHExposure.

    RSHExposure already has pre-computed CSH values at occ1990dd level.

    Returns DataFrame with occ1990dd, csh columns.
    """
    results = []

    for occ1990dd, csh in exposure._csh.items():
        results.append({
            "occ1990dd": occ1990dd,
            "csh": csh,
        })

    return pd.DataFrame(results)


def compute_cz_csh(
    cz_occ_emp: pd.DataFrame,
    occ_csh: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute CZ-level CSH as employment-weighted average of occupation CSH.

    CSH_cz = Σ_k (emp_share_cz_k × CSH_k)
    """
    # Merge occupation CSH onto CZ × occupation data
    merged = cz_occ_emp.merge(occ_csh, on="occ1990dd", how="left")

    # Filter to occupations with CSH
    merged = merged[merged["csh"].notna()].copy()

    # Compute CZ-level employment shares (within matched occupations)
    cz_totals = merged.groupby("czone")["employment"].sum().reset_index()
    cz_totals = cz_totals.rename(columns={"employment": "cz_total"})

    merged = merged.merge(cz_totals, on="czone")
    merged["emp_share"] = merged["employment"] / merged["cz_total"]

    # Compute weighted average CSH by CZ
    merged["weighted_csh"] = merged["emp_share"] * merged["csh"]

    cz_csh = merged.groupby("czone").agg({
        "weighted_csh": "sum",
        "cz_total": "first",
    }).reset_index()
    cz_csh = cz_csh.rename(columns={"weighted_csh": "csh_cz"})

    return cz_csh[["czone", "csh_cz"]]


def compute_occ_level_rsh(exposure: RSHExposure) -> pd.DataFrame:
    """
    Extract occupation-level RSH (binary) from RSHExposure.

    Returns DataFrame with occ1990dd, rsh columns.
    """
    results = []

    for occ1990dd in exposure._rti.keys():
        try:
            rsh = exposure.discrete_exposure(str(occ1990dd), binary=True)
            results.append({
                "occ1990dd": occ1990dd,
                "rsh": rsh,
            })
        except (KeyError, ValueError):
            continue

    return pd.DataFrame(results)


def compute_cz_rsh(
    cz_occ_emp: pd.DataFrame,
    occ_rsh: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute CZ-level RSH (discrete routine share).

    RSH_cz = Σ_k (emp_share_cz_k × RSH_k) where RSH_k ∈ {0, 1}
    """
    # Merge with CZ × occ data
    merged = cz_occ_emp.merge(occ_rsh, on="occ1990dd", how="left")
    merged = merged[merged["rsh"].notna()].copy()

    # CZ employment shares
    cz_totals = merged.groupby("czone")["employment"].sum().reset_index()
    cz_totals = cz_totals.rename(columns={"employment": "cz_total"})

    merged = merged.merge(cz_totals, on="czone")
    merged["emp_share"] = merged["employment"] / merged["cz_total"]

    # Weighted RSH
    merged["weighted_rsh"] = merged["emp_share"] * merged["rsh"]

    cz_rsh = merged.groupby("czone")["weighted_rsh"].sum().reset_index()
    cz_rsh = cz_rsh.rename(columns={"weighted_rsh": "rsh_cz"})

    return cz_rsh


def run_autor_dorn_regression(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: list[str],
    weight_col: str = "timepwt48",
) -> dict:
    """Run Autor-Dorn style regression with FE and weights."""
    # Drop missing
    all_cols = [outcome, treatment, weight_col] + controls
    data = df[all_cols].dropna()

    # Create state dummies
    state_dummies = pd.get_dummies(data["statefip"], prefix="state", drop_first=True)

    # Create time dummies
    time_dummies = pd.get_dummies(data["yr"], prefix="year", drop_first=True)

    # Model 1: Treatment + state FE + time FE
    X1 = pd.concat([
        data[[treatment]],
        state_dummies.astype(float),
        time_dummies.astype(float),
    ], axis=1)
    X1 = sm.add_constant(X1)

    model1 = sm.WLS(
        data[outcome].astype(float),
        X1.astype(float),
        weights=data[weight_col].astype(float),
    ).fit(cov_type="HC1")

    # Model 2: Add additional controls
    other_controls = [c for c in controls if c not in ["statefip", "yr"]]
    X2 = pd.concat([
        data[[treatment] + other_controls],
        state_dummies.astype(float),
        time_dummies.astype(float),
    ], axis=1)
    X2 = sm.add_constant(X2)

    model2 = sm.WLS(
        data[outcome].astype(float),
        X2.astype(float),
        weights=data[weight_col].astype(float),
    ).fit(cov_type="HC1")

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
            "n": int(model2.nobs),
        },
    }


def run_two_stage_regression(
    df: pd.DataFrame,
    outcome: str,
    rsh_col: str,
    csh_resid_col: str,
    controls: list[str],
    weight_col: str = "timepwt48",
) -> dict:
    """Run two-model regression: RSH only, then RSH + CSH_resid."""
    # Drop missing
    all_cols = [outcome, rsh_col, csh_resid_col, weight_col] + controls
    data = df[all_cols].dropna()

    # Create state dummies
    state_dummies = pd.get_dummies(data["statefip"], prefix="state", drop_first=True)
    time_dummies = pd.get_dummies(data["yr"], prefix="year", drop_first=True)

    # Model 1: RSH only
    X1 = pd.concat([
        data[[rsh_col]],
        state_dummies.astype(float),
        time_dummies.astype(float),
    ], axis=1)
    X1 = sm.add_constant(X1)

    model1 = sm.WLS(
        data[outcome].astype(float),
        X1.astype(float),
        weights=data[weight_col].astype(float),
    ).fit(cov_type="HC1")

    # Model 2: RSH + CSH_resid
    X2 = pd.concat([
        data[[rsh_col, csh_resid_col]],
        state_dummies.astype(float),
        time_dummies.astype(float),
    ], axis=1)
    X2 = sm.add_constant(X2)

    model2 = sm.WLS(
        data[outcome].astype(float),
        X2.astype(float),
        weights=data[weight_col].astype(float),
    ).fit(cov_type="HC1")

    return {
        "model1": {
            "beta_rsh": float(model1.params[rsh_col]),
            "se_rsh": float(model1.bse[rsh_col]),
            "pvalue_rsh": float(model1.pvalues[rsh_col]),
            "r2": float(model1.rsquared),
            "n": int(model1.nobs),
        },
        "model2": {
            "beta_rsh": float(model2.params[rsh_col]),
            "se_rsh": float(model2.bse[rsh_col]),
            "pvalue_rsh": float(model2.pvalues[rsh_col]),
            "beta_csh_resid": float(model2.params[csh_resid_col]),
            "se_csh_resid": float(model2.bse[csh_resid_col]),
            "pvalue_csh_resid": float(model2.pvalues[csh_resid_col]),
            "r2": float(model2.rsquared),
            "delta_r2": float(model2.rsquared - model1.rsquared),
            "n": int(model2.nobs),
        },
    }


def interpret_result(beta: float, pvalue: float, delta_r2: float) -> str:
    """
    Interpret Test B result.

    For routine outcomes, we expect negative coefficient (more routine → more decline).
    """
    if pvalue < 0.05 and delta_r2 >= 0.01:
        return "+"  # Significant and meaningful
    elif pvalue < 0.05 and beta > 0:
        return "-"  # Wrong sign
    else:
        return "0"  # Not significant or small effect


def main():
    print("=" * 70)
    print("Test B (Proper): Autor-Dorn Polarization with Independent CSH_cz")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    dorn_df = load_dorn_workfile()
    print(f"   Dorn workfile: {len(dorn_df)} CZ-periods")

    # Load 1980 CZ × occupation data (baseline for Autor-Dorn)
    cz_occ_1980 = load_cz_occ_employment(1980)
    print(f"   1980 CZ×occ cells: {len(cz_occ_1980)}")

    # Initialize exposure measure (loads from artifacts)
    print("\n2. Loading occupation-level CSH/RSH from artifacts...")
    exposure = RSHExposure.from_artifacts()
    print(f"   Occupations with RTI: {len(exposure._rti)}")
    print(f"   Occupations with CSH: {len(exposure._csh)}")

    # Extract occupation-level measures
    occ_csh = compute_occ_level_csh(exposure)
    occ_rsh = compute_occ_level_rsh(exposure)
    print(f"   Occupations with RSH (binary): {len(occ_rsh)}")

    # Compute CZ-level CSH
    print("\n3. Computing CZ-level CSH from occupation data...")
    cz_csh = compute_cz_csh(cz_occ_1980, occ_csh)
    print(f"   CZs with CSH: {len(cz_csh)}")

    # Compute CZ-level RSH for comparison
    print("\n4. Computing CZ-level RSH...")
    cz_rsh = compute_cz_rsh(cz_occ_1980, occ_rsh)
    print(f"   CZs with RSH: {len(cz_rsh)}")

    # KEY VALIDATION: Correlation between CSH_cz and RSH_cz
    merged_check = cz_csh.merge(cz_rsh, on="czone")
    r_csh_rsh = np.corrcoef(merged_check["csh_cz"], merged_check["rsh_cz"])[0, 1]
    print(f"\n   *** KEY VALIDATION ***")
    print(f"   r(CSH_cz, RSH_cz) = {r_csh_rsh:.4f}")
    print(f"   (Should be < 1.0; expect ~0.8-0.9)")

    if r_csh_rsh > 0.99:
        print("   WARNING: Correlation too high! CSH may not be independent.")

    # Merge onto Dorn workfile
    print("\n5. Merging with Dorn workfile...")
    analysis_df = dorn_df.merge(cz_csh, on="czone", how="left")
    analysis_df = analysis_df.merge(cz_rsh, on="czone", how="left")

    valid = analysis_df["csh_cz"].notna() & analysis_df["rsh_cz"].notna()
    print(f"   Valid CZ-periods: {valid.sum()} / {len(analysis_df)}")

    # Filter to 1980-2000 period (3 decades as in Autor-Dorn)
    analysis_df = analysis_df[
        (analysis_df["yr"].isin([1980, 1990, 2000])) &
        valid
    ].copy()
    print(f"   After period filter (1980-2000): {len(analysis_df)}")

    # Compute CSH_resid (residualized on RSH band)
    # Simple approach: regress CSH on RSH and take residuals
    from scipy import stats
    slope, intercept, _, _, _ = stats.linregress(
        analysis_df["rsh_cz"], analysis_df["csh_cz"]
    )
    analysis_df["csh_resid"] = analysis_df["csh_cz"] - (intercept + slope * analysis_df["rsh_cz"])

    # Report variance retained
    var_csh = analysis_df["csh_cz"].var()
    var_resid = analysis_df["csh_resid"].var()
    print(f"   CSH_resid variance retained: {var_resid/var_csh:.1%}")

    # Run regressions
    print("\n6. Running regressions...")
    print("-" * 70)

    outcomes = {
        "d_sh_routine33a": ("Δ routine share", "negative"),
        "d_shocc1_clericretail_nc": ("Δ clerical/retail share", "negative"),
        "d_shocc1_operator_nc": ("Δ operator share", "negative"),
        "d_shocc1_service_nc": ("Δ service share", "positive"),
        "d_shocc1_mgmtproftech_nc": ("Δ mgmt/prof/tech share", "ambiguous"),
    }

    controls = ["statefip", "yr"]
    results = {}

    for outcome_var, (outcome_label, expected_sign) in outcomes.items():
        print(f"\n   {outcome_label}:")

        result = run_two_stage_regression(
            analysis_df,
            outcome=outcome_var,
            rsh_col="rsh_cz",
            csh_resid_col="csh_resid",
            controls=controls,
        )

        results[outcome_var] = result

        # Print key stats
        m2 = result["model2"]
        print(f"     β(RSH): {m2['beta_rsh']:.4f} (p={m2['pvalue_rsh']:.4f})")
        print(f"     β(CSH_resid): {m2['beta_csh_resid']:.4f} (p={m2['pvalue_csh_resid']:.4f})")
        print(f"     ΔR²: {m2['delta_r2']:.4f}")

        # Interpret
        verdict = interpret_result(
            m2["beta_csh_resid"],
            m2["pvalue_csh_resid"],
            m2["delta_r2"],
        )
        results[outcome_var]["verdict"] = verdict
        print(f"     Verdict: {verdict}")

    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETATION MATRIX")
    print("=" * 70)

    verdicts = {k: v["verdict"] for k, v in results.items()}
    for outcome_var, (outcome_label, _) in outcomes.items():
        print(f"   {outcome_label}: {verdicts[outcome_var]}")

    n_plus = sum(1 for v in verdicts.values() if v == "+")
    n_minus = sum(1 for v in verdicts.values() if v == "-")
    n_zero = sum(1 for v in verdicts.values() if v == "0")

    print(f"\n   Summary: {n_plus}+, {n_minus}−, {n_zero}(0)")

    # Save results
    output = {
        "version": "0.7.2.3",
        "test": "B",
        "description": "Autor-Dorn polarization: CSH_resid vs RSH (proper CZ aggregation)",
        "timestamp": datetime.now().isoformat(),
        "validation": {
            "r_csh_rsh": float(r_csh_rsh),
            "csh_resid_variance_retained": float(var_resid / var_csh),
            "n_cz_with_csh": int(len(cz_csh)),
            "methodology": "Independent CZ-level CSH from IPUMS 1980 Census via occupation weights",
        },
        "sample": {
            "n_cz": int(analysis_df["czone"].nunique()),
            "n_periods": 3,
            "period": "1980-2000",
            "n_obs": int(len(analysis_df)),
        },
        "outcomes": results,
        "interpretation_matrix": verdicts,
        "summary": {
            "plus": n_plus,
            "minus": n_minus,
            "zero": n_zero,
        },
    }

    output_path = Path("outputs/experiments/battery_test_b_v0723.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    main()
