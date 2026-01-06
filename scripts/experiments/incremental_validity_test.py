#!/usr/bin/env python3
"""
Incremental Validity Test v0.6.5.3

Tests whether semantic exposure adds predictive power for employment
changes beyond existing automation indices (RTI, AIOE).

Design:
    Outcome: ΔEmployment = log(emp_2024) - log(emp_2019)
    Predictors: RTI (proper AA 16-element), AIOE, SemanticExposure
    Controls: Telework (Dingel-Neiman), Job Zone (O*NET)

Models:
    1. ΔEmp ~ RTI (baseline)
    2. ΔEmp ~ RTI + SemanticExposure
    3. ΔEmp ~ AIOE (baseline)
    4. ΔEmp ~ AIOE + SemanticExposure
    5. ΔEmp ~ RTI + AIOE + SemanticExposure (horse race)
    6. ΔEmp ~ All + Controls (telework, job_zone)

Output: outputs/experiments/incremental_validity_v0653.json

Usage:
    PYTHONPATH=src python scripts/experiments/incremental_validity_test.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from task_space.data.aioe import load_aioe
from task_space.data.oes import load_oes_employment
from task_space.data.classifications import get_aa_task_scores_df, get_job_zones
from task_space.data.telework import load_telework
from task_space.data.crosswalk import onet_to_soc
from task_space.domain import build_dwa_occupation_measures
from task_space.similarity.kernel import build_kernel_matrix
from task_space.shocks.propagation import propagate_shock
from task_space.shocks.profiles import shock_capability_v1
from task_space.data.artifacts import get_embeddings, get_distance_matrix
from task_space.data.onet import load_dwa_reference


# Paths
ONET_PATH = Path("data/onet/db_30_0_excel")
OUTPUT_PATH = Path("outputs/experiments/incremental_validity_v0653.json")


def load_and_merge_data() -> pd.DataFrame:
    """
    Merge: OES 2019, OES 2024, AIOE, RTI (proper AA), Semantic Exposure, Controls
    Compute: delta_log_emp = log(emp_2024) - log(emp_2019)
    Return: analysis-ready DataFrame
    """
    print("Loading data sources...")

    # Load OES employment
    emp_2019 = load_oes_employment(2019)
    emp_2024 = load_oes_employment(2024)

    print(f"  OES 2019: {len(emp_2019)} occupations")
    print(f"  OES 2024: {len(emp_2024)} occupations")

    # Merge employment data
    emp = emp_2019[["soc_code", "tot_emp"]].rename(columns={"tot_emp": "emp_2019"})
    emp = emp.merge(
        emp_2024[["soc_code", "tot_emp"]].rename(columns={"tot_emp": "emp_2024"}),
        on="soc_code",
        how="inner"
    )

    # Remove duplicates (some SOC codes appear twice in OES)
    emp = emp.drop_duplicates(subset=["soc_code"])

    print(f"  Employment panel: {len(emp)} occupations with both years")

    # Compute employment change
    emp["log_emp_2019"] = np.log(emp["emp_2019"])
    emp["log_emp_2024"] = np.log(emp["emp_2024"])
    emp["delta_log_emp"] = emp["log_emp_2024"] - emp["log_emp_2019"]

    # Load AIOE
    aioe_data = load_aioe(include_lm=True)
    aioe = aioe_data.scores[["soc_code", "aioe", "lm_aioe"]].copy()
    print(f"  AIOE: {len(aioe)} occupations")

    # Merge AIOE
    df = emp.merge(aioe, on="soc_code", how="inner")
    print(f"  After AIOE merge: {len(df)} occupations")

    # Load proper AA RTI from 16-element composite
    print("  Loading AA task scores (16 O*NET elements)...")
    aa_df = get_aa_task_scores_df(ONET_PATH)

    # Map O*NET codes to SOC codes
    aa_df["soc_code"] = aa_df["occ_code"].apply(onet_to_soc)

    # Average RTI across O*NET codes that map to same SOC
    rti_agg = aa_df.groupby("soc_code").agg({
        "rti": "mean",
        "nr_cognitive_analytical": "mean",
        "nr_cognitive_interpersonal": "mean",
        "routine_cognitive": "mean",
        "routine_manual": "mean",
        "nr_manual_physical": "mean",
    }).reset_index()
    print(f"  AA RTI: {len(rti_agg)} SOC codes")

    # Merge RTI
    df = df.merge(rti_agg, on="soc_code", how="inner")
    print(f"  After RTI merge: {len(df)} occupations")

    # Compute semantic exposure
    print("Computing semantic exposure...")
    semantic_exp = compute_semantic_exposure()
    df = df.merge(semantic_exp, on="soc_code", how="inner")
    print(f"  After semantic exposure merge: {len(df)} occupations")

    # Load controls
    print("Loading controls...")

    # Telework (Dingel-Neiman)
    try:
        telework_data = load_telework()
        telework = telework_data.scores[["soc_code", "teleworkable"]].copy()
        df = df.merge(telework, on="soc_code", how="left")
        print(f"  Telework: {df['teleworkable'].notna().sum()} occupations matched")
    except FileNotFoundError:
        print("  Telework: Data not found, skipping")
        df["teleworkable"] = np.nan

    # Job Zone (O*NET)
    job_zones = get_job_zones(ONET_PATH)
    jz_df = pd.DataFrame([
        {"onet_code": code, "job_zone": jz}
        for code, jz in job_zones.items()
    ])
    jz_df["soc_code"] = jz_df["onet_code"].apply(onet_to_soc)
    jz_agg = jz_df.groupby("soc_code")["job_zone"].mean().reset_index()
    df = df.merge(jz_agg, on="soc_code", how="left")
    print(f"  Job Zone: {df['job_zone'].notna().sum()} occupations matched")

    # Add 2-digit SOC for clustering
    df["soc_2digit"] = df["soc_code"].str[:2]

    # Standardize predictors for comparability
    for col in ["rti", "aioe", "lm_aioe", "semantic_exposure",
                "nr_cognitive_analytical", "nr_cognitive_interpersonal",
                "routine_cognitive", "routine_manual", "nr_manual_physical"]:
        df[f"{col}_std"] = (df[col] - df[col].mean()) / df[col].std()

    # Standardize controls if present
    if df["teleworkable"].notna().any():
        df["telework_std"] = (df["teleworkable"] - df["teleworkable"].mean()) / df["teleworkable"].std()
    if df["job_zone"].notna().any():
        df["job_zone_std"] = (df["job_zone"] - df["job_zone"].mean()) / df["job_zone"].std()

    return df


def compute_semantic_exposure() -> pd.DataFrame:
    """
    Use existing infrastructure to compute semantic exposure:
    - Load domain, occupation measures
    - Apply capability_v1 shock profile (cognitive positive, physical negative)
    - Propagate through kernel
    - Return occupation-level exposure scores aggregated to SOC
    """
    # Load occupation measures
    occ_measures = build_dwa_occupation_measures(ONET_PATH)

    # Load DWA reference to get activity titles
    dwa_ref = load_dwa_reference(ONET_PATH)
    dwa_titles = dwa_ref.set_index("DWA ID")["DWA Title"].to_dict()

    # Get activity titles in order of activity_ids
    activity_titles = [dwa_titles.get(aid, aid) for aid in occ_measures.activity_ids]

    # Get embeddings and compute distance matrix
    embeddings = get_embeddings(activity_titles)
    dist_matrix = get_distance_matrix(embeddings, metric="cosine")

    # Build kernel (auto-calibrates sigma)
    kernel_matrix, sigma = build_kernel_matrix(dist_matrix)
    print(f"  Kernel sigma: {sigma:.4f}")

    # Create shock profile using capability_v1
    # (cognitive positive, physical negative, technical moderate)
    I_t = shock_capability_v1(
        domain=occ_measures,
        onet_path=ONET_PATH,
    )

    # Propagate shock
    result = propagate_shock(
        I_t=I_t,
        kernel_matrix=kernel_matrix,
        occ_measures=occ_measures.occupation_matrix,
        shock_name="capability_v1",
    )

    # Build DataFrame with O*NET codes
    exposure_df = pd.DataFrame({
        "onet_code": occ_measures.occupation_codes,
        "semantic_exposure": result.E,
    })

    # Map to SOC codes and aggregate
    exposure_df["soc_code"] = exposure_df["onet_code"].apply(onet_to_soc)
    exposure_agg = exposure_df.groupby("soc_code")["semantic_exposure"].mean().reset_index()

    return exposure_agg


def run_ols_clustered(y, X, clusters) -> dict:
    """
    Run OLS with clustered standard errors.

    Args:
        y: Dependent variable (Series)
        X: Independent variables (DataFrame with const)
        clusters: Cluster variable (Series)

    Returns:
        Dict with coefficients, std_errors, t_statistics, p_values, r2
    """
    model = sm.OLS(y, X)
    # Fit with clustered SEs
    results = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})

    return {
        "r2": results.rsquared,
        "r2_adj": results.rsquared_adj,
        "n_obs": int(results.nobs),
        "n_clusters": len(clusters.unique()),
        "coefficients": results.params.to_dict(),
        "std_errors": results.bse.to_dict(),
        "t_statistics": results.tvalues.to_dict(),
        "p_values": results.pvalues.to_dict(),
    }


def run_regressions(df: pd.DataFrame) -> dict:
    """
    Run all regression models.

    Models:
        1. ΔEmp ~ RTI
        2. ΔEmp ~ RTI + SemanticExposure
        3. ΔEmp ~ AIOE
        4. ΔEmp ~ AIOE + SemanticExposure
        5. ΔEmp ~ RTI + AIOE + SemanticExposure (horse race)
        6. ΔEmp ~ LM_AIOE + Semantic (robustness)
        7. ΔEmp ~ Full + Telework + JobZone (with controls)
    """
    y = df["delta_log_emp"]
    clusters = df["soc_2digit"]

    models = {}

    # Model 1: RTI only (proper AA RTI)
    X1 = sm.add_constant(df[["rti_std"]])
    models["model_1_rti_only"] = run_ols_clustered(y, X1, clusters)

    # Model 2: RTI + Semantic
    X2 = sm.add_constant(df[["rti_std", "semantic_exposure_std"]])
    models["model_2_rti_semantic"] = run_ols_clustered(y, X2, clusters)

    # Model 3: AIOE only
    X3 = sm.add_constant(df[["aioe_std"]])
    models["model_3_aioe_only"] = run_ols_clustered(y, X3, clusters)

    # Model 4: AIOE + Semantic
    X4 = sm.add_constant(df[["aioe_std", "semantic_exposure_std"]])
    models["model_4_aioe_semantic"] = run_ols_clustered(y, X4, clusters)

    # Model 5: Full horse race
    X5 = sm.add_constant(df[["rti_std", "aioe_std", "semantic_exposure_std"]])
    models["model_5_full"] = run_ols_clustered(y, X5, clusters)

    # Model 6: LM_AIOE + Semantic (robustness)
    X6 = sm.add_constant(df[["lm_aioe_std", "semantic_exposure_std"]])
    models["model_6_lm_aioe_semantic"] = run_ols_clustered(y, X6, clusters)

    # Model 7: Full + Controls (if available)
    if "telework_std" in df.columns and "job_zone_std" in df.columns:
        # Use subset with complete cases for controls
        df_complete = df.dropna(subset=["telework_std", "job_zone_std"])
        if len(df_complete) > 50:  # Need sufficient obs
            y_ctrl = df_complete["delta_log_emp"]
            clusters_ctrl = df_complete["soc_2digit"]
            X7 = sm.add_constant(df_complete[[
                "rti_std", "aioe_std", "semantic_exposure_std",
                "telework_std", "job_zone_std"
            ]])
            models["model_7_full_controls"] = run_ols_clustered(y_ctrl, X7, clusters_ctrl)

            # Model 8: Semantic + Controls only (test incremental over controls)
            X8 = sm.add_constant(df_complete[[
                "telework_std", "job_zone_std", "semantic_exposure_std"
            ]])
            models["model_8_semantic_controls"] = run_ols_clustered(y_ctrl, X8, clusters_ctrl)

            # Model 9: Controls only (baseline for comparison)
            X9 = sm.add_constant(df_complete[[
                "telework_std", "job_zone_std"
            ]])
            models["model_9_controls_only"] = run_ols_clustered(y_ctrl, X9, clusters_ctrl)

    return models


def compute_correlations(df: pd.DataFrame) -> dict:
    """Compute correlation matrix between predictors."""
    cols = ["rti_std", "aioe_std", "lm_aioe_std", "semantic_exposure_std"]
    if "telework_std" in df.columns:
        cols.append("telework_std")
    if "job_zone_std" in df.columns:
        cols.append("job_zone_std")

    corr = df[cols].corr()

    result = {
        "rti_aioe": corr.loc["rti_std", "aioe_std"],
        "rti_semantic": corr.loc["rti_std", "semantic_exposure_std"],
        "aioe_semantic": corr.loc["aioe_std", "semantic_exposure_std"],
        "aioe_lm_aioe": corr.loc["aioe_std", "lm_aioe_std"],
        "lm_aioe_semantic": corr.loc["lm_aioe_std", "semantic_exposure_std"],
    }

    if "telework_std" in cols:
        result["rti_telework"] = corr.loc["rti_std", "telework_std"]
        result["semantic_telework"] = corr.loc["semantic_exposure_std", "telework_std"]
    if "job_zone_std" in cols:
        result["rti_jobzone"] = corr.loc["rti_std", "job_zone_std"]
        result["semantic_jobzone"] = corr.loc["semantic_exposure_std", "job_zone_std"]

    return result


def compute_vif(df: pd.DataFrame) -> dict:
    """Compute Variance Inflation Factors for multicollinearity check."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = df[["rti_std", "aioe_std", "semantic_exposure_std"]]
    X = sm.add_constant(X)

    vif_data = {}
    for i, col in enumerate(X.columns):
        if col != "const":
            vif_data[col] = variance_inflation_factor(X.values, i)

    return vif_data


def main():
    print("=" * 60)
    print("Incremental Validity Test v0.6.5.3")
    print("=" * 60)
    print()
    print("RTI: Proper Acemoglu-Autor 16-element composite")
    print("Controls: Telework (Dingel-Neiman), Job Zone (O*NET)")
    print()

    # Load and merge data
    df = load_and_merge_data()
    print()

    # Compute correlations
    print("Predictor correlations:")
    corr = compute_correlations(df)
    for k, v in corr.items():
        if pd.notna(v):
            print(f"  {k}: {v:.3f}")
    print()

    # Compute VIF
    print("Variance Inflation Factors:")
    vif = compute_vif(df)
    for k, v in vif.items():
        print(f"  {k}: {v:.2f}")
    print()

    # Run regressions
    print("Running regressions...")
    models = run_regressions(df)
    print()

    # Print results table
    print("=" * 80)
    print("REGRESSION RESULTS (Clustered SEs by 2-digit SOC)")
    print("=" * 80)
    print()
    print(f"{'Model':<35} {'R²':>8} {'R²adj':>8} {'N':>6} {'Clusters':>8}")
    print("-" * 80)
    for name, result in models.items():
        print(f"{name:<35} {result['r2']:>8.4f} {result['r2_adj']:>8.4f} "
              f"{result['n_obs']:>6} {result['n_clusters']:>8}")
    print()

    # Detailed coefficients
    print("COEFFICIENT DETAILS")
    print("-" * 80)
    for name, result in models.items():
        print(f"\n{name}:")
        for var in result["coefficients"]:
            coef = result["coefficients"][var]
            se = result["std_errors"][var]
            t = result["t_statistics"][var]
            p = result["p_values"][var]
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"  {var:<25} β={coef:>8.4f}  SE={se:>7.4f}  t={t:>7.2f}  p={p:.4f} {sig}")

    # Compute incremental R²
    incremental_r2 = {
        "semantic_over_rti": models["model_2_rti_semantic"]["r2"] - models["model_1_rti_only"]["r2"],
        "semantic_over_aioe": models["model_4_aioe_semantic"]["r2"] - models["model_3_aioe_only"]["r2"],
    }

    # Add incremental R² over controls if available
    if "model_9_controls_only" in models and "model_8_semantic_controls" in models:
        incremental_r2["semantic_over_controls"] = (
            models["model_8_semantic_controls"]["r2"] - models["model_9_controls_only"]["r2"]
        )

    print()
    print("INCREMENTAL R² FROM SEMANTIC EXPOSURE")
    print("-" * 80)

    # Handle potential division by zero
    rti_r2 = models["model_1_rti_only"]["r2"]
    aioe_r2 = models["model_3_aioe_only"]["r2"]

    if rti_r2 > 0.0001:
        print(f"  Over RTI alone:  {incremental_r2['semantic_over_rti']:+.4f} "
              f"({incremental_r2['semantic_over_rti'] / rti_r2 * 100:+.1f}%)")
    else:
        print(f"  Over RTI alone:  {incremental_r2['semantic_over_rti']:+.4f} (RTI R² ≈ 0)")

    print(f"  Over AIOE alone: {incremental_r2['semantic_over_aioe']:+.4f} "
          f"({incremental_r2['semantic_over_aioe'] / aioe_r2 * 100:+.1f}%)")

    if "semantic_over_controls" in incremental_r2:
        ctrl_r2 = models["model_9_controls_only"]["r2"]
        if ctrl_r2 > 0.0001:
            print(f"  Over controls:   {incremental_r2['semantic_over_controls']:+.4f} "
                  f"({incremental_r2['semantic_over_controls'] / ctrl_r2 * 100:+.1f}%)")
        else:
            print(f"  Over controls:   {incremental_r2['semantic_over_controls']:+.4f}")

    # Interpretation
    semantic_sig_over_rti = models["model_2_rti_semantic"]["p_values"].get("semantic_exposure_std", 1) < 0.05
    semantic_sig_over_aioe = models["model_4_aioe_semantic"]["p_values"].get("semantic_exposure_std", 1) < 0.05

    interpretation = {
        "semantic_significant_beyond_rti": semantic_sig_over_rti,
        "semantic_significant_beyond_aioe": semantic_sig_over_aioe,
    }

    if "model_7_full_controls" in models:
        interpretation["semantic_significant_with_controls"] = (
            models["model_7_full_controls"]["p_values"].get("semantic_exposure_std", 1) < 0.05
        )

    print()
    print("INTERPRETATION")
    print("-" * 80)
    print(f"  Semantic exposure significant beyond RTI: {semantic_sig_over_rti}")
    print(f"  Semantic exposure significant beyond AIOE: {semantic_sig_over_aioe}")
    if "semantic_significant_with_controls" in interpretation:
        print(f"  Semantic exposure significant with controls: {interpretation['semantic_significant_with_controls']}")

    # Build output
    output = {
        "version": "0.6.5.3",
        "timestamp": datetime.now().isoformat(),
        "sample": {
            "n_occupations": len(df),
            "n_clusters": df["soc_2digit"].nunique(),
            "employment_coverage_pct": len(df) / 774 * 100,  # vs AIOE baseline
            "years": [2019, 2024],
            "telework_coverage": int(df["teleworkable"].notna().sum()),
            "job_zone_coverage": int(df["job_zone"].notna().sum()),
        },
        "outcome": {
            "variable": "delta_log_emp",
            "description": "log(employment_2024) - log(employment_2019)",
            "mean": float(df["delta_log_emp"].mean()),
            "std": float(df["delta_log_emp"].std()),
        },
        "predictors": {
            "rti": {
                "source": "Acemoglu-Autor 16-element composite from O*NET",
                "elements": "4.A.2.a.4, 4.A.2.b.2, 4.A.4.a.1 (NR-Cog-Analytical), "
                           "4.A.4.a.4, 4.A.4.b.4, 4.A.4.b.5 (NR-Cog-Interpersonal), "
                           "4.C.3.b.7, 4.C.3.b.4, 4.C.3.b.8 (R-Cognitive), "
                           "4.C.3.d.3, 4.A.3.a.3, 4.C.2.d.1.i (R-Manual), "
                           "4.A.3.a.4, 4.C.2.d.1.g, 1.A.2.a.2, 1.A.1.f.1 (NR-Manual-Physical)",
                "mean": float(df["rti"].mean()),
                "std": float(df["rti"].std()),
            },
            "aioe": {
                "source": "Felten, Raj, Seamans (2021)",
                "mean": float(df["aioe"].mean()),
                "std": float(df["aioe"].std()),
            },
            "semantic_exposure": {
                "source": "capability_v1 shock propagated through task-space kernel",
                "mean": float(df["semantic_exposure"].mean()),
                "std": float(df["semantic_exposure"].std()),
            },
        },
        "controls": {
            "telework": {
                "source": "Dingel & Neiman (2020)",
                "mean": float(df["teleworkable"].mean()) if df["teleworkable"].notna().any() else None,
                "coverage": int(df["teleworkable"].notna().sum()),
            },
            "job_zone": {
                "source": "O*NET Job Zones (1-5 scale)",
                "mean": float(df["job_zone"].mean()) if df["job_zone"].notna().any() else None,
                "coverage": int(df["job_zone"].notna().sum()),
            },
        },
        "correlations": {k: float(v) if pd.notna(v) else None for k, v in corr.items()},
        "vif": vif,
        "models": models,
        "incremental_r2": incremental_r2,
        "interpretation": interpretation,
    }

    # Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
