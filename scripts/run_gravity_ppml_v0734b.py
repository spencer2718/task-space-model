#!/usr/bin/env python3
"""
v0.7.3.4b: PPML Robustness Check for Gravity Model

Addresses methodological concern that ln(1+flow) OLS is biased under heteroskedasticity.

Model: Flow_ij ~ exp(β·d(i,j) + μ_i + η_j)

Where:
- Flow_ij = raw count (not logged)
- μ_i = origin fixed effects
- η_j = destination fixed effects
- d(i,j) = distance metric

Uses statsmodels GLM with Poisson family.
"""

import time
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson

from task_space.mobility.io import load_distance_matrix, load_transitions
from task_space.utils.experiments import save_experiment_output, get_output_path


def build_gravity_dataset(
    transitions_df: pd.DataFrame,
    distance_matrices: Dict[str, np.ndarray],
    census_codes: List[int],
) -> pd.DataFrame:
    """Build gravity dataset with origin/destination indices for FE."""
    n_occ = len(census_codes)
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    # Aggregate bilateral flows
    bilateral = transitions_df.groupby(['origin_occ', 'dest_occ']).size()
    bilateral = bilateral.reset_index(name='flow')
    bilateral_dict = {(o, d): f for o, d, f in bilateral.values}

    # Employment proxy: total outflows from origin, total inflows to destination
    emp_origin = transitions_df.groupby('origin_occ').size().to_dict()
    emp_dest = transitions_df.groupby('dest_occ').size().to_dict()

    # Build all pairs
    rows = []
    for i, orig in enumerate(census_codes):
        for j, dest in enumerate(census_codes):
            if orig == dest:
                continue

            flow = bilateral_dict.get((orig, dest), 0)

            row = {
                'origin': orig,
                'dest': dest,
                'origin_idx': i,
                'dest_idx': j,
                'flow': flow,
                'emp_origin': emp_origin.get(orig, 1),
                'emp_dest': emp_dest.get(dest, 1),
            }

            # Add distances
            for metric, d_matrix in distance_matrices.items():
                row[f'd_{metric}'] = d_matrix[i, j]

            rows.append(row)

    df = pd.DataFrame(rows)

    # Add log employment
    df['ln_emp_origin'] = np.log(df['emp_origin'] + 1)
    df['ln_emp_dest'] = np.log(df['emp_dest'] + 1)

    return df


def fit_ppml_with_mass(df: pd.DataFrame, distance_col: str) -> Dict:
    """
    Fit PPML with distance and log-employment controls (like OLS).

    This is a compromise: can't do full FE (computationally prohibitive with 892 dummies),
    but we can include the same mass controls as OLS.

    Model: Flow ~ exp(β₀ + β₁·d + β₂·ln(emp_i) + β₃·ln(emp_j))
    """
    X = df[[distance_col, 'ln_emp_origin', 'ln_emp_dest']].copy()
    X = sm.add_constant(X)
    y = df['flow']

    try:
        model = sm.GLM(y, X, family=Poisson())
        result = model.fit(maxiter=100, method='IRLS')

        beta_distance = result.params[distance_col]
        se_distance = result.bse[distance_col]
        pseudo_r2 = 1 - (result.deviance / result.null_deviance)

        return {
            'beta_distance': float(beta_distance),
            'se': float(se_distance),
            't_stat': float(beta_distance / se_distance),
            'pseudo_r2': float(pseudo_r2),
            'beta_emp_origin': float(result.params['ln_emp_origin']),
            'beta_emp_dest': float(result.params['ln_emp_dest']),
            'converged': result.converged,
            'method': 'ppml_with_mass_controls',
        }

    except Exception as e:
        return {
            'error': str(e),
            'converged': False,
        }


def fit_ppml_no_fe(df: pd.DataFrame, distance_col: str) -> Dict:
    """Fit PPML without fixed effects (for comparison)."""
    X = df[[distance_col]]
    X = sm.add_constant(X)
    y = df['flow']

    try:
        model = sm.GLM(y, X, family=Poisson())
        result = model.fit(maxiter=100, method='IRLS')

        beta_distance = result.params[distance_col]
        se_distance = result.bse[distance_col]
        pseudo_r2 = 1 - (result.deviance / result.null_deviance)

        return {
            'beta_distance': float(beta_distance),
            'se': float(se_distance),
            'pseudo_r2': float(pseudo_r2),
            'converged': result.converged,
        }

    except Exception as e:
        return {'error': str(e), 'converged': False}


def main():
    import json  # Only needed for reading OLS results

    print("=" * 70)
    print("v0.7.3.4b: PPML Robustness Check for Gravity Model")
    print("=" * 70)

    start_time = time.time()

    # Load distance matrices
    print("\n1. Loading distance matrices...")
    metrics = ['wasserstein', 'cosine_embed', 'cosine_onet', 'euclidean_dwa']
    distance_matrices = {}
    census_codes = None

    for metric in metrics:
        d, codes = load_distance_matrix(kind=metric)
        distance_matrices[metric] = d
        if census_codes is None:
            census_codes = codes
        print(f"  {metric}: shape={d.shape}")

    n_occ = len(census_codes)
    print(f"  Total occupations: {n_occ}")
    print(f"  FE dimensions: {n_occ-1} origin + {n_occ-1} dest = {2*(n_occ-1)} dummies")

    # Load transitions
    print("\n2. Loading CPS transitions...")
    transitions = load_transitions()
    valid = set(census_codes)
    mask = transitions['origin_occ'].isin(valid) & transitions['dest_occ'].isin(valid)
    trans = transitions[mask].copy()
    print(f"  Transitions: {len(trans):,}")

    # Build dataset
    print("\n3. Building gravity dataset...")
    df = build_gravity_dataset(trans, distance_matrices, census_codes)
    print(f"  Total pairs: {len(df):,}")
    print(f"  Positive flow: {(df['flow'] > 0).sum():,}")
    print(f"  Total flow: {df['flow'].sum():,}")

    # Load OLS results for comparison
    print("\n4. Loading OLS results from v0.7.3.4...")
    ols_path = get_output_path("gravity_model_v0734")
    with open(ols_path) as f:
        ols_results = json.load(f)

    # Fit PPML models
    print("\n5. Fitting PPML models...")
    print("   Note: Full FE (892 dummies) is computationally prohibitive.")
    print("   Using PPML with mass controls (same as OLS) for comparability.")

    results_mass = {}
    results_no_mass = {}

    for metric in metrics:
        print(f"\n  --- {metric} ---")

        # PPML without controls (distance only)
        print("    Fitting PPML (distance only)...")
        t0 = time.time()
        res_no_mass = fit_ppml_no_fe(df, f'd_{metric}')
        print(f"    Done in {time.time()-t0:.1f}s")
        results_no_mass[metric] = res_no_mass

        if res_no_mass.get('converged', False):
            print(f"    β = {res_no_mass['beta_distance']:.4f}, pseudo-R² = {res_no_mass['pseudo_r2']:.4f}")

        # PPML with mass controls (comparable to OLS)
        print("    Fitting PPML (with mass controls)...")
        t0 = time.time()
        res_mass = fit_ppml_with_mass(df, f'd_{metric}')
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")
        results_mass[metric] = res_mass

        if res_mass.get('converged', False):
            print(f"    β = {res_mass['beta_distance']:.4f} (SE = {res_mass['se']:.4f})")
            print(f"    pseudo-R² = {res_mass['pseudo_r2']:.4f}")
        else:
            print(f"    ERROR: {res_mass.get('error', 'Did not converge')}")

    # Compare to OLS
    print("\n6. Comparing to OLS results...")

    # Check rankings by beta magnitude (more negative = stronger effect)
    ols_ranking = sorted(metrics, key=lambda m: ols_results['models'][m]['beta_distance'])
    ppml_ranking = sorted(
        [m for m in metrics if results_mass[m].get('converged', False)],
        key=lambda m: results_mass[m]['beta_distance']
    )

    print(f"  OLS ranking (by β, most negative first): {ols_ranking}")
    print(f"  PPML ranking (by β, most negative first): {ppml_ranking}")

    rankings_preserved = (ols_ranking[:2] == ppml_ranking[:2]) if len(ppml_ranking) >= 2 else False

    # Check signs
    signs_preserved = all(
        results_mass[m].get('beta_distance', 0) < 0
        for m in metrics
        if results_mass[m].get('converged', False)
    )

    print(f"\n  Rankings preserved (top-2): {rankings_preserved}")
    print(f"  All signs negative: {signs_preserved}")

    # Compute magnitude ratios (PPML / OLS)
    print("\n  Beta comparison (PPML vs OLS):")
    mag_ratios = []
    for m in metrics:
        if results_mass[m].get('converged', False):
            ols_beta = ols_results['models'][m]['beta_distance']
            ppml_beta = results_mass[m]['beta_distance']
            ratio = ppml_beta / ols_beta if ols_beta != 0 else float('inf')
            mag_ratios.append(ratio)
            print(f"    {m}: OLS β={ols_beta:.4f}, PPML β={ppml_beta:.4f}, ratio={ratio:.2f}")

    avg_mag_ratio = np.mean(mag_ratios) if mag_ratios else None

    # Interpretation
    if signs_preserved and rankings_preserved:
        interpretation = "PPML confirms OLS findings: rankings and signs preserved. Distance effect robust to heteroskedasticity correction."
    elif signs_preserved:
        interpretation = "PPML signs match OLS (all negative), but rankings differ slightly. Results directionally consistent."
    else:
        interpretation = "PPML results differ substantially from OLS. Heteroskedasticity may be affecting OLS estimates."

    print(f"\n  Interpretation: {interpretation}")

    # Build output
    output = {
        "version": "0.7.3.4b",
        "method": "PPML with mass controls (FE computationally prohibitive)",
        "note": "Full origin+destination FE (892 dummies) caused memory issues. Using log-employment controls for comparability with OLS.",
        "sample": {
            "n_pairs": len(df),
            "n_occupations": n_occ,
        },
        "models": {},
        "models_distance_only": {},
        "comparison_to_ols": {
            "ols_ranking": ols_ranking,
            "ppml_ranking": ppml_ranking,
            "rankings_preserved": rankings_preserved,
            "signs_preserved": signs_preserved,
            "avg_magnitude_ratio": float(avg_mag_ratio) if avg_mag_ratio else None,
        },
        "interpretation": interpretation,
    }

    for metric in metrics:
        output["models"][metric] = results_mass[metric]
        output["models_distance_only"][metric] = results_no_mass[metric]

    # Save
    output_path = save_experiment_output("gravity_ppml_v0734b", output)
    print(f"\nSaved: {output_path}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: PPML with Mass Controls")
    print("=" * 70)
    print(f"{'Metric':<18} {'β_dist':>10} {'SE':>8} {'Pseudo-R²':>12} {'Converged':>10}")
    print("-" * 70)
    for metric in metrics:
        r = results_mass[metric]
        if r.get('converged', False):
            se_str = f"{r['se']:.4f}" if r.get('se') else "—"
            print(f"{metric:<18} {r['beta_distance']:>10.4f} {se_str:>8} {r['pseudo_r2']:>12.4f} {'Yes':>10}")
        else:
            print(f"{metric:<18} {'—':>10} {'—':>8} {'—':>12} {'No':>10}")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
