#!/usr/bin/env python3
"""
v0.7.3.4: Gravity Model for Bilateral Occupation Flows

Implements gravity model to assess task distance explanatory power:
    ln(Flow_ij + 1) = α + β₁·ln(Emp_i + 1) + β₂·ln(Emp_j + 1) + β₃·d(i,j) + ε

Compares embedding-based vs O*NET-based distance metrics.

Engages Cortes-Gallipoli (2018) finding that task-specific costs account
for "no more than 15%" of switching costs.

Sample: All 447 × 446 = 199,362 occupation pairs (including zeros)
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from task_space.mobility.io import load_distance_matrix, load_transitions
from task_space.utils.experiments import save_experiment_output


@dataclass
class GravityResult:
    """Result container for gravity regression."""
    metric: str
    beta_const: float
    beta_emp_origin: float
    beta_emp_origin_se: float
    beta_emp_dest: float
    beta_emp_dest_se: float
    beta_distance: float
    beta_distance_se: float
    t_stat_distance: float
    r2: float
    n_obs: int


def build_gravity_dataset(
    transitions_df: pd.DataFrame,
    distance_matrices: Dict[str, np.ndarray],
    census_codes: List[int],
) -> pd.DataFrame:
    """
    Build gravity model dataset with all occupation pairs.

    Returns DataFrame with columns:
        - origin, dest: occupation codes
        - flow: count of transitions (0 for unobserved pairs)
        - ln_flow: ln(flow + 1)
        - emp_origin, emp_dest: employment proxies
        - ln_emp_origin, ln_emp_dest: log employment
        - d_<metric>: distance for each metric
    """
    n_occ = len(census_codes)
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    # Aggregate bilateral flows
    bilateral = transitions_df.groupby(['origin_occ', 'dest_occ']).size()
    bilateral = bilateral.reset_index(name='flow')

    # Employment proxy: total outflows from origin, total inflows to destination
    emp_origin = transitions_df.groupby('origin_occ').size()
    emp_dest = transitions_df.groupby('dest_occ').size()

    # Fill missing with small value (1) to avoid log(0)
    emp_origin_dict = emp_origin.to_dict()
    emp_dest_dict = emp_dest.to_dict()

    # Build all pairs
    rows = []
    for i, orig in enumerate(census_codes):
        for j, dest in enumerate(census_codes):
            if orig == dest:
                continue

            # Get flow
            flow_mask = (bilateral['origin_occ'] == orig) & (bilateral['dest_occ'] == dest)
            if flow_mask.any():
                flow = bilateral.loc[flow_mask, 'flow'].values[0]
            else:
                flow = 0

            # Get employment
            e_orig = emp_origin_dict.get(orig, 1)
            e_dest = emp_dest_dict.get(dest, 1)

            row = {
                'origin': orig,
                'dest': dest,
                'origin_idx': i,
                'dest_idx': j,
                'flow': flow,
                'emp_origin': e_orig,
                'emp_dest': e_dest,
            }

            # Add distances
            for metric, d_matrix in distance_matrices.items():
                row[f'd_{metric}'] = d_matrix[i, j]

            rows.append(row)

    df = pd.DataFrame(rows)

    # Log transforms
    df['ln_flow'] = np.log(df['flow'] + 1)
    df['ln_emp_origin'] = np.log(df['emp_origin'] + 1)
    df['ln_emp_dest'] = np.log(df['emp_dest'] + 1)

    return df


def fit_gravity_model(
    df: pd.DataFrame,
    distance_col: str,
) -> GravityResult:
    """
    Fit gravity model with OLS.

    Model: ln(Flow + 1) = α + β₁·ln(Emp_i + 1) + β₂·ln(Emp_j + 1) + β₃·d(i,j) + ε
    """
    X = df[['ln_emp_origin', 'ln_emp_dest', distance_col]].copy()
    X = sm.add_constant(X)
    y = df['ln_flow']

    model = sm.OLS(y, X).fit()

    metric = distance_col.replace('d_', '')

    return GravityResult(
        metric=metric,
        beta_const=model.params['const'],
        beta_emp_origin=model.params['ln_emp_origin'],
        beta_emp_origin_se=model.bse['ln_emp_origin'],
        beta_emp_dest=model.params['ln_emp_dest'],
        beta_emp_dest_se=model.bse['ln_emp_dest'],
        beta_distance=model.params[distance_col],
        beta_distance_se=model.bse[distance_col],
        t_stat_distance=model.tvalues[distance_col],
        r2=model.rsquared,
        n_obs=len(df),
    )


def fit_mass_only_model(df: pd.DataFrame) -> Tuple[float, float, float, float, float]:
    """
    Fit model with only mass terms (no distance).

    Model: ln(Flow + 1) = α + β₁·ln(Emp_i + 1) + β₂·ln(Emp_j + 1) + ε

    Returns: (r2, beta_emp_origin, beta_emp_origin_se, beta_emp_dest, beta_emp_dest_se)
    """
    X = df[['ln_emp_origin', 'ln_emp_dest']].copy()
    X = sm.add_constant(X)
    y = df['ln_flow']

    model = sm.OLS(y, X).fit()

    return (
        model.rsquared,
        model.params['ln_emp_origin'],
        model.bse['ln_emp_origin'],
        model.params['ln_emp_dest'],
        model.bse['ln_emp_dest'],
    )


def main():
    print("=" * 70)
    print("v0.7.3.4: Gravity Model for Bilateral Occupation Flows")
    print("=" * 70)

    start_time = time.time()

    # Load distance matrices
    print("\n1. Loading distance matrices...")
    metrics = ['cosine_embed', 'wasserstein', 'euclidean_dwa', 'cosine_onet']
    distance_matrices = {}
    census_codes = None

    for metric in metrics:
        d, codes = load_distance_matrix(kind=metric)
        distance_matrices[metric] = d
        if census_codes is None:
            census_codes = codes
        print(f"  {metric}: shape={d.shape}, range=[{d.min():.4f}, {d.max():.4f}]")

    # Load transitions
    print("\n2. Loading CPS transitions...")
    transitions = load_transitions()

    # Filter to valid codes
    valid = set(census_codes)
    mask = transitions['origin_occ'].isin(valid) & transitions['dest_occ'].isin(valid)
    trans = transitions[mask].copy()
    print(f"  Transitions: {len(trans):,}")

    # Build gravity dataset
    print("\n3. Building gravity dataset...")
    df = build_gravity_dataset(trans, distance_matrices, census_codes)

    n_positive = (df['flow'] > 0).sum()
    n_zero = (df['flow'] == 0).sum()
    print(f"  Total pairs: {len(df):,}")
    print(f"  Positive flow: {n_positive:,} ({100*n_positive/len(df):.1f}%)")
    print(f"  Zero flow: {n_zero:,} ({100*n_zero/len(df):.1f}%)")

    # Fit mass-only model
    print("\n4. Fitting mass-only model...")
    r2_mass, beta_emp_o, se_emp_o, beta_emp_d, se_emp_d = fit_mass_only_model(df)
    print(f"  R² (mass only): {r2_mass:.4f} ({100*r2_mass:.2f}%)")
    print(f"  β_emp_origin = {beta_emp_o:.4f} (SE = {se_emp_o:.4f})")
    print(f"  β_emp_dest = {beta_emp_d:.4f} (SE = {se_emp_d:.4f})")

    # Fit models for each distance metric
    print("\n5. Fitting gravity models...")
    results = {}

    for metric in metrics:
        result = fit_gravity_model(df, f'd_{metric}')
        results[metric] = result

        partial_r2 = result.r2 - r2_mass

        print(f"\n  --- {metric} ---")
        print(f"  β_distance = {result.beta_distance:.4f} (SE = {result.beta_distance_se:.4f}, t = {result.t_stat_distance:.2f})")
        print(f"  R² (full): {result.r2:.4f} ({100*result.r2:.2f}%)")
        print(f"  Partial R² (distance): {partial_r2:.4f} ({100*partial_r2:.2f}%)")

    # Check stop conditions
    print("\n6. Checking stop conditions...")
    stop = False
    for metric, result in results.items():
        if result.beta_distance > 0:
            print(f"  WARNING: β_distance > 0 for {metric}! (β = {result.beta_distance:.4f})")
            stop = True

    if stop:
        print("\n  STOP: β_distance > 0 detected. This contradicts theory (greater distance should mean fewer transitions).")
        # Don't stop - document and continue
    else:
        print("  OK: All β_distance < 0 (distance reduces flow, as expected)")

    # Compute comparisons
    print("\n7. Computing comparisons...")

    embedding_metrics = ['cosine_embed', 'wasserstein']
    onet_metrics = ['euclidean_dwa', 'cosine_onet']

    embedding_partial_r2 = np.mean([results[m].r2 - r2_mass for m in embedding_metrics])
    onet_partial_r2 = np.mean([results[m].r2 - r2_mass for m in onet_metrics])

    print(f"  Embedding-based avg partial R²: {embedding_partial_r2:.4f} ({100*embedding_partial_r2:.2f}%)")
    print(f"  O*NET-based avg partial R²: {onet_partial_r2:.4f} ({100*onet_partial_r2:.2f}%)")

    if onet_partial_r2 > 0:
        ratio = embedding_partial_r2 / onet_partial_r2
        print(f"  Ratio (embedding / O*NET): {ratio:.2f}×")
    else:
        ratio = float('inf')
        print(f"  Ratio: undefined (O*NET partial R² ≈ 0)")

    # Interpretation relative to C-G benchmark
    print("\n8. Interpretation...")
    best_partial_r2 = max(r.r2 - r2_mass for r in results.values())
    print(f"  Best partial R² (distance): {best_partial_r2:.4f} ({100*best_partial_r2:.2f}%)")
    print(f"  Cortes-Gallipoli benchmark: 15%")

    if best_partial_r2 > 0.15:
        interpretation = f"Task distance explains {100*best_partial_r2:.1f}% of variance, exceeding C-G's 15% benchmark. Semantic embeddings capture substantial switching cost structure."
    elif best_partial_r2 > 0.10:
        interpretation = f"Task distance explains {100*best_partial_r2:.1f}% of variance, approaching C-G's 15% benchmark. Consistent with task costs being a meaningful but not dominant factor."
    else:
        interpretation = f"Task distance explains {100*best_partial_r2:.1f}% of variance, below C-G's 15% benchmark. Consistent with C-G finding that task-specific costs are a modest share of switching costs."

    print(f"  {interpretation}")

    # Build output
    output = {
        "version": "0.7.3.4",
        "sample": {
            "n_pairs_total": int(len(df)),
            "n_pairs_positive_flow": int(n_positive),
            "n_pairs_zero_flow": int(n_zero),
            "total_transitions": int(len(trans)),
            "n_occupations": int(len(census_codes)),
        },
        "mass_only_model": {
            "r2": float(r2_mass),
            "beta_emp_origin": float(beta_emp_o),
            "beta_emp_origin_se": float(se_emp_o),
            "beta_emp_dest": float(beta_emp_d),
            "beta_emp_dest_se": float(se_emp_d),
        },
        "models": {},
        "comparison": {
            "embedding_avg_partial_r2": float(embedding_partial_r2),
            "onet_avg_partial_r2": float(onet_partial_r2),
            "ratio": float(ratio) if ratio != float('inf') else None,
        },
        "cortes_gallipoli_benchmark": "15%",
        "interpretation": interpretation,
    }

    for metric in metrics:
        r = results[metric]
        partial_r2 = r.r2 - r2_mass
        output["models"][metric] = {
            "beta_distance": float(r.beta_distance),
            "beta_distance_se": float(r.beta_distance_se),
            "t_stat": float(r.t_stat_distance),
            "r2_full": float(r.r2),
            "partial_r2_distance": float(partial_r2),
        }

    # Save results
    output_path = save_experiment_output("gravity_model_v0734", output)
    print(f"\nSaved: {output_path}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Metric':<20} {'β_dist':>10} {'SE':>8} {'t':>8} {'R²_full':>10} {'Partial R²':>12}")
    print("-" * 70)
    for metric in metrics:
        r = results[metric]
        partial = r.r2 - r2_mass
        print(f"{metric:<20} {r.beta_distance:>10.4f} {r.beta_distance_se:>8.4f} {r.t_stat_distance:>8.1f} {r.r2:>10.4f} {partial:>12.4f}")
    print("-" * 70)
    print(f"Mass-only R²: {r2_mass:.4f}")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
