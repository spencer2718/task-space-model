#!/usr/bin/env python3
"""
v0.7.7.3: Pre/Post COVID Re-estimation with Centroid Specification

Re-runs the pre/post COVID period estimation from v0.7.4.1 using the
cosine_embed (centroid) distance matrix instead of Wasserstein. This produces
a complete parameter set (α, SE, γ, SE, LL, pseudo-R²) for both periods,
enabling Table 9 to use centroid values and removing the Wasserstein footnote.

Methodology matches v0741 exactly:
- Same sample split (pre: year < 2020, post: year >= 2022)
- Same seeds (pre=42, post=43, combined=44)
- Same choice set construction (n_alternatives=10)
- Same clustered standard errors
- Same structural break test (LR test on interacted model)

Stop-and-return:
- Pre-COVID α not within ±0.05 of 7.394 → seed/filtering mismatch
- Structural break p < 0.05 → stability fails under centroid
"""

import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.discrete.conditional_models import ConditionalLogit

from task_space.mobility.io import load_transitions, load_distance_matrix
from task_space.utils.experiments import save_experiment_output


@dataclass
class PeriodEstimate:
    period: str
    n_transitions: int
    alpha: float
    se_alpha: float
    gamma: float
    se_gamma: float
    ll: float
    pseudo_r2: float
    converged: bool


def build_choice_dataset_with_period(
    transitions_df: pd.DataFrame,
    d_sem_matrix: np.ndarray,
    d_inst_matrix: np.ndarray,
    occ_codes: List[int],
    n_alternatives: int = 10,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build choice dataset — matches v0741 implementation exactly."""
    np.random.seed(random_seed)

    occ_to_idx = {occ: i for i, occ in enumerate(occ_codes)}
    all_occs = set(occ_codes)

    rows = []
    transition_id = 0

    for _, row in transitions_df.iterrows():
        origin = int(row["origin_occ"])
        dest = int(row["dest_occ"])
        year = int(row["year"])

        if origin not in occ_to_idx or dest not in occ_to_idx:
            continue

        origin_idx = occ_to_idx[origin]
        dest_idx = occ_to_idx[dest]

        post_covid = 1 if year >= 2022 else 0

        available = list(all_occs - {dest})
        if len(available) < n_alternatives:
            continue

        sampled_alts = np.random.choice(available, size=n_alternatives, replace=False)

        neg_d_sem = -d_sem_matrix[origin_idx, dest_idx]
        neg_d_inst = -d_inst_matrix[origin_idx, dest_idx]

        rows.append({
            "transition_id": transition_id,
            "origin_occ": origin,
            "occ": dest,
            "chosen": 1,
            "neg_d_sem": neg_d_sem,
            "neg_d_inst": neg_d_inst,
            "post_covid": post_covid,
            "neg_d_sem_x_post": neg_d_sem * post_covid,
            "neg_d_inst_x_post": neg_d_inst * post_covid,
        })

        for alt in sampled_alts:
            alt_idx = occ_to_idx[alt]
            neg_d_sem_alt = -d_sem_matrix[origin_idx, alt_idx]
            neg_d_inst_alt = -d_inst_matrix[origin_idx, alt_idx]

            rows.append({
                "transition_id": transition_id,
                "origin_occ": origin,
                "occ": alt,
                "chosen": 0,
                "neg_d_sem": neg_d_sem_alt,
                "neg_d_inst": neg_d_inst_alt,
                "post_covid": post_covid,
                "neg_d_sem_x_post": neg_d_sem_alt * post_covid,
                "neg_d_inst_x_post": neg_d_inst_alt * post_covid,
            })

        transition_id += 1

    return pd.DataFrame(rows)


def fit_conditional_logit_clustered(
    choice_df: pd.DataFrame,
    covariate_cols: List[str],
    cluster_col: str = "origin_occ",
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """Fit conditional logit with clustered SEs — matches v0741."""
    endog = choice_df["chosen"].values
    exog = choice_df[covariate_cols].values
    groups = choice_df["transition_id"].values
    clusters = choice_df[cluster_col].values

    model = ConditionalLogit(endog, exog, groups=groups)

    try:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": clusters}, disp=False)
        converged = result.mle_retvals.get("converged", True) if hasattr(result, "mle_retvals") else True
    except Exception as e:
        print(f"  Warning: Clustered SE failed ({e}), using robust SE")
        result = model.fit(cov_type="HC0", disp=False)
        converged = result.mle_retvals.get("converged", True) if hasattr(result, "mle_retvals") else True

    return result.params, result.bse, result.llf, converged


def compute_pseudo_r2(ll_model: float, n_transitions: int, n_alternatives: int) -> float:
    ll_null = n_transitions * np.log(1.0 / n_alternatives)
    if ll_null == 0:
        return 0.0
    return 1.0 - (ll_model / ll_null)


def estimate_period(choice_df: pd.DataFrame, period: str) -> PeriodEstimate:
    covariate_cols = ["neg_d_sem", "neg_d_inst"]
    coefs, ses, ll, converged = fit_conditional_logit_clustered(choice_df, covariate_cols)

    n_transitions = choice_df["transition_id"].nunique()
    n_alternatives = len(choice_df) // n_transitions
    pseudo_r2 = compute_pseudo_r2(ll, n_transitions, n_alternatives)

    return PeriodEstimate(
        period=period,
        n_transitions=n_transitions,
        alpha=float(coefs[0]),
        se_alpha=float(ses[0]),
        gamma=float(coefs[1]),
        se_gamma=float(ses[1]),
        ll=float(ll),
        pseudo_r2=float(pseudo_r2),
        converged=converged,
    )


def main():
    print("=" * 70)
    print("v0.7.7.3: Pre/Post COVID — Centroid Specification")
    print("=" * 70)
    start_time = time.time()

    # Load distance matrices
    print("\nLoading distance matrices...")
    d_sem, census_codes = load_distance_matrix("cosine_embed")
    d_inst, _ = load_distance_matrix("institutional")
    print(f"  Semantic (cosine_embed): {d_sem.shape}")
    print(f"  Institutional: {d_inst.shape}")

    # Load and split transitions
    print("\nLoading transitions...")
    df = load_transitions()
    print(f"  Total: {len(df):,}, year range: {df['year'].min()}-{df['year'].max()}")

    pre_covid = df[df["year"] < 2020].copy()
    post_covid = df[df["year"] >= 2022].copy()

    valid_codes = set(census_codes)
    pre_covid = pre_covid[pre_covid["origin_occ"].isin(valid_codes) & pre_covid["dest_occ"].isin(valid_codes)].copy()
    post_covid = post_covid[post_covid["origin_occ"].isin(valid_codes) & post_covid["dest_occ"].isin(valid_codes)].copy()
    combined = pd.concat([pre_covid, post_covid], ignore_index=True)

    print(f"  Pre-COVID (valid): {len(pre_covid):,}")
    print(f"  Post-COVID (valid): {len(post_covid):,}")

    # Period-specific estimation
    print("\n--- Period-Specific Estimation ---")

    print("\n  Building pre-COVID choice set (seed=42)...")
    choice_pre = build_choice_dataset_with_period(pre_covid, d_sem, d_inst, census_codes, random_seed=42)
    print(f"    {choice_pre['transition_id'].nunique():,} transitions")

    print("  Building post-COVID choice set (seed=43)...")
    choice_post = build_choice_dataset_with_period(post_covid, d_sem, d_inst, census_codes, random_seed=43)
    print(f"    {choice_post['transition_id'].nunique():,} transitions")

    print("\n  Estimating pre-COVID...")
    result_pre = estimate_period(choice_pre, "pre_covid")
    print(f"    α = {result_pre.alpha:.4f} (SE = {result_pre.se_alpha:.4f})")
    print(f"    γ = {result_pre.gamma:.4f} (SE = {result_pre.se_gamma:.4f})")
    print(f"    LL = {result_pre.ll:.2f}, pseudo-R² = {result_pre.pseudo_r2:.4f} ({result_pre.pseudo_r2*100:.2f}%)")

    print("\n  Estimating post-COVID...")
    result_post = estimate_period(choice_post, "post_covid")
    print(f"    α = {result_post.alpha:.4f} (SE = {result_post.se_alpha:.4f})")
    print(f"    γ = {result_post.gamma:.4f} (SE = {result_post.se_gamma:.4f})")
    print(f"    LL = {result_post.ll:.2f}, pseudo-R² = {result_post.pseudo_r2:.4f} ({result_post.pseudo_r2*100:.2f}%)")

    alpha_pct_change = 100 * (result_post.alpha - result_pre.alpha) / abs(result_pre.alpha)
    print(f"\n  α change: {result_pre.alpha:.4f} → {result_post.alpha:.4f} ({alpha_pct_change:+.2f}%)")

    # Stop-and-return: pre-COVID α reproduction
    if abs(result_pre.alpha - 7.394) > 0.05:
        print(f"\nSTOP: Pre-COVID α = {result_pre.alpha:.4f}, expected ~7.394 ±0.05. Seed/filtering mismatch.")
        sys.exit(1)
    print(f"  ✓ Pre-COVID α reproduces ({result_pre.alpha:.4f} vs expected ~7.394)")

    # Structural break test
    print("\n--- Structural Break Test ---")
    print("  Building combined choice set (seed=44)...")
    choice_combined = build_choice_dataset_with_period(combined, d_sem, d_inst, census_codes, random_seed=44)
    print(f"    {choice_combined['transition_id'].nunique():,} transitions")

    # Pooled (restricted)
    coefs_pooled, ses_pooled, ll_pooled, _ = fit_conditional_logit_clustered(
        choice_combined, ["neg_d_sem", "neg_d_inst"]
    )
    n_pooled = choice_combined["transition_id"].nunique()
    n_alts_pooled = len(choice_combined) // n_pooled
    pseudo_r2_pooled = compute_pseudo_r2(ll_pooled, n_pooled, n_alts_pooled)

    # Interacted (unrestricted)
    coefs_inter, ses_inter, ll_inter, _ = fit_conditional_logit_clustered(
        choice_combined, ["neg_d_sem", "neg_d_inst", "neg_d_sem_x_post", "neg_d_inst_x_post"]
    )

    lr_stat = 2 * (ll_inter - ll_pooled)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=2)

    print(f"  LR stat = {lr_stat:.4f}, df = 2, p = {p_value:.4f}")
    print(f"  δ_α = {coefs_inter[2]:.4f} (SE = {ses_inter[2]:.4f})")
    print(f"  δ_γ = {coefs_inter[3]:.4f} (SE = {ses_inter[3]:.4f})")

    # Stop-and-return: structural break
    if p_value < 0.05:
        print(f"\nSTOP: Structural break p = {p_value:.4f} < 0.05. Stability fails under centroid.")
        print("Saving partial output before stopping...")
        partial = {
            "version": "0.7.7.3",
            "status": "STOPPED_STRUCTURAL_BREAK",
            "specification": "cosine_embed (centroid)",
            "pre_covid": {"alpha": result_pre.alpha, "se_alpha": result_pre.se_alpha,
                          "gamma": result_pre.gamma, "se_gamma": result_pre.se_gamma,
                          "ll": result_pre.ll, "pseudo_r2": result_pre.pseudo_r2,
                          "n_transitions": result_pre.n_transitions},
            "post_covid": {"alpha": result_post.alpha, "se_alpha": result_post.se_alpha,
                           "gamma": result_post.gamma, "se_gamma": result_post.se_gamma,
                           "ll": result_post.ll, "pseudo_r2": result_post.pseudo_r2,
                           "n_transitions": result_post.n_transitions},
            "structural_break_test": {"chi2": float(lr_stat), "df": 2, "p_value": float(p_value)},
            "alpha_pct_change": float(alpha_pct_change),
        }
        save_experiment_output("covid_centroid_v0773", partial)
        sys.exit(1)

    print(f"  ✓ No structural break (p = {p_value:.4f})")

    # Build output
    output = {
        "version": "0.7.7.3",
        "specification": "cosine_embed (centroid)",
        "pre_covid": {
            "alpha": float(result_pre.alpha),
            "se_alpha": float(result_pre.se_alpha),
            "gamma": float(result_pre.gamma),
            "se_gamma": float(result_pre.se_gamma),
            "ll": float(result_pre.ll),
            "pseudo_r2": float(result_pre.pseudo_r2),
            "n_transitions": int(result_pre.n_transitions),
        },
        "post_covid": {
            "alpha": float(result_post.alpha),
            "se_alpha": float(result_post.se_alpha),
            "gamma": float(result_post.gamma),
            "se_gamma": float(result_post.se_gamma),
            "ll": float(result_post.ll),
            "pseudo_r2": float(result_post.pseudo_r2),
            "n_transitions": int(result_post.n_transitions),
        },
        "pooled": {
            "alpha": float(coefs_pooled[0]),
            "se_alpha": float(ses_pooled[0]),
            "gamma": float(coefs_pooled[1]),
            "se_gamma": float(ses_pooled[1]),
            "ll": float(ll_pooled),
            "pseudo_r2": float(pseudo_r2_pooled),
            "n_transitions": int(n_pooled),
        },
        "structural_break_test": {
            "chi2": float(lr_stat),
            "df": 2,
            "p_value": float(p_value),
            "delta_alpha": float(coefs_inter[2]),
            "se_delta_alpha": float(ses_inter[2]),
            "delta_gamma": float(coefs_inter[3]),
            "se_delta_gamma": float(ses_inter[3]),
        },
        "alpha_pct_change": float(alpha_pct_change),
    }

    output_path = save_experiment_output("covid_centroid_v0773", output)
    print(f"\nSaved: {output_path}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Centroid COVID Re-estimation")
    print("=" * 70)
    print(f"{'Period':<25} {'α':>8} {'SE':>8} {'γ':>8} {'SE':>8} {'R²':>8}")
    print("-" * 70)
    print(f"{'Pre-COVID (2015-2019)':<25} {result_pre.alpha:>8.3f} {result_pre.se_alpha:>8.3f} "
          f"{result_pre.gamma:>8.3f} {result_pre.se_gamma:>8.3f} {result_pre.pseudo_r2*100:>7.1f}%")
    print(f"{'Post-COVID (2022-2024)':<25} {result_post.alpha:>8.3f} {result_post.se_alpha:>8.3f} "
          f"{result_post.gamma:>8.3f} {result_post.se_gamma:>8.3f} {result_post.pseudo_r2*100:>7.1f}%")
    print("-" * 70)
    print(f"α change: {alpha_pct_change:+.2f}%")
    print(f"Structural break: χ²(2) = {lr_stat:.2f}, p = {p_value:.4f}")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
