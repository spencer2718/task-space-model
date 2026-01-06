#!/usr/bin/env python3
"""
v0.7.4.1: Pre/Post COVID Geometry Comparison (Full)

Implements complete analysis of whether COVID altered task-similarity geometry.

Steps:
  1. Sample Split - Load transitions, split on interview year
  2. Period-Specific Estimation - Conditional logit for each period
  3. Structural Break Test - LR test for coefficient equality
  4. Remote Work Heterogeneity - Dingel-Neiman interaction analysis
  5. Robustness - cosine_embed, exclude 2022

Model:
    U_kj = α·(-d_sem_ij) + γ·(-d_inst_ij)

Evidence Class: E1 (Score Robustness) - EXPLORATORY per MS3.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.discrete.conditional_models import ConditionalLogit

from task_space.mobility.io import (
    load_transitions,
    load_wasserstein_census,
    load_institutional_census,
    load_distance_matrix,
)
from task_space.utils.experiments import save_experiment_output


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PeriodEstimate:
    """Conditional logit results for a single period."""
    period: str
    n_transitions: int
    alpha: float
    se_alpha: float
    gamma: float
    se_gamma: float
    ll: float
    pseudo_r2: float
    converged: bool


@dataclass
class StructuralBreakResult:
    """LR test results for structural break."""
    lr_stat: float
    df: int
    p_value: float
    delta_alpha: float
    se_delta_alpha: float
    delta_gamma: float
    se_delta_gamma: float


# =============================================================================
# Dingel-Neiman Teleworkable Loading
# =============================================================================

def load_dingel_neiman_teleworkable(
    dn_path: str = "data/external/dingel_neiman/onet_teleworkable_blscodes.csv",
    crosswalk_path: str = ".cache/artifacts/v1/mobility/onet_to_census_improved.csv",
) -> Tuple[Dict[int, float], float, int]:
    """
    Load Dingel-Neiman teleworkable scores and crosswalk to Census codes.

    Args:
        dn_path: Path to Dingel-Neiman CSV
        crosswalk_path: Path to O*NET-Census crosswalk

    Returns:
        Tuple of:
            - Dict mapping Census code -> teleworkable score (0-1)
            - Coverage rate (fraction of Census codes with scores)
            - Number of matched codes
    """
    # Load Dingel-Neiman data
    dn = pd.read_csv(dn_path)
    dn["soc_6digit"] = dn["OCC_CODE"].str.strip()

    # Load crosswalk
    xwalk = pd.read_csv(crosswalk_path)
    xwalk = xwalk[xwalk["matched"] == True].copy()

    # Merge on SOC 6-digit code
    merged = xwalk.merge(dn[["soc_6digit", "teleworkable"]], on="soc_6digit", how="left")

    # Aggregate to Census level (average when multiple SOC -> one Census)
    census_scores = merged.groupby("census_2010")["teleworkable"].mean()

    # Build dict
    telework_dict = {}
    for census_code, score in census_scores.items():
        if pd.notna(score):
            telework_dict[int(census_code)] = float(score)

    # Coverage
    n_census_total = xwalk["census_2010"].nunique()
    n_matched = len(telework_dict)
    coverage = n_matched / n_census_total if n_census_total > 0 else 0

    return telework_dict, coverage, n_matched


# =============================================================================
# Choice Dataset Construction
# =============================================================================

def build_choice_dataset_with_period(
    transitions_df: pd.DataFrame,
    d_sem_matrix: np.ndarray,
    d_inst_matrix: np.ndarray,
    occ_codes: List[int],
    n_alternatives: int = 10,
    random_seed: int = 42,
    telework_dict: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """
    Build choice dataset for conditional logit estimation.

    Includes:
    - origin_occ column for clustering
    - post_covid indicator
    - Interaction terms for structural break test
    - Teleworkable score and interactions (if provided)
    """
    np.random.seed(random_seed)

    occ_to_idx = {occ: i for i, occ in enumerate(occ_codes)}
    all_occs = set(occ_codes)

    rows = []
    transition_id = 0

    for _, row in transitions_df.iterrows():
        origin = int(row["origin_occ"])
        dest = int(row["dest_occ"])
        year = int(row["year"])

        # Skip if codes not in distance matrix
        if origin not in occ_to_idx or dest not in occ_to_idx:
            continue

        origin_idx = occ_to_idx[origin]
        dest_idx = occ_to_idx[dest]

        # Determine period
        post_covid = 1 if year >= 2022 else 0

        # Sample alternatives (excluding chosen destination)
        available = list(all_occs - {dest})
        if len(available) < n_alternatives:
            continue

        sampled_alts = np.random.choice(available, size=n_alternatives, replace=False)

        # Add chosen destination
        neg_d_sem = -d_sem_matrix[origin_idx, dest_idx]
        neg_d_inst = -d_inst_matrix[origin_idx, dest_idx]

        # Teleworkable score for destination
        telework = telework_dict.get(dest, 0.0) if telework_dict else 0.0

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
            "telework": telework,
            "neg_d_sem_x_telework": neg_d_sem * telework,
            "neg_d_sem_x_post_x_telework": neg_d_sem * post_covid * telework,
        })

        # Add sampled alternatives
        for alt in sampled_alts:
            alt_idx = occ_to_idx[alt]
            neg_d_sem_alt = -d_sem_matrix[origin_idx, alt_idx]
            neg_d_inst_alt = -d_inst_matrix[origin_idx, alt_idx]
            telework_alt = telework_dict.get(alt, 0.0) if telework_dict else 0.0

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
                "telework": telework_alt,
                "neg_d_sem_x_telework": neg_d_sem_alt * telework_alt,
                "neg_d_sem_x_post_x_telework": neg_d_sem_alt * post_covid * telework_alt,
            })

        transition_id += 1

    return pd.DataFrame(rows)


# =============================================================================
# Model Estimation
# =============================================================================

def fit_conditional_logit_clustered(
    choice_df: pd.DataFrame,
    covariate_cols: List[str],
    cluster_col: str = "origin_occ",
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Fit conditional logit with clustered standard errors.
    """
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
    """Compute McFadden pseudo-R² for conditional logit."""
    ll_null = n_transitions * np.log(1.0 / n_alternatives)
    if ll_null == 0:
        return 0.0
    return 1.0 - (ll_model / ll_null)


def estimate_period(
    choice_df: pd.DataFrame,
    period: str,
) -> PeriodEstimate:
    """Estimate conditional logit for a single period."""
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


def estimate_structural_break(
    choice_df: pd.DataFrame,
) -> Tuple[PeriodEstimate, StructuralBreakResult]:
    """Test for structural break using LR test."""
    # Pooled model (restricted)
    pooled_cols = ["neg_d_sem", "neg_d_inst"]
    coefs_pooled, ses_pooled, ll_pooled, conv_pooled = fit_conditional_logit_clustered(
        choice_df, pooled_cols
    )

    n_transitions = choice_df["transition_id"].nunique()
    n_alternatives = len(choice_df) // n_transitions
    pseudo_r2_pooled = compute_pseudo_r2(ll_pooled, n_transitions, n_alternatives)

    pooled_result = PeriodEstimate(
        period="pooled",
        n_transitions=n_transitions,
        alpha=float(coefs_pooled[0]),
        se_alpha=float(ses_pooled[0]),
        gamma=float(coefs_pooled[1]),
        se_gamma=float(ses_pooled[1]),
        ll=float(ll_pooled),
        pseudo_r2=float(pseudo_r2_pooled),
        converged=conv_pooled,
    )

    # Interacted model (unrestricted)
    interacted_cols = ["neg_d_sem", "neg_d_inst", "neg_d_sem_x_post", "neg_d_inst_x_post"]
    coefs_inter, ses_inter, ll_inter, conv_inter = fit_conditional_logit_clustered(
        choice_df, interacted_cols
    )

    # LR test
    lr_stat = 2 * (ll_inter - ll_pooled)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=2)

    break_result = StructuralBreakResult(
        lr_stat=float(lr_stat),
        df=2,
        p_value=float(p_value),
        delta_alpha=float(coefs_inter[2]),
        se_delta_alpha=float(ses_inter[2]),
        delta_gamma=float(coefs_inter[3]),
        se_delta_gamma=float(ses_inter[3]),
    )

    return pooled_result, break_result


def estimate_remote_work_model(
    choice_df: pd.DataFrame,
) -> Dict:
    """
    Estimate remote work interaction model.

    Model:
        U_kj = α·(-d_sem) + γ·(-d_inst)
             + δ₂·(-d_sem)×PostCOVID
             + δ₃·(-d_sem)×Teleworkable_j
             + δ₄·(-d_sem)×PostCOVID×Teleworkable_j
    """
    covariate_cols = [
        "neg_d_sem",           # α
        "neg_d_inst",          # γ
        "neg_d_sem_x_post",    # δ₂
        "neg_d_sem_x_telework",      # δ₃
        "neg_d_sem_x_post_x_telework",  # δ₄
    ]

    coefs, ses, ll, converged = fit_conditional_logit_clustered(choice_df, covariate_cols)

    # Check power flag: SE(δ₄) > 2×|δ₄|
    delta_4 = float(coefs[4])
    se_delta_4 = float(ses[4])
    power_flag = se_delta_4 > 2 * abs(delta_4)

    # p-values
    z_scores = coefs / ses
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    return {
        "alpha": float(coefs[0]),
        "se_alpha": float(ses[0]),
        "gamma": float(coefs[1]),
        "se_gamma": float(ses[1]),
        "delta_2": float(coefs[2]),
        "se_delta_2": float(ses[2]),
        "p_delta_2": float(p_values[2]),
        "delta_3": float(coefs[3]),
        "se_delta_3": float(ses[3]),
        "p_delta_3": float(p_values[3]),
        "delta_4": delta_4,
        "se_delta_4": se_delta_4,
        "p_delta_4": float(p_values[4]),
        "power_flag": power_flag,
        "ll": float(ll),
        "converged": converged,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("v0.7.4.1: Pre/Post COVID Geometry Comparison (Full)")
    print("=" * 70)

    start_time = time.time()
    flags = []

    # =========================================================================
    # Step 1: Sample Split
    # =========================================================================
    print("\n--- Step 1: Sample Split ---")

    print("Loading transitions...")
    df = load_transitions()
    print(f"  Total transitions: {len(df):,}")
    print(f"  Year range: {df['year'].min()}-{df['year'].max()}")

    # Split by period
    pre_covid = df[df["year"] < 2020].copy()
    post_covid = df[df["year"] >= 2022].copy()
    excluded = df[(df["year"] >= 2020) & (df["year"] < 2022)].copy()

    n_pre = len(pre_covid)
    n_post = len(post_covid)
    n_excluded = len(excluded)

    print(f"\n  Pre-COVID (year < 2020):  {n_pre:,}")
    print(f"  Post-COVID (year >= 2022): {n_post:,}")
    print(f"  Excluded (2020-2021):      {n_excluded:,}")

    # STOP condition: n < 20,000
    if n_pre < 20000:
        print(f"\n  STOP: Pre-COVID sample size {n_pre:,} < 20,000 threshold")
        return {"error": "pre_covid_sample_too_small", "n_pre": n_pre}

    if n_post < 20000:
        print(f"\n  STOP: Post-COVID sample size {n_post:,} < 20,000 threshold")
        return {"error": "post_covid_sample_too_small", "n_post": n_post}

    print("\n  OK: Both periods exceed 20,000 threshold")

    # =========================================================================
    # Load Distance Matrices
    # =========================================================================
    print("\nLoading distance matrices...")
    d_sem, census_codes = load_wasserstein_census()
    d_inst, _ = load_institutional_census()

    print(f"  Semantic (Wasserstein): {d_sem.shape}")
    print(f"  Institutional: {d_inst.shape}")

    # Filter transitions to valid codes
    valid_codes = set(census_codes)

    def filter_to_valid(df_in):
        mask = df_in["origin_occ"].isin(valid_codes) & df_in["dest_occ"].isin(valid_codes)
        return df_in[mask].copy()

    pre_covid = filter_to_valid(pre_covid)
    post_covid = filter_to_valid(post_covid)
    combined = pd.concat([pre_covid, post_covid], ignore_index=True)

    print(f"\n  After filtering to valid codes:")
    print(f"    Pre-COVID:  {len(pre_covid):,}")
    print(f"    Post-COVID: {len(post_covid):,}")
    print(f"    Combined:   {len(combined):,}")

    # =========================================================================
    # Step 2: Period-Specific Estimation
    # =========================================================================
    print("\n--- Step 2: Period-Specific Estimation ---")

    print("\nBuilding choice datasets...")

    print("  Pre-COVID...")
    choice_pre = build_choice_dataset_with_period(
        pre_covid, d_sem, d_inst, census_codes, random_seed=42
    )
    print(f"    {choice_pre['transition_id'].nunique():,} transitions, {len(choice_pre):,} rows")

    print("  Post-COVID...")
    choice_post = build_choice_dataset_with_period(
        post_covid, d_sem, d_inst, census_codes, random_seed=43
    )
    print(f"    {choice_post['transition_id'].nunique():,} transitions, {len(choice_post):,} rows")

    print("\nEstimating period-specific models...")

    print("  Pre-COVID model...")
    result_pre = estimate_period(choice_pre, "pre_covid")
    print(f"    α = {result_pre.alpha:.4f} (SE = {result_pre.se_alpha:.4f})")
    print(f"    γ = {result_pre.gamma:.4f} (SE = {result_pre.se_gamma:.4f})")
    print(f"    LL = {result_pre.ll:.2f}, pseudo-R² = {result_pre.pseudo_r2:.4f}")

    if not result_pre.converged:
        print("  STOP: Pre-COVID model failed to converge")
        return {"error": "pre_covid_convergence_failure"}

    print("\n  Post-COVID model...")
    result_post = estimate_period(choice_post, "post_covid")
    print(f"    α = {result_post.alpha:.4f} (SE = {result_post.se_alpha:.4f})")
    print(f"    γ = {result_post.gamma:.4f} (SE = {result_post.se_gamma:.4f})")
    print(f"    LL = {result_post.ll:.2f}, pseudo-R² = {result_post.pseudo_r2:.4f}")

    if not result_post.converged:
        print("  STOP: Post-COVID model failed to converge")
        return {"error": "post_covid_convergence_failure"}

    # Flags
    if result_pre.pseudo_r2 < 0.05:
        flags.append(f"pre_covid_pseudo_r2_low ({result_pre.pseudo_r2:.4f})")
    if result_post.pseudo_r2 < 0.05:
        flags.append(f"post_covid_pseudo_r2_low ({result_post.pseudo_r2:.4f})")

    alpha_diff_pct = 100 * (result_post.alpha - result_pre.alpha) / abs(result_pre.alpha)
    gamma_diff_pct = 100 * (result_post.gamma - result_pre.gamma) / abs(result_pre.gamma) if result_pre.gamma != 0 else 0

    print(f"\n  Coefficient comparison:")
    print(f"    α change: {result_pre.alpha:.4f} → {result_post.alpha:.4f} ({alpha_diff_pct:+.1f}%)")
    print(f"    γ change: {result_pre.gamma:.4f} → {result_post.gamma:.4f} ({gamma_diff_pct:+.1f}%)")

    # =========================================================================
    # Step 3: Structural Break Test
    # =========================================================================
    print("\n--- Step 3: Structural Break Test ---")

    print("\nBuilding combined choice dataset...")
    choice_combined = build_choice_dataset_with_period(
        combined, d_sem, d_inst, census_codes, random_seed=44
    )
    print(f"  {choice_combined['transition_id'].nunique():,} transitions, {len(choice_combined):,} rows")

    print("\nEstimating structural break test...")
    pooled_result, break_result = estimate_structural_break(choice_combined)

    print(f"\n  Pooled model:")
    print(f"    α = {pooled_result.alpha:.4f} (SE = {pooled_result.se_alpha:.4f})")
    print(f"    γ = {pooled_result.gamma:.4f} (SE = {pooled_result.se_gamma:.4f})")

    print(f"\n  Structural break test:")
    print(f"    LR stat = {break_result.lr_stat:.4f}, df = {break_result.df}, p = {break_result.p_value:.4f}")
    print(f"    δ_α = {break_result.delta_alpha:.4f} (SE = {break_result.se_delta_alpha:.4f})")
    print(f"    δ_γ = {break_result.delta_gamma:.4f} (SE = {break_result.se_delta_gamma:.4f})")

    if break_result.p_value < 0.05:
        interpretation = "structural_break_detected"
    elif break_result.p_value < 0.10:
        interpretation = "marginal_break"
    else:
        interpretation = "geometry_stable"

    print(f"\n  RESULT: {interpretation}")

    # =========================================================================
    # Step 4: Remote Work Heterogeneity
    # =========================================================================
    print("\n--- Step 4: Remote Work Heterogeneity ---")

    print("\nLoading Dingel-Neiman teleworkable scores...")
    telework_dict, dn_coverage, n_matched = load_dingel_neiman_teleworkable()
    print(f"  Coverage: {dn_coverage:.1%} ({n_matched} Census codes matched)")

    # STOP condition: coverage < 70%
    if dn_coverage < 0.70:
        print(f"\n  STOP: Dingel-Neiman coverage {dn_coverage:.1%} < 70% threshold")
        return {
            "error": "dn_coverage_too_low",
            "coverage": dn_coverage,
            "n_matched": n_matched,
        }

    print("\n  OK: Coverage exceeds 70% threshold")

    print("\nBuilding choice dataset with teleworkable scores...")
    choice_rw = build_choice_dataset_with_period(
        combined, d_sem, d_inst, census_codes, random_seed=45, telework_dict=telework_dict
    )
    print(f"  {choice_rw['transition_id'].nunique():,} transitions")

    print("\nEstimating remote work interaction model...")
    rw_result = estimate_remote_work_model(choice_rw)

    print(f"\n  Results:")
    print(f"    δ₂ (sem×post) = {rw_result['delta_2']:.4f} (SE = {rw_result['se_delta_2']:.4f}, p = {rw_result['p_delta_2']:.4f})")
    print(f"    δ₃ (sem×telework) = {rw_result['delta_3']:.4f} (SE = {rw_result['se_delta_3']:.4f}, p = {rw_result['p_delta_3']:.4f})")
    print(f"    δ₄ (sem×post×telework) = {rw_result['delta_4']:.4f} (SE = {rw_result['se_delta_4']:.4f}, p = {rw_result['p_delta_4']:.4f})")
    print(f"    Power flag: {rw_result['power_flag']}")

    if rw_result['power_flag']:
        flags.append("remote_work_insufficient_power")

    # =========================================================================
    # Step 5: Robustness
    # =========================================================================
    print("\n--- Step 5: Robustness Checks ---")

    # 5a. cosine_embed instead of wasserstein
    print("\n5a. Robustness: cosine_embed distance...")
    try:
        d_cosine, _ = load_distance_matrix("cosine_embed")

        choice_pre_cos = build_choice_dataset_with_period(
            pre_covid, d_cosine, d_inst, census_codes, random_seed=42
        )
        choice_post_cos = build_choice_dataset_with_period(
            post_covid, d_cosine, d_inst, census_codes, random_seed=43
        )

        result_pre_cos = estimate_period(choice_pre_cos, "pre_covid_cosine")
        result_post_cos = estimate_period(choice_post_cos, "post_covid_cosine")

        alpha_cos_diff_pct = 100 * (result_post_cos.alpha - result_pre_cos.alpha) / abs(result_pre_cos.alpha)

        print(f"    α_pre = {result_pre_cos.alpha:.4f}, α_post = {result_post_cos.alpha:.4f} ({alpha_cos_diff_pct:+.1f}%)")

        cosine_robustness = {
            "alpha_pre": result_pre_cos.alpha,
            "alpha_post": result_post_cos.alpha,
            "pct_change": alpha_cos_diff_pct,
        }
    except Exception as e:
        print(f"    WARNING: cosine_embed robustness failed: {e}")
        cosine_robustness = {"error": str(e)}

    # 5b. Exclude 2022 (use 2023-2024 only)
    print("\n5b. Robustness: exclude 2022 (2023-2024 only)...")
    post_covid_no_2022 = post_covid[post_covid["year"] >= 2023].copy()
    print(f"    Post-COVID (2023+): {len(post_covid_no_2022):,} transitions")

    if len(post_covid_no_2022) >= 10000:
        combined_no_2022 = pd.concat([pre_covid, post_covid_no_2022], ignore_index=True)

        choice_no_2022 = build_choice_dataset_with_period(
            combined_no_2022, d_sem, d_inst, census_codes, random_seed=46, telework_dict=telework_dict
        )

        rw_no_2022 = estimate_remote_work_model(choice_no_2022)

        print(f"    δ₄ (excl 2022) = {rw_no_2022['delta_4']:.4f} (SE = {rw_no_2022['se_delta_4']:.4f}, p = {rw_no_2022['p_delta_4']:.4f})")

        exclude_2022_robustness = {
            "delta_4": rw_no_2022['delta_4'],
            "se_delta_4": rw_no_2022['se_delta_4'],
            "p_delta_4": rw_no_2022['p_delta_4'],
            "n_post_covid": len(post_covid_no_2022),
        }
    else:
        print(f"    WARNING: Post-COVID (2023+) sample too small ({len(post_covid_no_2022):,})")
        exclude_2022_robustness = {"error": "sample_too_small", "n": len(post_covid_no_2022)}

    # =========================================================================
    # Output
    # =========================================================================

    output = {
        "version": "0.7.4.1",
        "phase": "full",
        "sample_sizes": {
            "pre_covid": n_pre,
            "post_covid": n_post,
            "excluded_2020_2021": n_excluded,
            "pre_covid_valid": int(len(pre_covid)),
            "post_covid_valid": int(len(post_covid)),
        },
        "pre_covid": {
            "alpha": result_pre.alpha,
            "se_alpha": result_pre.se_alpha,
            "gamma": result_pre.gamma,
            "se_gamma": result_pre.se_gamma,
            "ll": result_pre.ll,
            "pseudo_r2": result_pre.pseudo_r2,
            "n_transitions": result_pre.n_transitions,
        },
        "post_covid": {
            "alpha": result_post.alpha,
            "se_alpha": result_post.se_alpha,
            "gamma": result_post.gamma,
            "se_gamma": result_post.se_gamma,
            "ll": result_post.ll,
            "pseudo_r2": result_post.pseudo_r2,
            "n_transitions": result_post.n_transitions,
        },
        "pooled": {
            "alpha": pooled_result.alpha,
            "se_alpha": pooled_result.se_alpha,
            "gamma": pooled_result.gamma,
            "se_gamma": pooled_result.se_gamma,
            "ll": pooled_result.ll,
            "pseudo_r2": pooled_result.pseudo_r2,
            "n_transitions": pooled_result.n_transitions,
        },
        "structural_break": {
            "lr_stat": break_result.lr_stat,
            "df": break_result.df,
            "p_value": break_result.p_value,
            "delta_alpha": break_result.delta_alpha,
            "se_delta_alpha": break_result.se_delta_alpha,
            "delta_gamma": break_result.delta_gamma,
            "se_delta_gamma": break_result.se_delta_gamma,
        },
        "remote_work": {
            "dn_coverage": dn_coverage,
            "n_matched": n_matched,
            "delta_2": rw_result['delta_2'],
            "se_delta_2": rw_result['se_delta_2'],
            "p_delta_2": rw_result['p_delta_2'],
            "delta_3": rw_result['delta_3'],
            "se_delta_3": rw_result['se_delta_3'],
            "p_delta_3": rw_result['p_delta_3'],
            "delta_4": rw_result['delta_4'],
            "se_delta_4": rw_result['se_delta_4'],
            "p_delta_4": rw_result['p_delta_4'],
            "power_flag": rw_result['power_flag'],
        },
        "robustness": {
            "cosine_embed": cosine_robustness,
            "exclude_2022": exclude_2022_robustness,
        },
        "coefficient_changes": {
            "alpha_pct_change": alpha_diff_pct,
            "gamma_pct_change": gamma_diff_pct,
        },
        "interpretation": interpretation,
        "flags": flags,
    }

    # Save
    output_path = save_experiment_output("pre_post_covid_v0741", output)
    print(f"\nSaved: {output_path}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Sample sizes: pre={n_pre:,}, post={n_post:,}")
    print(f"α_pre = {result_pre.alpha:.4f}, α_post = {result_post.alpha:.4f} ({alpha_diff_pct:+.1f}%)")
    print(f"LR test: χ²({break_result.df}) = {break_result.lr_stat:.2f}, p = {break_result.p_value:.4f}")
    print(f"Interpretation: {interpretation}")
    print(f"\nRemote work (D-N coverage: {dn_coverage:.1%}):")
    print(f"  δ₄ = {rw_result['delta_4']:.4f} (SE = {rw_result['se_delta_4']:.4f}, p = {rw_result['p_delta_4']:.4f})")
    print(f"  Power flag: {rw_result['power_flag']}")
    print(f"\nRobustness (cosine_embed): α change = {cosine_robustness.get('pct_change', 'N/A'):.1f}%" if 'pct_change' in cosine_robustness else "\nRobustness (cosine_embed): FAILED")
    if flags:
        print(f"\nFlags: {', '.join(flags)}")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
