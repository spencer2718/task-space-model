#!/usr/bin/env python3
"""
v0.7.3.3: Ground Metric Validation

Tests whether the embedding-based ground metric adds value over identity ground metric.
This confirms whether semantic task similarity (e.g., "operating forklift" ≈ "driving delivery vehicle")
contributes beyond raw task overlap.

**Wasserstein-identity:** Ground cost c(i,j) = 1 if DWA_i ≠ DWA_j, 0 if same.
This reduces to Total Variation distance: W_identity(P,Q) = 0.5 * Σ|p_i - q_i|

**Wasserstein-embedding:** Existing at .cache/artifacts/v1/mobility/d_wasserstein_census.npz
Uses MPNet cosine distance as ground metric.

**Comparison:** Run same conditional logit on 89,329 CPS transitions:
P(j|i) ∝ exp(-α·d(i,j) - γ·d_inst(i,j))

Success criterion: Wasserstein-embedding achieves ≥5% higher pseudo-R² than Wasserstein-identity.
"""

import json
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from task_space.domain import build_dwa_occupation_measures
from task_space.mobility.census_crosswalk import (
    load_census_onet_crosswalk,
    aggregate_distances_to_census,
)
from task_space.mobility.choice_model import (
    build_choice_dataset,
    fit_conditional_logit,
)


CACHE_DIR = Path(__file__).parent.parent / ".cache" / "artifacts" / "v1" / "mobility"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "experiments"


@dataclass
class ModelResult:
    """Result container for conditional logit."""
    name: str
    alpha: float
    alpha_se: float
    gamma: float
    gamma_se: float
    ll: float
    ll_null: float
    pseudo_r2: float
    n_transitions: int
    converged: bool


def compute_wasserstein_identity_onet(occ_measures):
    """
    Compute Wasserstein distance with identity ground metric.

    With identity ground cost c(i,j) = 1 - δ(i,j), Wasserstein reduces to
    Total Variation distance: W_TV(P,Q) = 0.5 * Σ|p_i - q_i|

    Parameters
    ----------
    occ_measures : OccupationMeasures
        Object with occupation_matrix (normalized to sum to 1 per row)

    Returns
    -------
    tuple
        (distance_matrix, occupation_codes)
    """
    print("\n=== Computing Wasserstein-identity (Total Variation) ===")

    # Use normalized probability distributions (rows sum to 1)
    P = occ_measures.occupation_matrix
    n_occ = P.shape[0]

    print(f"Matrix shape: {P.shape}")

    # Verify normalization
    row_sums = P.sum(axis=1)
    print(f"Row sums range: [{row_sums.min():.6f}, {row_sums.max():.6f}]")

    # Compute pairwise Total Variation distance
    # TV(P,Q) = 0.5 * ||P - Q||_1
    D = np.zeros((n_occ, n_occ))

    for i in range(n_occ):
        # Vectorized: compute |P[i] - P[j]| for all j >= i
        diff = np.abs(P[i:i+1, :] - P[i:, :])  # Broadcasting
        tv_dists = 0.5 * diff.sum(axis=1)
        D[i, i:] = tv_dists
        D[i:, i] = tv_dists  # Symmetric

    print(f"Distance range: [{D.min():.4f}, {D.max():.4f}]")
    print(f"Diagonal zero check: {np.allclose(np.diag(D), 0)}")

    # Non-zero distances
    nonzero = D[D > 0]
    print(f"Non-zero distances: median={np.median(nonzero):.4f}, mean={np.mean(nonzero):.4f}")

    return D, occ_measures.occupation_codes


def load_distance_matrix(name: str) -> Tuple[np.ndarray, List[int]]:
    """Load a distance matrix and census codes."""
    path = CACHE_DIR / f"d_{name}_census.npz"
    data = np.load(path, allow_pickle=True)
    distances = data["distances"]

    # Try different key names for census codes
    if "census_codes" in data.files:
        codes = list(data["census_codes"])
    elif "occupation_codes" in data.files:
        codes = list(data["occupation_codes"])
    else:
        raise ValueError(f"No census codes found in {path}")

    return distances, codes


def load_institutional_distance() -> np.ndarray:
    """Load institutional distance matrix at Census level."""
    from task_space.mobility.census_crosswalk import aggregate_distances_to_census

    path = CACHE_DIR / "d_inst_census.npz"
    data = np.load(path, allow_pickle=True)

    d_inst_onet = data["d_inst_matrix"]
    onet_codes = list(data["occ_codes"])

    # Aggregate to Census level
    crosswalk = load_census_onet_crosswalk()
    d_inst_census, _ = aggregate_distances_to_census(
        d_inst_onet, onet_codes, crosswalk, aggregation="mean"
    )

    # Zero the diagonal
    np.fill_diagonal(d_inst_census, 0)

    return d_inst_census


def compute_null_ll(n_transitions: int, n_alternatives: int = 11) -> float:
    """Compute null log-likelihood for conditional logit."""
    return n_transitions * np.log(1 / n_alternatives)


def fit_model(
    name: str,
    d_sem: np.ndarray,
    d_inst: np.ndarray,
    census_codes: List[int],
    transitions_df: pd.DataFrame,
    n_alternatives: int = 10,
) -> ModelResult:
    """Fit conditional logit and compute pseudo-R²."""
    print(f"\n--- Fitting {name} model ---")

    # Build choice dataset
    choice_df = build_choice_dataset(
        transitions_df,
        d_sem_matrix=d_sem,
        d_inst_matrix=d_inst,
        occ_codes=census_codes,
        n_alternatives=n_alternatives,
        random_seed=42,
    )

    n_trans = choice_df["transition_id"].nunique()
    print(f"  Transitions: {n_trans:,}")

    # Fit model
    result = fit_conditional_logit(choice_df)

    # Compute null LL and pseudo-R²
    ll_null = compute_null_ll(n_trans, n_alternatives + 1)
    pseudo_r2 = 1 - (result.log_likelihood / ll_null)

    print(f"  α = {result.alpha:.4f} (SE = {result.alpha_se:.4f})")
    print(f"  γ = {result.beta:.4f} (SE = {result.beta_se:.4f})")
    print(f"  LL = {result.log_likelihood:,.1f}")
    print(f"  LL_null = {ll_null:,.1f}")
    print(f"  Pseudo-R² = {pseudo_r2:.4f} ({pseudo_r2*100:.2f}%)")

    return ModelResult(
        name=name,
        alpha=result.alpha,
        alpha_se=result.alpha_se,
        gamma=result.beta,
        gamma_se=result.beta_se,
        ll=result.log_likelihood,
        ll_null=ll_null,
        pseudo_r2=pseudo_r2,
        n_transitions=n_trans,
        converged=result.converged,
    )


def main():
    print("=" * 70)
    print("v0.7.3.3: Ground Metric Validation")
    print("=" * 70)

    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Compute Wasserstein-identity at O*NET level
    print("\n1. Computing Wasserstein-identity (Total Variation)...")
    occ_measures = build_dwa_occupation_measures(normalize=True)
    d_identity_onet, onet_codes = compute_wasserstein_identity_onet(occ_measures)
    print(f"O*NET level: {d_identity_onet.shape}")

    # Step 2: Aggregate to Census level
    print("\n2. Aggregating to Census level...")
    crosswalk = load_census_onet_crosswalk()
    d_identity_census, census_codes = aggregate_distances_to_census(
        d_identity_onet, onet_codes, crosswalk, aggregation="mean"
    )
    # Zero the diagonal (aggregation can create non-zero diagonal)
    np.fill_diagonal(d_identity_census, 0)
    print(f"Census level: {d_identity_census.shape}")
    print(f"Range: [{d_identity_census.min():.4f}, {d_identity_census.max():.4f}]")

    # Save the matrix
    np.savez_compressed(
        CACHE_DIR / "d_wasserstein_identity_census.npz",
        distances=d_identity_census,
        census_codes=census_codes,
        metric="wasserstein_identity",
        description="Wasserstein with identity ground metric (= Total Variation distance)",
    )
    print(f"Saved: {CACHE_DIR / 'd_wasserstein_identity_census.npz'}")

    # Step 3: Load Wasserstein-embedding for comparison
    print("\n3. Loading Wasserstein-embedding...")
    d_embedding, _ = load_distance_matrix("wasserstein")
    print(f"Shape: {d_embedding.shape}")
    print(f"Range: [{d_embedding.min():.4f}, {d_embedding.max():.4f}]")

    # Compare the two distances
    print("\n4. Distance comparison...")
    # Flatten upper triangle for correlation
    triu_idx = np.triu_indices(len(census_codes), k=1)
    d_id_flat = d_identity_census[triu_idx]
    d_em_flat = d_embedding[triu_idx]

    corr = np.corrcoef(d_id_flat, d_em_flat)[0, 1]
    print(f"Pearson r(identity, embedding): {corr:.4f}")

    # Rank correlation
    from scipy.stats import spearmanr
    rho, _ = spearmanr(d_id_flat, d_em_flat)
    print(f"Spearman ρ(identity, embedding): {rho:.4f}")

    # Step 5: Load institutional distance
    print("\n5. Loading institutional distance...")
    d_inst = load_institutional_distance()
    print(f"Shape: {d_inst.shape}")

    # Step 6: Load transitions
    print("\n6. Loading CPS transitions...")
    transitions_df = pd.read_parquet("data/processed/mobility/verified_transitions.parquet")
    print(f"Total: {len(transitions_df):,}")

    # Filter to valid codes
    valid_codes = set(census_codes)
    mask = (
        transitions_df["origin_occ"].isin(valid_codes) &
        transitions_df["dest_occ"].isin(valid_codes)
    )
    transitions_filtered = transitions_df[mask].copy()
    print(f"Filtered: {len(transitions_filtered):,}")

    # Step 7: Fit models
    print("\n7. Fitting models...")

    results = {}
    results["wasserstein_identity"] = fit_model(
        name="wasserstein_identity",
        d_sem=d_identity_census,
        d_inst=d_inst,
        census_codes=census_codes,
        transitions_df=transitions_filtered,
    )

    results["wasserstein_embedding"] = fit_model(
        name="wasserstein_embedding",
        d_sem=d_embedding,
        d_inst=d_inst,
        census_codes=census_codes,
        transitions_df=transitions_filtered,
    )

    # Step 8: Comparison and gate evaluation
    print("\n8. Gate evaluation...")

    r_id = results["wasserstein_identity"]
    r_em = results["wasserstein_embedding"]

    delta_ll = r_em.ll - r_id.ll
    delta_pseudo_r2 = r_em.pseudo_r2 - r_id.pseudo_r2
    pseudo_r2_ratio = r_em.pseudo_r2 / r_id.pseudo_r2 if r_id.pseudo_r2 > 0 else float('inf')
    embedding_improvement_pct = 100 * (pseudo_r2_ratio - 1)

    print(f"\nWasserstein-identity pseudo-R²: {r_id.pseudo_r2:.4f} ({r_id.pseudo_r2*100:.2f}%)")
    print(f"Wasserstein-embedding pseudo-R²: {r_em.pseudo_r2:.4f} ({r_em.pseudo_r2*100:.2f}%)")
    print(f"ΔLL = {delta_ll:+,.1f}")
    print(f"Δpseudo-R² = {delta_pseudo_r2:+.4f} ({delta_pseudo_r2*100:+.2f}pp)")
    print(f"Pseudo-R² ratio = {pseudo_r2_ratio:.4f}")
    print(f"Embedding improvement = {embedding_improvement_pct:+.2f}%")

    # Success criterion: embedding >= 5% higher pseudo-R²
    gate_passed = pseudo_r2_ratio >= 1.05
    print(f"\nSuccess criterion (≥5% improvement): {'PASS' if gate_passed else 'FAIL'}")

    # Interpretation
    if gate_passed:
        interpretation = "Embedding ground metric substantially improves over identity; semantic task similarity matters"
    elif pseudo_r2_ratio >= 1.01:
        interpretation = "Embedding ground metric provides modest improvement; distributional treatment may be doing the work"
    else:
        interpretation = "Embedding ground metric does not improve over identity; knowing 'different tasks' is sufficient"

    print(f"Interpretation: {interpretation}")

    # Build output
    output = {
        "version": "0.7.3.3",
        "sample_n": int(r_em.n_transitions),
        "models": {
            "wasserstein_identity": {
                "alpha": float(r_id.alpha),
                "alpha_se": float(r_id.alpha_se),
                "gamma": float(r_id.gamma),
                "gamma_se": float(r_id.gamma_se),
                "ll": float(r_id.ll),
                "pseudo_r2": float(r_id.pseudo_r2),
            },
            "wasserstein_embedding": {
                "alpha": float(r_em.alpha),
                "alpha_se": float(r_em.alpha_se),
                "gamma": float(r_em.gamma),
                "gamma_se": float(r_em.gamma_se),
                "ll": float(r_em.ll),
                "pseudo_r2": float(r_em.pseudo_r2),
            },
        },
        "comparison": {
            "delta_ll": float(delta_ll),
            "delta_pseudo_r2": float(delta_pseudo_r2),
            "pseudo_r2_ratio": float(pseudo_r2_ratio),
            "embedding_improvement_pct": float(embedding_improvement_pct),
        },
        "distance_correlation": {
            "pearson_r": float(corr),
            "spearman_rho": float(rho),
        },
        "gate_passed": bool(gate_passed),
        "interpretation": interpretation,
        "notes": [
            "Wasserstein-identity = Total Variation distance (ground cost = 1 if different DWA)",
            "Wasserstein-embedding = Original Wasserstein with MPNet cosine ground metric",
            f"r(identity, embedding) = {corr:.4f} — distances are {('moderately' if 0.3 < abs(corr) < 0.7 else 'highly' if abs(corr) >= 0.7 else 'weakly')} correlated",
        ],
    }

    # Save results
    output_path = OUTPUT_DIR / "ground_metric_validation_v0733.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'α':>8} {'γ':>8} {'LL':>12} {'Pseudo-R²':>10}")
    print("-" * 70)
    print(f"{'wasserstein_identity':<25} {r_id.alpha:>8.3f} {r_id.gamma:>8.3f} {r_id.ll:>12,.0f} {r_id.pseudo_r2:>10.4f}")
    print(f"{'wasserstein_embedding':<25} {r_em.alpha:>8.3f} {r_em.gamma:>8.3f} {r_em.ll:>12,.0f} {r_em.pseudo_r2:>10.4f}")
    print("-" * 70)
    print(f"Δ (embedding - identity)   {r_em.alpha - r_id.alpha:>+8.3f} {r_em.gamma - r_id.gamma:>+8.3f} {delta_ll:>+12,.0f} {delta_pseudo_r2:>+10.4f}")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
