#!/usr/bin/env python3
"""
Asymmetric Institutional Barriers Test v0.6.6.0

Tests whether upward occupational mobility (into higher job zones)
faces stronger institutional barriers than downward mobility.

Theory: Licensing restricts entry, not exit (Jackson 2023).
        Credentials are one-way gates.

Hypothesis: β_up > β_down (possibly β_down ≈ 0)

Models:
    1. Symmetric:  U_j = α * (-d_sem) + β * (-d_inst_sym) + ε
    2. Asymmetric: U_j = α * (-d_sem) + β_up * (-d_up) + β_down * (-d_down) + ε

Output: outputs/experiments/asymmetric_mobility_v0660.json

Usage:
    PYTHONPATH=src python scripts/experiments/asymmetric_mobility_test.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from task_space.mobility import (
    build_asymmetric_institutional_distance,
    verify_asymmetric_decomposition,
    load_census_onet_crosswalk,
    aggregate_distances_to_census,
    load_verified_transitions,
    build_choice_dataset,
    fit_conditional_logit,
    build_asymmetric_choice_dataset,
    fit_asymmetric_conditional_logit,
    compute_odds_ratios,
    compute_asymmetric_odds_ratios,
)


# Paths
ONET_PATH = Path("data/onet/db_30_0_excel")
CACHE_PATH = Path(".cache/artifacts/v1/mobility")
OUTPUT_PATH = Path("outputs/experiments/asymmetric_mobility_v0660.json")


def load_semantic_distance_onet() -> tuple[np.ndarray, list]:
    """Load cached O*NET-level semantic distance matrix."""
    cached = CACHE_PATH / "d_sem_census.npz"
    if not cached.exists():
        raise FileNotFoundError(
            f"Cached semantic distance not found at {cached}. "
            "Run the original mobility analysis first."
        )

    data = np.load(cached, allow_pickle=True)
    d_sem = data["d_sem"]
    occ_codes = data["occ_codes"].tolist()
    return d_sem, occ_codes


def aggregate_distance_to_census(
    d_matrix: np.ndarray,
    onet_codes: list,
    crosswalk,
) -> tuple[np.ndarray, list]:
    """
    Aggregate O*NET-level distance matrix to Census level.

    Uses the census_crosswalk module's aggregation function.
    """
    return aggregate_distances_to_census(
        d_matrix,
        onet_codes,
        crosswalk,
        aggregation="mean",
    )


def build_asymmetric_distance_census(
    crosswalk,
    onet_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Build Census-level asymmetric institutional distance matrices.

    Returns:
        d_up_census: Upward barrier matrix
        d_down_census: Downward barrier matrix
        d_sym_census: Symmetric distance matrix (for comparison)
        census_codes: Census occupation codes
    """
    print("  Building O*NET-level asymmetric distances...")
    asym_result = build_asymmetric_institutional_distance(onet_path)

    # Verify decomposition
    verification = verify_asymmetric_decomposition(asym_result)
    print(f"    Decomposition valid: {verification['all_properties_pass']}")

    print("  Aggregating to Census level...")
    # Aggregate each matrix separately
    d_up_census, census_codes = aggregate_distances_to_census(
        asym_result.d_up,
        asym_result.occupations,
        crosswalk,
        aggregation="mean",
    )

    d_down_census, _ = aggregate_distances_to_census(
        asym_result.d_down,
        asym_result.occupations,
        crosswalk,
        aggregation="mean",
    )

    d_sym_census, _ = aggregate_distances_to_census(
        asym_result.d_symmetric,
        asym_result.occupations,
        crosswalk,
        aggregation="mean",
    )

    return d_up_census, d_down_census, d_sym_census, census_codes


def compute_distance_correlations(
    d_sem: np.ndarray,
    d_up: np.ndarray,
    d_down: np.ndarray,
    d_sym: np.ndarray,
) -> dict:
    """Compute correlations between distance measures (off-diagonal only)."""
    n = d_sem.shape[0]

    # Extract upper triangle (off-diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    sem_flat = d_sem[mask]
    up_flat = d_up[mask]
    down_flat = d_down[mask]
    sym_flat = d_sym[mask]

    return {
        "sem_up": float(np.corrcoef(sem_flat, up_flat)[0, 1]),
        "sem_down": float(np.corrcoef(sem_flat, down_flat)[0, 1]),
        "sem_sym": float(np.corrcoef(sem_flat, sym_flat)[0, 1]),
        "up_down": float(np.corrcoef(up_flat, down_flat)[0, 1]),
        "up_sym": float(np.corrcoef(up_flat, sym_flat)[0, 1]),
        "down_sym": float(np.corrcoef(down_flat, sym_flat)[0, 1]),
    }


def compute_distance_stats(d: np.ndarray, name: str) -> dict:
    """Compute summary statistics for a distance matrix."""
    n = d.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = d[mask]

    return {
        "name": name,
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "frac_zero": float(np.mean(vals == 0)),
    }


def main():
    print("=" * 70)
    print("Asymmetric Institutional Barriers Test v0.6.6.0")
    print("=" * 70)
    print()
    print("Hypothesis: β_up > β_down (possibly β_down ≈ 0)")
    print("            Credentials restrict entry, not exit")
    print()

    # Load crosswalk
    print("Loading Census-O*NET crosswalk...")
    crosswalk = load_census_onet_crosswalk()
    print(f"  Coverage: {crosswalk.coverage:.1%} of O*NET codes")
    print(f"  Census occupations: {crosswalk.n_census}")
    print()

    # Load and aggregate semantic distance
    print("Loading semantic distance (O*NET level)...")
    d_sem_onet, sem_onet_codes = load_semantic_distance_onet()
    print(f"  O*NET matrix shape: {d_sem_onet.shape}")

    print("  Aggregating to Census level...")
    d_sem, sem_census_codes = aggregate_distance_to_census(d_sem_onet, sem_onet_codes, crosswalk)
    print(f"  Census matrix shape: {d_sem.shape}")
    print()

    # Build asymmetric institutional distances
    print("Building asymmetric institutional distances...")
    d_up, d_down, d_sym, inst_census_codes = build_asymmetric_distance_census(
        crosswalk, ONET_PATH
    )
    print(f"  Matrix shape: {d_up.shape}")
    print()

    # Verify census codes match
    if sem_census_codes != inst_census_codes:
        # Find intersection
        common_codes = sorted(set(sem_census_codes) & set(inst_census_codes))
        print(f"  Warning: Census codes differ. Using {len(common_codes)} common codes.")

        # Re-index matrices to common codes
        sem_idx = [sem_census_codes.index(c) for c in common_codes]
        inst_idx = [inst_census_codes.index(c) for c in common_codes]

        d_sem = d_sem[np.ix_(sem_idx, sem_idx)]
        d_up = d_up[np.ix_(inst_idx, inst_idx)]
        d_down = d_down[np.ix_(inst_idx, inst_idx)]
        d_sym = d_sym[np.ix_(inst_idx, inst_idx)]
        census_codes = common_codes
    else:
        census_codes = sem_census_codes

    print(f"Final matrix dimension: {len(census_codes)} occupations")
    print()

    # Distance statistics
    print("Distance matrix statistics:")
    stats = {
        "d_sem": compute_distance_stats(d_sem, "semantic"),
        "d_up": compute_distance_stats(d_up, "upward"),
        "d_down": compute_distance_stats(d_down, "downward"),
        "d_sym": compute_distance_stats(d_sym, "symmetric"),
    }
    for name, s in stats.items():
        print(f"  {name}: mean={s['mean']:.3f}, std={s['std']:.3f}, "
              f"frac_zero={s['frac_zero']:.1%}")
    print()

    # Distance correlations
    print("Distance correlations (off-diagonal pairs):")
    corr = compute_distance_correlations(d_sem, d_up, d_down, d_sym)
    for k, v in corr.items():
        print(f"  {k}: {v:.3f}")
    print()

    # Load verified transitions
    print("Loading verified transitions...")
    transitions = load_verified_transitions()
    print(f"  Transitions: {len(transitions):,}")
    print()

    # Build choice datasets
    print("Building choice datasets...")

    # Symmetric model dataset
    choice_sym = build_choice_dataset(
        transitions,
        d_sem,
        d_sym,  # Use symmetric institutional distance
        census_codes,
        n_alternatives=10,
        random_seed=42,
    )
    print(f"  Symmetric choice set: {len(choice_sym):,} rows")

    # Asymmetric model dataset
    choice_asym = build_asymmetric_choice_dataset(
        transitions,
        d_sem,
        d_up,
        d_down,
        census_codes,
        n_alternatives=10,
        random_seed=42,
    )
    print(f"  Asymmetric choice set: {len(choice_asym):,} rows")
    print()

    # Fit models
    print("Fitting models...")
    print()

    # Model 1: Symmetric
    print("  Model 1: Symmetric (baseline)")
    result_sym = fit_conditional_logit(choice_sym)
    print(f"    α (semantic):       {result_sym.alpha:.4f} (t={result_sym.alpha_t:.1f})")
    print(f"    β (institutional):  {result_sym.beta:.4f} (t={result_sym.beta_t:.1f})")
    print(f"    Log-likelihood:     {result_sym.log_likelihood:.1f}")
    print()

    # Model 2: Asymmetric
    print("  Model 2: Asymmetric (test)")
    result_asym = fit_asymmetric_conditional_logit(choice_asym)
    print(f"    α (semantic):       {result_asym.alpha:.4f} (t={result_asym.alpha_t:.1f})")
    print(f"    β_up (upward):      {result_asym.beta_up:.4f} (t={result_asym.beta_up_t:.1f})")
    print(f"    β_down (downward):  {result_asym.beta_down:.4f} (t={result_asym.beta_down_t:.1f})")
    print(f"    Log-likelihood:     {result_asym.log_likelihood:.1f}")
    print()

    # Hypothesis tests
    print("=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)
    print()

    print(f"Asymmetry ratio: |β_up| / |β_down| = {result_asym.asymmetry_ratio:.2f}")
    print()

    print(f"LR test (H0: β_up = β_down):")
    print(f"  LR statistic:  {result_asym.lr_test_statistic:.2f}")
    print(f"  p-value:       {result_asym.lr_test_pvalue:.2e}")
    print(f"  LL_restricted: {result_asym.ll_restricted:.1f}")
    print(f"  LL_unrestr:    {result_asym.log_likelihood:.1f}")
    print()

    # Odds ratios
    print("Odds ratios (1-unit distance increase):")
    or_asym = compute_asymmetric_odds_ratios(result_asym)
    for component in ["semantic", "upward", "downward"]:
        or_val = or_asym[component]["odds_ratio"]
        interp = or_asym[component]["interpretation"]
        print(f"  {component}: OR={or_val:.3f} ({interp})")
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    beta_up_sig = bool(result_asym.beta_up_p < 0.05)
    beta_down_sig = bool(result_asym.beta_down_p < 0.05)
    asymmetry_sig = bool(result_asym.lr_test_pvalue < 0.05)
    beta_down_near_zero = bool(abs(result_asym.beta_down) < 0.05)  # threshold for "near zero"

    interpretation = {
        "beta_up_significant": beta_up_sig,
        "beta_down_significant": beta_down_sig,
        "asymmetry_significant": asymmetry_sig,
        "beta_down_near_zero": beta_down_near_zero,
        "asymmetry_ratio": float(result_asym.asymmetry_ratio),
    }

    print(f"β_up significant (p < 0.05):   {beta_up_sig}")
    print(f"β_down significant (p < 0.05): {beta_down_sig}")
    print(f"Asymmetry significant:         {asymmetry_sig}")
    print(f"β_down near zero (|β| < 0.05): {beta_down_near_zero}")
    print()

    # Decision guidance
    if result_asym.asymmetry_ratio > 2 and asymmetry_sig:
        if beta_down_near_zero:
            verdict = "STRONG SUPPORT: Credentials are one-way gates (β_down ≈ 0)"
        else:
            verdict = "MODERATE SUPPORT: Both directions have friction, but upward is harder"
    elif result_asym.asymmetry_ratio < 1.5 or not asymmetry_sig:
        verdict = "SURPRISING NULL: Barriers appear symmetric"
    elif result_asym.beta_down > result_asym.beta_up:
        verdict = "CONTRADICTS THEORY: Downward harder than upward (investigate)"
    else:
        verdict = "WEAK ASYMMETRY: Ratio < 2, may not be economically meaningful"

    print(f"VERDICT: {verdict}")
    print()

    # Build output
    output = {
        "version": "0.6.6.0",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "beta_up > beta_down (credentials are one-way gates)",
        "sample": {
            "n_transitions": result_asym.n_transitions,
            "n_alternatives": 10,
            "n_census_occupations": len(census_codes),
        },
        "distance_statistics": stats,
        "distance_correlations": corr,
        "models": {
            "symmetric": {
                "alpha": result_sym.alpha,
                "alpha_se": result_sym.alpha_se,
                "alpha_t": result_sym.alpha_t,
                "alpha_p": result_sym.alpha_p,
                "beta": result_sym.beta,
                "beta_se": result_sym.beta_se,
                "beta_t": result_sym.beta_t,
                "beta_p": result_sym.beta_p,
                "log_likelihood": result_sym.log_likelihood,
                "n_transitions": result_sym.n_transitions,
            },
            "asymmetric": result_asym.to_dict(),
        },
        "hypothesis_tests": {
            "lr_test_statistic": result_asym.lr_test_statistic,
            "lr_test_pvalue": result_asym.lr_test_pvalue,
            "asymmetry_ratio": result_asym.asymmetry_ratio,
            "ll_restricted": result_asym.ll_restricted,
            "ll_unrestricted": result_asym.log_likelihood,
        },
        "odds_ratios": {
            "semantic": or_asym["semantic"]["odds_ratio"],
            "upward": or_asym["upward"]["odds_ratio"],
            "downward": or_asym["downward"]["odds_ratio"],
        },
        "interpretation": interpretation,
        "verdict": verdict,
    }

    # Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {OUTPUT_PATH}")
    print()

    # Return for decision points
    return output


if __name__ == "__main__":
    output = main()

    # Check decision points
    print("=" * 70)
    print("DECISION POINT CHECK")
    print("=" * 70)

    beta_down = output["models"]["asymmetric"]["beta_down"]
    beta_up = output["models"]["asymmetric"]["beta_up"]
    ratio = output["hypothesis_tests"]["asymmetry_ratio"]

    flags = []
    if beta_down > 0 and output["models"]["asymmetric"]["beta_down_p"] < 0.05:
        flags.append("⚠️  β_down is POSITIVE and significant")
    if abs(beta_up - beta_down) < 0.1 and ratio < 1.5:
        flags.append("⚠️  β_up ≈ β_down (surprising null)")
    if output["models"]["asymmetric"]["beta_up_p"] > 0.05:
        flags.append("⚠️  β_up is NOT significant")
    if ratio < 1.5:
        flags.append("⚠️  Asymmetry ratio < 1.5 (weak asymmetry)")

    if flags:
        print("FLAGS FOR HUMAN REVIEW:")
        for flag in flags:
            print(f"  {flag}")
    else:
        print("✓ No decision point flags triggered")
