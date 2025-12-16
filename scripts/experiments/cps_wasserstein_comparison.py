"""
Path A: CPS Mobility Comparison — Wasserstein vs Kernel Overlap

Pre-committed interpretation:
- OT significantly better: Adopt OT (Δ log-lik > 100 or Δα > 10%)
- Kernel significantly better: Stay with kernel
- OT ≈ Kernel: Prefer kernel (simpler)

Given ρ = 0.44 correlation, we expect meaningful differences in predictive power.

Approach:
1. Load both O*NET-level distance matrices
2. Aggregate to Census level using crosswalk (mean across O*NET codes per Census code)
3. Build choice dataset for CPS transitions
4. Run conditional logit with both distance measures
5. Compare coefficients and log-likelihoods
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from typing import Tuple, List, Dict

from task_space.mobility.census_crosswalk import (
    load_census_onet_crosswalk,
    aggregate_distances_to_census,
)
from task_space.mobility.choice_model import (
    build_choice_dataset,
    fit_conditional_logit,
    ChoiceModelResult,
)


def load_onet_distance_matrices() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load Wasserstein and kernel distances at O*NET level."""
    # Wasserstein
    wass_path = Path(".cache/artifacts/v1/wasserstein/d_wasserstein_onet.npz")
    wass_data = np.load(wass_path, allow_pickle=True)
    d_wasserstein = wass_data["distance_matrix"]
    wass_codes = list(wass_data["occupation_codes"])

    # Kernel (semantic) - misnamed "census" but is O*NET level
    sem_path = Path(".cache/artifacts/v1/mobility/d_sem_census.npz")
    sem_data = np.load(sem_path, allow_pickle=True)
    d_kernel = sem_data["d_sem"]
    kernel_codes = list(sem_data["occ_codes"])

    # Verify codes match
    assert wass_codes == kernel_codes, "O*NET codes don't match between matrices"

    print(f"Loaded O*NET-level matrices: {d_wasserstein.shape}")
    print(f"  Wasserstein range: [{d_wasserstein.min():.4f}, {d_wasserstein.max():.4f}]")
    print(f"  Kernel range: [{d_kernel.min():.4f}, {d_kernel.max():.4f}]")

    return d_wasserstein, d_kernel, wass_codes


def aggregate_to_census(
    d_onet: np.ndarray,
    onet_codes: List[str],
    crosswalk,
) -> Tuple[np.ndarray, List[int]]:
    """Aggregate O*NET distances to Census level."""
    d_census, census_codes = aggregate_distances_to_census(
        d_onet, onet_codes, crosswalk, aggregation="mean"
    )
    return d_census, census_codes


def load_institutional_distances() -> Tuple[np.ndarray, List[str]]:
    """Load institutional distances (already at O*NET level)."""
    inst_path = Path(".cache/artifacts/v1/mobility/d_inst_census.npz")
    inst_data = np.load(inst_path, allow_pickle=True)
    d_inst = inst_data["d_inst_matrix"]
    inst_codes = list(inst_data["occ_codes"])
    return d_inst, inst_codes


def run_single_model(
    transitions_df: pd.DataFrame,
    d_sem: np.ndarray,
    d_inst: np.ndarray,
    occ_codes: List,
    model_name: str,
) -> ChoiceModelResult:
    """Run conditional logit with given semantic distance matrix."""
    print(f"\n--- Running {model_name} Model ---")

    # Build choice dataset
    print("Building choice dataset...")
    choice_df = build_choice_dataset(
        transitions_df,
        d_sem_matrix=d_sem,
        d_inst_matrix=d_inst,
        occ_codes=occ_codes,
        n_alternatives=10,
        random_seed=42,
    )
    print(f"  Transitions: {choice_df['transition_id'].nunique():,}")
    print(f"  Choice rows: {len(choice_df):,}")

    # Fit model
    print("Fitting conditional logit...")
    result = fit_conditional_logit(choice_df)

    print(f"  α = {result.alpha:.4f} (t = {result.alpha_t:.1f})")
    print(f"  β = {result.beta:.4f} (t = {result.beta_t:.1f})")
    print(f"  Log-likelihood = {result.log_likelihood:.1f}")

    return result


def compare_models(
    kernel_result: ChoiceModelResult,
    wass_result: ChoiceModelResult,
) -> Dict:
    """Compare kernel and Wasserstein model results."""
    comparison = {
        "kernel": {
            "alpha": kernel_result.alpha,
            "alpha_t": kernel_result.alpha_t,
            "beta": kernel_result.beta,
            "beta_t": kernel_result.beta_t,
            "log_likelihood": kernel_result.log_likelihood,
            "n_transitions": kernel_result.n_transitions,
        },
        "wasserstein": {
            "alpha": wass_result.alpha,
            "alpha_t": wass_result.alpha_t,
            "beta": wass_result.beta,
            "beta_t": wass_result.beta_t,
            "log_likelihood": wass_result.log_likelihood,
            "n_transitions": wass_result.n_transitions,
        },
        "differences": {
            "delta_alpha": wass_result.alpha - kernel_result.alpha,
            "delta_alpha_pct": (wass_result.alpha - kernel_result.alpha) / kernel_result.alpha * 100,
            "delta_log_lik": wass_result.log_likelihood - kernel_result.log_likelihood,
            "delta_beta": wass_result.beta - kernel_result.beta,
        },
    }
    return comparison


def interpret_results(comparison: Dict) -> str:
    """Apply pre-committed interpretation rules."""
    delta_ll = comparison["differences"]["delta_log_lik"]
    delta_alpha_pct = comparison["differences"]["delta_alpha_pct"]

    # Pre-committed thresholds
    if delta_ll > 100 or delta_alpha_pct > 10:
        return "OT >> Kernel: Adopt OT (metric structure matters)"
    elif delta_ll < -100 or delta_alpha_pct < -10:
        return "Kernel >> OT: Stay with kernel (ground metric noise dominates)"
    else:
        return "OT ≈ Kernel: Prefer kernel (simpler, similar performance)"


def main():
    print("=" * 70)
    print("Path A: CPS Mobility Comparison — Wasserstein vs Kernel Overlap")
    print("=" * 70)

    # Load O*NET-level distance matrices
    print("\n1. Loading O*NET-level distance matrices...")
    d_wasserstein, d_kernel, onet_codes = load_onet_distance_matrices()

    # Load institutional distances
    print("\n2. Loading institutional distances...")
    d_inst, inst_codes = load_institutional_distances()

    # Load crosswalk
    print("\n3. Loading Census crosswalk...")
    crosswalk = load_census_onet_crosswalk()
    print(f"  O*NET codes: {crosswalk.n_onet}")
    print(f"  Census codes: {crosswalk.n_census}")
    print(f"  Coverage: {crosswalk.coverage:.1%}")

    # Aggregate to Census level
    print("\n4. Aggregating to Census level...")
    d_kernel_census, census_codes_kernel = aggregate_to_census(d_kernel, onet_codes, crosswalk)
    d_wass_census, census_codes_wass = aggregate_to_census(d_wasserstein, onet_codes, crosswalk)
    d_inst_census, census_codes_inst = aggregate_to_census(d_inst, inst_codes, crosswalk)

    print(f"  Kernel Census matrix: {d_kernel_census.shape}")
    print(f"  Wasserstein Census matrix: {d_wass_census.shape}")
    print(f"  Institutional Census matrix: {d_inst_census.shape}")

    # Verify Census codes match
    assert census_codes_kernel == census_codes_wass, "Census codes don't match"
    assert census_codes_kernel == census_codes_inst, "Census codes don't match with institutional"
    census_codes = census_codes_kernel

    # Load CPS transitions
    print("\n5. Loading CPS transitions...")
    transitions_df = pd.read_parquet("data/processed/mobility/verified_transitions.parquet")
    print(f"  Total transitions: {len(transitions_df):,}")

    # Filter to transitions with valid Census codes
    valid_codes = set(census_codes)
    mask = (
        transitions_df["origin_occ"].isin(valid_codes) &
        transitions_df["dest_occ"].isin(valid_codes)
    )
    transitions_filtered = transitions_df[mask].copy()
    print(f"  Valid transitions: {len(transitions_filtered):,}")

    # Run kernel model
    kernel_result = run_single_model(
        transitions_filtered,
        d_kernel_census,
        d_inst_census,
        census_codes,
        "Kernel"
    )

    # Run Wasserstein model
    wass_result = run_single_model(
        transitions_filtered,
        d_wass_census,
        d_inst_census,
        census_codes,
        "Wasserstein"
    )

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    comparison = compare_models(kernel_result, wass_result)

    print("\n| Metric | Kernel | Wasserstein | Δ |")
    print("|--------|--------|-------------|---|")
    print(f"| α | {comparison['kernel']['alpha']:.4f} | {comparison['wasserstein']['alpha']:.4f} | {comparison['differences']['delta_alpha']:+.4f} ({comparison['differences']['delta_alpha_pct']:+.1f}%) |")
    print(f"| α t-stat | {comparison['kernel']['alpha_t']:.1f} | {comparison['wasserstein']['alpha_t']:.1f} | — |")
    print(f"| β | {comparison['kernel']['beta']:.4f} | {comparison['wasserstein']['beta']:.4f} | {comparison['differences']['delta_beta']:+.4f} |")
    print(f"| Log-lik | {comparison['kernel']['log_likelihood']:.1f} | {comparison['wasserstein']['log_likelihood']:.1f} | {comparison['differences']['delta_log_lik']:+.1f} |")
    print(f"| N trans | {comparison['kernel']['n_transitions']:,} | {comparison['wasserstein']['n_transitions']:,} | — |")

    # Interpretation
    interpretation = interpret_results(comparison)
    print(f"\n**INTERPRETATION:** {interpretation}")

    # Save results
    output_dir = Path("outputs/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "path_a_wasserstein_vs_kernel",
        "version": "0.6.7.2",
        "timestamp": datetime.now().isoformat(),
        "comparison": comparison,
        "interpretation": interpretation,
        "pre_committed_thresholds": {
            "adopt_OT_if": "Δ log-lik > 100 OR Δα > 10%",
            "stay_kernel_if": "Δ log-lik < -100 OR Δα < -10%",
            "prefer_kernel_otherwise": "OT ≈ Kernel (simpler)",
        },
    }

    output_path = output_dir / "path_a_wasserstein_comparison_v0672.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return comparison, interpretation


if __name__ == "__main__":
    comparison, interpretation = main()
