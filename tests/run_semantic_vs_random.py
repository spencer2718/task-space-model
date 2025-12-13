#!/usr/bin/env python3
"""
Semantic vs. Random Comparison with Fixed Kernel.

Critical validation: Does semantic distance now beat random distance matrices
when kernel is properly calibrated?

This closes the loop on the original "random > semantic" anomaly from v0.5.0.

Usage:
    PYTHONPATH=src python tests/run_semantic_vs_random.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from scipy.stats import percentileofscore

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from task_space.domain import build_dwa_occupation_measures
from task_space.crosswalk import load_oes_panel, build_onet_oes_crosswalk, compute_wage_comovement, onet_to_soc


def compute_overlap_and_regression(
    occ_measures: np.ndarray,
    dist_matrix: np.ndarray,
    sigma: float,
    comovement_matrix: np.ndarray,
    occ_codes: list[str],
    comovement_codes: list[str],
    crosswalk_map: dict[str, str],
) -> tuple[float, float, float]:
    """
    Compute unnormalized kernel overlap and run OLS regression.

    Returns: (beta, t_stat, r_squared)
    """
    # Compute unnormalized kernel
    K = np.exp(-dist_matrix / sigma)

    # Compute overlap
    overlap = occ_measures @ K @ occ_measures.T

    # Aggregate to SOC level and run regression
    onet_to_soc_map = {k: v for k, v in crosswalk_map.items()}
    soc_codes = list(set(onet_to_soc_map.values()))
    soc_codes = [soc for soc in soc_codes if soc in comovement_codes]
    soc_codes = sorted(soc_codes)

    n_soc = len(soc_codes)
    soc_to_idx = {soc: i for i, soc in enumerate(soc_codes)}
    comovement_idx = {soc: i for i, soc in enumerate(comovement_codes)}
    onet_to_overlap_idx = {code: i for i, code in enumerate(occ_codes)}

    soc_overlap = np.zeros((n_soc, n_soc))
    soc_counts = np.zeros((n_soc, n_soc))

    for onet_i, soc_i in onet_to_soc_map.items():
        if soc_i not in soc_to_idx:
            continue
        for onet_j, soc_j in onet_to_soc_map.items():
            if soc_j not in soc_to_idx:
                continue
            if onet_i >= onet_j:
                continue
            i_idx = onet_to_overlap_idx[onet_i]
            j_idx = onet_to_overlap_idx[onet_j]
            overlap_val = overlap[i_idx, j_idx]
            si = soc_to_idx[soc_i]
            sj = soc_to_idx[soc_j]
            if si > sj:
                si, sj = sj, si
            soc_overlap[si, sj] += overlap_val
            soc_counts[si, sj] += 1

    soc_counts[soc_counts == 0] = 1
    soc_overlap = soc_overlap / soc_counts

    # Build pair arrays
    pairs_x, pairs_y = [], []
    for i in range(n_soc):
        for j in range(i + 1, n_soc):
            ci = comovement_idx[soc_codes[i]]
            cj = comovement_idx[soc_codes[j]]
            comove_val = comovement_matrix[ci, cj]
            if np.isnan(comove_val):
                continue
            pairs_x.append(soc_overlap[i, j])
            pairs_y.append(comove_val)

    x = np.array(pairs_x)
    y = np.array(pairs_y)

    # Simple OLS regression
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()

    # Beta = Cov(x,y) / Var(x)
    cov_xy = np.sum((x - x_mean) * (y - y_mean)) / n
    var_x = np.sum((x - x_mean) ** 2) / n
    beta = cov_xy / var_x if var_x > 0 else 0

    # R-squared
    y_pred = x_mean + beta * (x - x_mean)  # Using centered form
    ss_res = np.sum((y - (y_mean + beta * (x - x_mean))) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # t-stat from correlation
    r, p = sp_stats.pearsonr(x, y)
    t_stat = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else float('inf')

    return beta, t_stat, r_squared


def main():
    output_dir = Path("outputs/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fixed kernel parameters from Phase 1
    SIGMA = 0.2230  # Median of NN distances
    N_RANDOM_SEEDS = 100

    print("=" * 60)
    print("Semantic vs. Random Comparison (Fixed Kernel)")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  σ = {SIGMA}")
    print(f"  Kernel: Unnormalized")
    print(f"  Random seeds: {N_RANDOM_SEEDS}")

    # Load data
    print("\n[1/4] Loading data...")
    measures = build_dwa_occupation_measures()

    # Load semantic distance matrix
    embeddings = np.load(output_dir / "activity_embeddings.npy")
    from sklearn.metrics.pairwise import cosine_distances
    semantic_dist = cosine_distances(embeddings)

    print(f"  - {len(measures.occupation_codes)} occupations")
    print(f"  - {semantic_dist.shape[0]} activities")

    # Load wage comovement
    print("\n[2/4] Loading wage comovement...")
    oes_panel = load_oes_panel(years=[2019, 2020, 2021, 2022, 2023])
    comovement = compute_wage_comovement(oes_panel, min_years=4)

    # Build crosswalk
    crosswalk_map = {}
    for onet_code in measures.occupation_codes:
        soc = onet_to_soc(onet_code)
        if soc in comovement.occupation_codes:
            crosswalk_map[onet_code] = soc

    print(f"  - {len(comovement.occupation_codes)} SOC occupations")

    # Run semantic regression
    print("\n[3/4] Computing semantic overlap regression...")
    semantic_beta, semantic_t, semantic_r2 = compute_overlap_and_regression(
        occ_measures=measures.occupation_matrix,
        dist_matrix=semantic_dist,
        sigma=SIGMA,
        comovement_matrix=comovement.comovement_matrix,
        occ_codes=measures.occupation_codes,
        comovement_codes=comovement.occupation_codes,
        crosswalk_map=crosswalk_map,
    )
    print(f"  - Semantic β = {semantic_beta:.6f}")
    print(f"  - Semantic t = {semantic_t:.2f}")
    print(f"  - Semantic R² = {semantic_r2:.6f}")

    # Run random baselines
    print(f"\n[4/4] Computing {N_RANDOM_SEEDS} random baseline regressions...")
    random_betas = []
    random_ts = []
    random_r2s = []

    n_activities = semantic_dist.shape[0]

    for seed in range(N_RANDOM_SEEDS):
        if (seed + 1) % 20 == 0:
            print(f"  - Completed {seed + 1}/{N_RANDOM_SEEDS}")

        np.random.seed(seed)

        # Generate random distance matrix
        random_dist = np.random.rand(n_activities, n_activities)
        random_dist = (random_dist + random_dist.T) / 2  # Symmetrize
        np.fill_diagonal(random_dist, 0)

        beta, t, r2 = compute_overlap_and_regression(
            occ_measures=measures.occupation_matrix,
            dist_matrix=random_dist,
            sigma=SIGMA,
            comovement_matrix=comovement.comovement_matrix,
            occ_codes=measures.occupation_codes,
            comovement_codes=comovement.occupation_codes,
            crosswalk_map=crosswalk_map,
        )

        random_betas.append(beta)
        random_ts.append(t)
        random_r2s.append(r2)

    random_betas = np.array(random_betas)
    random_ts = np.array(random_ts)
    random_r2s = np.array(random_r2s)

    # Compute statistics
    semantic_percentile_beta = percentileofscore(random_betas, semantic_beta)
    semantic_percentile_t = percentileofscore(random_ts, semantic_t)
    semantic_percentile_r2 = percentileofscore(random_r2s, semantic_r2)

    semantic_beats_random_95 = semantic_beta > np.percentile(random_betas, 95)
    semantic_beats_random_99 = semantic_beta > np.percentile(random_betas, 99)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n## Semantic Results")
    print(f"  β = {semantic_beta:.6f}")
    print(f"  t = {semantic_t:.2f}")
    print(f"  R² = {semantic_r2:.6f}")

    print("\n## Random Baseline Distribution")
    print(f"  β: mean = {random_betas.mean():.6f}, std = {random_betas.std():.6f}")
    print(f"     p5 = {np.percentile(random_betas, 5):.6f}, p95 = {np.percentile(random_betas, 95):.6f}")
    print(f"  t: mean = {random_ts.mean():.2f}, std = {random_ts.std():.2f}")
    print(f"     p5 = {np.percentile(random_ts, 5):.2f}, p95 = {np.percentile(random_ts, 95):.2f}")
    print(f"  R²: mean = {random_r2s.mean():.6f}, std = {random_r2s.std():.6f}")

    print("\n## Comparison")
    print(f"  Semantic β percentile: {semantic_percentile_beta:.1f}%")
    print(f"  Semantic t percentile: {semantic_percentile_t:.1f}%")
    print(f"  Semantic R² percentile: {semantic_percentile_r2:.1f}%")
    print(f"  Semantic beats random (95th): {semantic_beats_random_95}")
    print(f"  Semantic beats random (99th): {semantic_beats_random_99}")

    # Decision
    print("\n" + "=" * 60)
    print("DECISION")
    print("=" * 60)

    if semantic_percentile_t >= 99:
        print(f"\n✓ SEMANTIC STRONGLY BEATS RANDOM")
        print(f"  Semantic t = {semantic_t:.2f} is at {semantic_percentile_t:.1f}th percentile")
        print(f"  Random t: mean = {random_ts.mean():.2f}, max = {random_ts.max():.2f}")
        print(f"\n  The 'random > semantic' finding from v0.5.0 was an ARTIFACT.")
        print(f"  With proper kernel calibration, semantic structure is highly predictive.")
    elif semantic_percentile_t >= 95:
        print(f"\n✓ SEMANTIC BEATS RANDOM (p < 0.05)")
        print(f"  Semantic t = {semantic_t:.2f} is at {semantic_percentile_t:.1f}th percentile")
    else:
        print(f"\n✗ SEMANTIC DOES NOT CLEARLY BEAT RANDOM")
        print(f"  Semantic t = {semantic_t:.2f} is at {semantic_percentile_t:.1f}th percentile")
        print(f"  Further investigation needed.")

    # Save results
    output = {
        "parameters": {
            "sigma": SIGMA,
            "n_random_seeds": N_RANDOM_SEEDS,
            "kernel_normalized": False,
        },
        "semantic": {
            "beta": float(semantic_beta),
            "t_stat": float(semantic_t),
            "r_squared": float(semantic_r2),
        },
        "random_distribution": {
            "beta_mean": float(random_betas.mean()),
            "beta_std": float(random_betas.std()),
            "beta_min": float(random_betas.min()),
            "beta_max": float(random_betas.max()),
            "beta_p5": float(np.percentile(random_betas, 5)),
            "beta_p25": float(np.percentile(random_betas, 25)),
            "beta_p50": float(np.percentile(random_betas, 50)),
            "beta_p75": float(np.percentile(random_betas, 75)),
            "beta_p95": float(np.percentile(random_betas, 95)),
            "beta_p99": float(np.percentile(random_betas, 99)),
            "t_mean": float(random_ts.mean()),
            "t_std": float(random_ts.std()),
            "t_p95": float(np.percentile(random_ts, 95)),
            "t_p99": float(np.percentile(random_ts, 99)),
            "r2_mean": float(random_r2s.mean()),
            "r2_std": float(random_r2s.std()),
        },
        "comparison": {
            "semantic_beta_percentile": float(semantic_percentile_beta),
            "semantic_t_percentile": float(semantic_percentile_t),
            "semantic_r2_percentile": float(semantic_percentile_r2),
            "semantic_beats_random_95": bool(semantic_beats_random_95),
            "semantic_beats_random_99": bool(semantic_beats_random_99),
        },
        "conclusion": {
            "original_finding": "Random > semantic (v0.5.0)",
            "revised_finding": "Semantic >> random with fixed kernel" if semantic_percentile_t >= 95 else "Inconclusive",
            "was_artifact": bool(semantic_percentile_t >= 95),
        }
    }

    with open(output_dir / "semantic_vs_random_fixed.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_dir}/semantic_vs_random_fixed.json")


if __name__ == "__main__":
    main()
