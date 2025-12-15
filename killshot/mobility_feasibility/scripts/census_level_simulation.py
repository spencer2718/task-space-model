"""
Task A: Re-Run Sparsity Simulation at Census Level

Aggregates O*NET occupations to Census 2010 codes (~460) and tests
whether parameter recovery still works under realistic assumptions.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from task_space import build_dwa_occupation_measures
from task_space.similarity.overlap import compute_normalized_overlap

OUTPUT_DIR = Path("temp/mobility_feasibility/outputs")
DATA_DIR = Path("temp/mobility_feasibility/data/crosswalks")


def main():
    print("=" * 70)
    print("TASK A: CENSUS-LEVEL SPARSITY SIMULATION")
    print("=" * 70)

    # Load O*NET measures
    print("\nLoading O*NET data...")
    measures = build_dwa_occupation_measures()
    rho_onet = measures.occupation_matrix
    onet_codes = np.array(measures.occupation_codes)
    n_onet = len(onet_codes)
    print(f"O*NET occupations: {n_onet}")

    # Load activity kernel
    dist_data = np.load(".cache/artifacts/v1/distances/cosine_920fd8058161b39f.npz")
    d_act = dist_data["distances"]
    sigma = 0.223
    K = np.exp(-d_act / sigma)

    # Compute O*NET-level semantic distances
    print("Computing O*NET-level semantic distances...")
    sim_sem_onet = compute_normalized_overlap(rho_onet, K)
    d_sem_onet = 1.0 - sim_sem_onet

    # Load job zone distances
    print("Loading institutional distances...")
    jz_data = np.load(OUTPUT_DIR / "job_zone_matrix.npz", allow_pickle=True)
    d_inst_full = jz_data["d_inst_matrix"]
    inst_codes = jz_data["occ_codes"]

    # Align to common occupations
    common = sorted(set(onet_codes) & set(inst_codes))
    print(f"Common occupations: {len(common)}")

    onet_idx = [np.where(onet_codes == c)[0][0] for c in common]
    inst_idx = [np.where(inst_codes == c)[0][0] for c in common]

    rho_aligned = rho_onet[onet_idx]
    d_sem_aligned = d_sem_onet[np.ix_(onet_idx, onet_idx)]
    d_inst_aligned = d_inst_full[np.ix_(inst_idx, inst_idx)]
    aligned_codes = np.array(common)

    # Compute O*NET-level correlation
    upper_sem_onet = d_sem_aligned[np.triu_indices_from(d_sem_aligned, k=1)]
    upper_inst_onet = d_inst_aligned[np.triu_indices_from(d_inst_aligned, k=1)]
    corr_onet = np.corrcoef(upper_sem_onet, upper_inst_onet)[0, 1]
    print(f"O*NET-level correlation: {corr_onet:.3f}")

    # Load crosswalk
    crosswalk = pd.read_csv(DATA_DIR / "onet_to_census_improved.csv")
    xw = crosswalk[crosswalk["matched"]].copy()

    # Get unique Census codes
    census_codes = sorted(xw["census_2010"].dropna().unique())
    n_census = len(census_codes)
    print(f"\nAggregating to {n_census} Census codes...")

    # Build Census → O*NET index mapping
    onet_to_idx = {c: i for i, c in enumerate(aligned_codes)}
    census_to_onet = {int(c): [] for c in census_codes}

    for _, row in xw.iterrows():
        onet = row["onet_soc"]
        census = int(row["census_2010"])
        if onet in onet_to_idx:
            census_to_onet[census].append(onet_to_idx[onet])

    # Aggregate task vectors
    n_act = rho_aligned.shape[1]
    rho_census = np.zeros((n_census, n_act))
    for i, census in enumerate(census_codes):
        onet_idxs = census_to_onet[int(census)]
        if onet_idxs:
            rho_census[i] = rho_aligned[onet_idxs].mean(axis=0)
            if rho_census[i].sum() > 0:
                rho_census[i] /= rho_census[i].sum()

    # Compute Census-level semantic distances
    print("Computing Census-level semantic distances...")
    sim_sem_census = compute_normalized_overlap(rho_census, K)
    d_sem_census = 1.0 - sim_sem_census

    # Aggregate institutional distances
    print("Aggregating institutional distances...")
    d_inst_census = np.zeros((n_census, n_census))
    for i, ci in enumerate(census_codes):
        for j, cj in enumerate(census_codes):
            onet_i = census_to_onet[int(ci)]
            onet_j = census_to_onet[int(cj)]
            if onet_i and onet_j:
                vals = [d_inst_aligned[oi, oj] for oi in onet_i for oj in onet_j]
                d_inst_census[i, j] = np.mean(vals)

    # Census-level correlation
    upper_sem = d_sem_census[np.triu_indices_from(d_sem_census, k=1)]
    upper_inst = d_inst_census[np.triu_indices_from(d_inst_census, k=1)]
    corr_census = np.corrcoef(upper_sem, upper_inst)[0, 1]

    print("\n" + "=" * 70)
    print("CORRELATION CHECK")
    print("=" * 70)
    print(f"O*NET level ({len(aligned_codes)} occs): corr = {corr_onet:.3f}")
    print(f"Census level ({n_census} occs):  corr = {corr_census:.3f}")
    print(f"Change: {corr_census - corr_onet:+.3f}")

    if corr_census > 0.7:
        print("\n⚠️  WARNING: Signal smearing! Correlation > 0.7")
    elif corr_census > 0.6:
        print("\n⚠️  CAUTION: Correlation approaching problematic levels")
    else:
        print("\n✓  Correlation acceptable")

    # Run simulations
    print("\n" + "=" * 70)
    print("RUNNING SIMULATIONS")
    print("=" * 70)

    def simulate_and_estimate(d_sem, d_inst, alpha, beta, n_workers, switch_rate, n_months, n_reps=5):
        """Run simulation and return recovery results."""
        n_occ = d_sem.shape[0]

        # Precompute transition probabilities
        log_probs = -alpha * d_sem - beta * d_inst
        log_probs_norm = log_probs - np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs_norm)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # Zero out diagonal for switching
        np.fill_diagonal(probs, 0)
        probs = probs / probs.sum(axis=1, keepdims=True)

        alpha_ests, beta_ests = [], []

        for rep in range(n_reps):
            np.random.seed(42 + rep)

            # Generate transitions
            transitions = []
            for _ in range(n_months):
                workers = np.random.randint(0, n_occ, n_workers)
                switchers = np.random.random(n_workers) < switch_rate
                for w in np.where(switchers)[0]:
                    orig = workers[w]
                    dest = np.random.choice(n_occ, p=probs[orig])
                    transitions.append((orig, dest))

            if len(transitions) < 100:
                continue

            # Estimate parameters
            def nll(params):
                a, b = params
                lp = -a * d_sem - b * d_inst
                total = 0.0
                for orig, dest in transitions:
                    lp_row = lp[orig].copy()
                    lp_row[orig] = -np.inf
                    log_Z = np.log(np.exp(lp_row - lp_row.max()).sum()) + lp_row.max()
                    total -= (lp[orig, dest] - log_Z)
                return total / len(transitions)

            res = minimize(nll, [1.0, 1.0], method="L-BFGS-B", bounds=[(0.01, 20), (0.01, 20)])
            if res.success:
                alpha_ests.append(res.x[0])
                beta_ests.append(res.x[1])

        if alpha_ests:
            return {
                "n_transitions": len(transitions),
                "alpha_mean": np.mean(alpha_ests),
                "alpha_std": np.std(alpha_ests),
                "beta_mean": np.mean(beta_ests),
                "beta_std": np.std(beta_ests),
                "alpha_bias": np.mean(alpha_ests) - alpha,
                "beta_bias": np.mean(beta_ests) - beta,
                "success": abs(np.mean(alpha_ests) - alpha) < 0.5 and abs(np.mean(beta_ests) - beta) < 0.5,
            }
        return {"success": False, "error": "All reps failed"}

    # Scenarios
    scenarios = [
        ("Pessimistic (0.5% switch)", 40000, 0.005, 93),
        ("Moderate (0.75% switch)", 45000, 0.0075, 93),
        ("After filter (0.25% switch)", 40000, 0.0025, 93),
        ("Minimum viable", 40000, 0.001, 93),
    ]

    results = []
    for name, workers, rate, months in scenarios:
        print(f"\n--- {name} ---")
        print(f"Workers/mo: {workers:,}, Rate: {rate*100:.2f}%, Months: {months}")

        expected = int(workers * rate * months)
        print(f"Expected transitions: ~{expected:,}")

        res = simulate_and_estimate(d_sem_census, d_inst_census, 2.0, 1.0, workers, rate, months)
        res["scenario"] = name
        res["expected_transitions"] = expected

        if res.get("success"):
            print(f"α̂ = {res['alpha_mean']:.3f} (bias: {res['alpha_bias']:+.3f})")
            print(f"β̂ = {res['beta_mean']:.3f} (bias: {res['beta_bias']:+.3f})")
            print("✓ RECOVERY SUCCESS")
        else:
            print("✗ RECOVERY FAILED")

        results.append(res)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    min_success = None
    for r in results:
        if r.get("success") and (min_success is None or r["expected_transitions"] < min_success):
            min_success = r["expected_transitions"]

    print(f"Minimum transitions for recovery: {min_success if min_success else 'NOT FOUND'}")

    # Save
    output = {
        "aggregation": {
            "onet_occupations": len(aligned_codes),
            "census_occupations": n_census,
        },
        "correlation_check": {
            "onet_level": float(corr_onet),
            "census_level": float(corr_census),
            "signal_smearing": corr_census > 0.7,
            "acceptable": corr_census < 0.6,
        },
        "scenarios": results,
        "minimum_transitions_for_recovery": min_success,
        "go_nogo": {
            "correlation_ok": corr_census < 0.6,
            "recovery_works": min_success is not None and min_success <= 20000,
        }
    }

    with open(OUTPUT_DIR / "census_level_simulation.json", "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved to: {OUTPUT_DIR / 'census_level_simulation.json'}")


if __name__ == "__main__":
    main()
