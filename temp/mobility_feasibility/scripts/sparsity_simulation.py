"""
Task 3: Synthetic Sparsity Simulation

Test whether we can recover α and β from sparse transition data.
Simulates: P(switch to j | from i) ∝ exp(-α * d_sem - β * d_inst)
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from task_space import build_dwa_occupation_measures
from task_space.similarity.overlap import compute_normalized_overlap

OUTPUT_DIR = Path("temp/mobility_feasibility/outputs")


def load_semantic_distances():
    """
    Load semantic similarity and convert to distance.
    d_sem = 1 - normalized_overlap (so d_sem ∈ [0, 1])
    """
    # Load occupation measures
    measures = build_dwa_occupation_measures()
    rho = measures.occupation_matrix  # normalized probability measures

    # Load activity distances
    dist_data = np.load(".cache/artifacts/v1/distances/cosine_920fd8058161b39f.npz")
    d_act = dist_data["distances"]

    # Compute kernel with calibrated bandwidth
    # σ = 0.223 (NN median, per Phase I findings)
    sigma = 0.223
    K = np.exp(-d_act / sigma)

    # Compute normalized overlap (similarity in [0, 1])
    sim_sem = compute_normalized_overlap(rho, K)

    # Convert to distance
    d_sem = 1.0 - sim_sem

    return d_sem, measures.occupation_codes


def load_institutional_distances():
    """Load job zone institutional distances from Task 1."""
    data = np.load(OUTPUT_DIR / "job_zone_matrix.npz", allow_pickle=True)
    d_inst = data["d_inst_matrix"]
    occ_codes = data["occ_codes"]
    return d_inst, occ_codes


def align_distance_matrices(d_sem, codes_sem, d_inst, codes_inst):
    """
    Align the two distance matrices to common occupation set.
    Returns aligned matrices and common codes.
    """
    # Find common occupations
    common = set(codes_sem) & set(codes_inst)
    common_sorted = sorted(common)

    # Get indices
    sem_idx = [np.where(codes_sem == c)[0][0] for c in common_sorted]
    inst_idx = [np.where(codes_inst == c)[0][0] for c in common_sorted]

    # Subset matrices
    d_sem_aligned = d_sem[np.ix_(sem_idx, sem_idx)]
    d_inst_aligned = d_inst[np.ix_(inst_idx, inst_idx)]

    return d_sem_aligned, d_inst_aligned, np.array(common_sorted)


def simulate_transitions(d_sem, d_inst, alpha, beta, n_workers, switch_rate=0.02):
    """
    Simulate occupation transitions.

    Model: P(i → j | switch) ∝ exp(-α * d_sem[i,j] - β * d_inst[i,j])

    Args:
        d_sem: (n_occ, n_occ) semantic distance matrix
        d_inst: (n_occ, n_occ) institutional distance matrix
        alpha: true coefficient on semantic distance
        beta: true coefficient on institutional distance
        n_workers: number of workers to simulate
        switch_rate: probability of switching in a period

    Returns:
        transitions: list of (origin, destination) pairs
    """
    n_occ = d_sem.shape[0]

    # Compute log-transition probabilities (unnormalized)
    log_probs = -alpha * d_sem - beta * d_inst

    # Normalize per origin (softmax)
    log_probs_norm = log_probs - np.max(log_probs, axis=1, keepdims=True)
    probs = np.exp(log_probs_norm)
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Assign workers to occupations (roughly uniform for simplicity)
    worker_occs = np.random.randint(0, n_occ, size=n_workers)

    # Determine who switches
    switches = np.random.random(n_workers) < switch_rate

    # For switchers, sample destination
    transitions = []
    for w in np.where(switches)[0]:
        origin = worker_occs[w]
        # Sample destination (exclude self)
        probs_this = probs[origin].copy()
        probs_this[origin] = 0
        probs_this = probs_this / probs_this.sum()
        dest = np.random.choice(n_occ, p=probs_this)
        transitions.append((origin, dest))

    return transitions


def estimate_params_from_transitions(transitions, d_sem, d_inst):
    """
    Estimate α and β from observed transitions using conditional logit.

    Uses the negative log-likelihood approach.
    """
    n_occ = d_sem.shape[0]

    # Build transition counts
    n_trans = len(transitions)
    if n_trans < 100:
        return None, None, "Too few transitions"

    def neg_log_likelihood(params):
        alpha, beta = params

        # Log-probs
        log_probs = -alpha * d_sem - beta * d_inst

        # Softmax normalization (per origin, excluding self)
        nll = 0.0
        for orig, dest in transitions:
            # Log prob of this transition
            log_p_unnorm = log_probs[orig, dest]

            # Log of partition (excluding self)
            log_probs_row = log_probs[orig].copy()
            log_probs_row[orig] = -np.inf  # exclude self
            log_Z = np.log(np.exp(log_probs_row - log_probs_row.max()).sum()) + log_probs_row.max()

            nll -= (log_p_unnorm - log_Z)

        return nll / n_trans

    # Optimize
    result = minimize(
        neg_log_likelihood,
        x0=[1.0, 1.0],
        method="L-BFGS-B",
        bounds=[(0.01, 20), (0.01, 20)],
    )

    if result.success:
        return result.x[0], result.x[1], None
    else:
        return None, None, result.message


def run_simulation_experiment(
    d_sem, d_inst,
    true_alpha=2.0, true_beta=1.0,
    n_workers_list=[50000, 100000, 500000, 1000000],
    switch_rate=0.02,
    n_reps=10
):
    """
    Run simulation experiment at different sample sizes.
    """
    results = []

    for n_workers in n_workers_list:
        alpha_estimates = []
        beta_estimates = []
        errors = []

        for rep in range(n_reps):
            np.random.seed(42 + rep)
            transitions = simulate_transitions(
                d_sem, d_inst, true_alpha, true_beta,
                n_workers, switch_rate
            )

            n_trans = len(transitions)
            alpha_hat, beta_hat, err = estimate_params_from_transitions(
                transitions, d_sem, d_inst
            )

            if err is None:
                alpha_estimates.append(alpha_hat)
                beta_estimates.append(beta_hat)
            else:
                errors.append(err)

        if len(alpha_estimates) > 0:
            result = {
                "n_workers": n_workers,
                "expected_transitions": int(n_workers * switch_rate),
                "actual_transitions_mean": int(np.mean([len(simulate_transitions(d_sem, d_inst, true_alpha, true_beta, n_workers, switch_rate)) for _ in range(3)])),
                "true_alpha": true_alpha,
                "true_beta": true_beta,
                "alpha_mean": float(np.mean(alpha_estimates)),
                "alpha_std": float(np.std(alpha_estimates)),
                "alpha_bias": float(np.mean(alpha_estimates) - true_alpha),
                "beta_mean": float(np.mean(beta_estimates)),
                "beta_std": float(np.std(beta_estimates)),
                "beta_bias": float(np.mean(beta_estimates) - true_beta),
                "n_successful_reps": len(alpha_estimates),
                "recovery_success": abs(np.mean(alpha_estimates) - true_alpha) < 0.5 and abs(np.mean(beta_estimates) - true_beta) < 0.5,
            }
        else:
            result = {
                "n_workers": n_workers,
                "expected_transitions": int(n_workers * switch_rate),
                "actual_transitions_mean": 0,
                "true_alpha": true_alpha,
                "true_beta": true_beta,
                "error": "All replications failed",
                "errors": errors[:3],
                "recovery_success": False,
            }

        results.append(result)
        print(f"n_workers={n_workers}: α̂={result.get('alpha_mean', 'N/A'):.3f}, β̂={result.get('beta_mean', 'N/A'):.3f}")

    return results


def estimate_cps_feasibility(results, cps_person_months=9_600_000, switch_rate=0.02):
    """
    Assess feasibility given CPS sample size.
    """
    cps_expected_transitions = cps_person_months * switch_rate

    # Find the minimum sample size where recovery is successful
    min_success = None
    for r in results:
        if r.get("recovery_success", False):
            min_success = r["n_workers"]
            break

    if min_success is None:
        return {
            "cps_expected_transitions": int(cps_expected_transitions),
            "min_sample_for_recovery": "Not found in tested range",
            "feasibility": "UNCERTAIN - may need more data",
        }

    min_transitions = int(min_success * switch_rate)

    return {
        "cps_expected_transitions": int(cps_expected_transitions),
        "min_workers_for_recovery": min_success,
        "min_transitions_for_recovery": min_transitions,
        "cps_has_sufficient_data": cps_expected_transitions > min_transitions,
        "margin": cps_expected_transitions / min_transitions if min_transitions > 0 else float("inf"),
        "feasibility": "FEASIBLE" if cps_expected_transitions > min_transitions * 1.5 else "MARGINAL",
    }


def main():
    print("=" * 60)
    print("SPARSITY SIMULATION: CAN WE RECOVER α AND β?")
    print("=" * 60)

    # Load data
    print("\nLoading semantic distances...")
    d_sem, codes_sem = load_semantic_distances()
    print(f"  Semantic distance matrix: {d_sem.shape}")

    print("\nLoading institutional distances...")
    d_inst, codes_inst = load_institutional_distances()
    print(f"  Institutional distance matrix: {d_inst.shape}")

    # Align
    print("\nAligning matrices to common occupations...")
    d_sem, d_inst, common_codes = align_distance_matrices(
        d_sem, np.array(codes_sem), d_inst, codes_inst
    )
    print(f"  Common occupations: {len(common_codes)}")
    print(f"  Aligned matrices: {d_sem.shape}")

    # Distance statistics
    print("\n--- Distance Statistics ---")
    upper_sem = d_sem[np.triu_indices_from(d_sem, k=1)]
    upper_inst = d_inst[np.triu_indices_from(d_inst, k=1)]
    print(f"  Semantic distance: mean={upper_sem.mean():.3f}, std={upper_sem.std():.3f}")
    print(f"  Institutional distance: mean={upper_inst.mean():.3f}, std={upper_inst.std():.3f}")
    print(f"  Correlation: {np.corrcoef(upper_sem, upper_inst)[0,1]:.3f}")

    # Run simulation
    print("\n" + "=" * 60)
    print("RUNNING SIMULATION EXPERIMENT")
    print("=" * 60)
    print("\nTrue parameters: α=2.0, β=1.0")
    print("Testing sample sizes: 50k, 100k, 500k, 1M workers")
    print("Switch rate: 2%")
    print()

    results = run_simulation_experiment(
        d_sem, d_inst,
        true_alpha=2.0, true_beta=1.0,
        n_workers_list=[50_000, 100_000, 500_000, 1_000_000],
        switch_rate=0.02,
        n_reps=5  # Fewer reps for speed
    )

    # CPS feasibility
    print("\n" + "=" * 60)
    print("CPS FEASIBILITY ASSESSMENT")
    print("=" * 60)
    feasibility = estimate_cps_feasibility(results)
    print(f"\nCPS expected transitions (9.6M person-months × 2%): {feasibility['cps_expected_transitions']:,}")
    print(f"Minimum transitions for recovery: {feasibility.get('min_transitions_for_recovery', 'Unknown'):,}")
    print(f"Margin: {feasibility.get('margin', 0):.1f}x")
    print(f"\nFEASIBILITY: {feasibility['feasibility']}")

    # Save results
    output = {
        "simulation_params": {
            "true_alpha": 2.0,
            "true_beta": 1.0,
            "switch_rate": 0.02,
            "n_occupations": len(common_codes),
        },
        "distance_stats": {
            "d_sem_mean": float(upper_sem.mean()),
            "d_sem_std": float(upper_sem.std()),
            "d_inst_mean": float(upper_inst.mean()),
            "d_inst_std": float(upper_inst.std()),
            "correlation": float(np.corrcoef(upper_sem, upper_inst)[0, 1]),
        },
        "results_by_sample_size": results,
        "cps_feasibility": feasibility,
        "recommendation": feasibility["feasibility"],
    }

    output_path = OUTPUT_DIR / "sparsity_simulation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
