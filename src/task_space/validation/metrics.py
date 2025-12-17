"""
Performance battery metrics for mobility model evaluation (MS8).

Implements three core metrics required by Methodology Standard 8:
1. Mean Percentile Rank (MPR) - E2 ranking adequacy
2. Realized Cumulative Mass (RCM) - consideration set inclusion
3. Effective Consideration Set (N_eff) - dispersion diagnostic

These metrics evaluate how well predicted probabilities rank the
realized destination, providing complementary evidence to ΔLL (E1).

References:
    - LEDGER.md MS8: Performance Battery Requirement
    - paper/main.tex Section 5.5
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class PerformanceBatteryResult:
    """
    Complete performance battery results.

    Attributes:
        mpr_mean: Mean Percentile Rank (1.0 = perfect, 0.5 = random)
        mpr_median: Median Percentile Rank
        mpr_std: Standard deviation of percentile ranks
        mpr_p25: 25th percentile of percentile ranks
        mpr_p75: 75th percentile of percentile ranks
        rcm_mean: Mean Realized Cumulative Mass
        rcm_median: Median RCM
        rcm_std: Standard deviation of RCM
        n_eff_mean: Mean effective consideration set size
        n_eff_median: Median effective consideration set size
        n_eff_std: Standard deviation of N_eff
        n_destinations: Total number of possible destinations
        n_eff_ratio: Mean N_eff / n_destinations (diffuseness)
        n_transitions: Number of transitions evaluated
        top_k_note: Explanation of why top-k is/isn't reported
    """
    # MPR metrics
    mpr_mean: float
    mpr_median: float
    mpr_std: float
    mpr_p25: float
    mpr_p75: float

    # RCM metrics
    rcm_mean: float
    rcm_median: float
    rcm_std: float

    # Dispersion metrics
    n_eff_mean: float
    n_eff_median: float
    n_eff_std: float
    n_destinations: int
    n_eff_ratio: float

    # Metadata
    n_transitions: int
    top_k_note: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "mpr": {
                "mean": self.mpr_mean,
                "median": self.mpr_median,
                "std": self.mpr_std,
                "p25": self.mpr_p25,
                "p75": self.mpr_p75,
            },
            "rcm": {
                "mean": self.rcm_mean,
                "median": self.rcm_median,
                "std": self.rcm_std,
            },
            "dispersion": {
                "n_eff_mean": self.n_eff_mean,
                "n_eff_median": self.n_eff_median,
                "n_eff_std": self.n_eff_std,
                "n_destinations": self.n_destinations,
                "n_eff_ratio": self.n_eff_ratio,
            },
            "n_transitions": self.n_transitions,
            "top_k_note": self.top_k_note,
        }


def compute_mean_percentile_rank(
    probabilities: np.ndarray,
    realized_idx: np.ndarray,
) -> dict:
    """
    Compute Mean Percentile Rank of realized destinations.

    For each transition, rank all destinations by predicted probability.
    Convert the realized destination's rank to a percentile (1 = best, 0 = worst).

    Args:
        probabilities: (n_transitions, n_destinations) probability matrix
        realized_idx: (n_transitions,) index of realized destination for each transition

    Returns:
        dict with 'mpr_mean', 'mpr_median', 'mpr_std', 'mpr_p25', 'mpr_p75'

    Interpretation:
        MPR = 1.0: Realized destination always ranked first
        MPR = 0.5: Random ranking (baseline)
        MPR > 0.5: Model places realized destinations in high-probability region
    """
    n_transitions, n_destinations = probabilities.shape
    percentile_ranks = np.zeros(n_transitions)

    for i in range(n_transitions):
        # Rank destinations by probability (descending)
        # argsort twice gives ranks; +1 for 1-indexed ranks
        ranks = np.argsort(np.argsort(-probabilities[i])) + 1
        realized_rank = ranks[realized_idx[i]]
        # Convert to percentile (1 = best, 0 = worst)
        percentile = 1 - (realized_rank - 1) / (n_destinations - 1)
        percentile_ranks[i] = percentile

    return {
        'mpr_mean': float(np.mean(percentile_ranks)),
        'mpr_median': float(np.median(percentile_ranks)),
        'mpr_std': float(np.std(percentile_ranks)),
        'mpr_p25': float(np.percentile(percentile_ranks, 25)),
        'mpr_p75': float(np.percentile(percentile_ranks, 75)),
        '_percentile_ranks': percentile_ranks,  # For diagnostic use
    }


def compute_realized_cumulative_mass(
    probabilities: np.ndarray,
    realized_idx: np.ndarray,
) -> dict:
    """
    Compute Realized Cumulative Mass.

    For each transition, RCM = sum of probability mass at or above the
    realized destination's probability level. This measures whether the
    model's "consideration set" includes the realized choice.

    Args:
        probabilities: (n_transitions, n_destinations) probability matrix
        realized_idx: (n_transitions,) index of realized destination

    Returns:
        dict with 'rcm_mean', 'rcm_median', 'rcm_std'

    Interpretation:
        RCM close to 0: Model concentrates mass on realized destination
        RCM close to 1: Realized destination among model's top choices
        Low RCM with low N_eff: Model is confident and correct
        High RCM with high N_eff: Model is diffuse (many similar destinations)
    """
    n_transitions = probabilities.shape[0]
    rcm_values = np.zeros(n_transitions)

    for i in range(n_transitions):
        realized_prob = probabilities[i, realized_idx[i]]
        # Sum probability mass at or above realized destination's probability
        rcm = np.sum(probabilities[i, probabilities[i] >= realized_prob])
        rcm_values[i] = rcm

    return {
        'rcm_mean': float(np.mean(rcm_values)),
        'rcm_median': float(np.median(rcm_values)),
        'rcm_std': float(np.std(rcm_values)),
        '_rcm_values': rcm_values,  # For diagnostic use
    }


def compute_effective_consideration_set(
    probabilities: np.ndarray,
) -> dict:
    """
    Compute effective consideration set size via entropy.

    N_eff = exp(entropy) is the "effective number of destinations" under
    the probability distribution. This explains why top-k overlap is
    structurally inappropriate for diffuse distributions.

    Args:
        probabilities: (n_transitions, n_destinations) probability matrix

    Returns:
        dict with 'n_eff_mean', 'n_eff_median', 'n_eff_std', 'n_destinations', 'n_eff_ratio'

    Interpretation:
        N_eff ≈ J (n_destinations): Uniform/diffuse distribution
        N_eff << J: Concentrated distribution (few likely destinations)
        N_eff ratio ≈ 1: Top-k metrics inappropriate (too diffuse)
        N_eff ratio << 1: Top-k metrics may be informative
    """
    n_transitions, n_destinations = probabilities.shape
    n_eff_values = np.zeros(n_transitions)

    for i in range(n_transitions):
        p = probabilities[i]
        # Filter out zeros to avoid log(0)
        p_nonzero = p[p > 1e-15]
        if len(p_nonzero) == 0:
            # Degenerate case: all probability on one destination
            n_eff_values[i] = 1.0
        else:
            entropy = -np.sum(p_nonzero * np.log(p_nonzero))
            n_eff_values[i] = np.exp(entropy)

    return {
        'n_eff_mean': float(np.mean(n_eff_values)),
        'n_eff_median': float(np.median(n_eff_values)),
        'n_eff_std': float(np.std(n_eff_values)),
        'n_destinations': n_destinations,
        'n_eff_ratio': float(np.mean(n_eff_values) / n_destinations),
        '_n_eff_values': n_eff_values,  # For diagnostic use
    }


def compute_full_destination_probabilities(
    transitions_df,
    d_sem_matrix: np.ndarray,
    d_inst_matrix: np.ndarray,
    occ_codes: List[int],
    alpha: float,
    beta: float,
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute predicted probabilities over ALL destinations for each transition.

    Uses fitted model coefficients to compute:
        P(j | i) = exp(α * (-d_sem) + β * (-d_inst)) / Σ_k exp(...)

    Args:
        transitions_df: DataFrame with origin/destination columns
        d_sem_matrix: (n_occ, n_occ) semantic distance matrix
        d_inst_matrix: (n_occ, n_occ) institutional distance matrix
        occ_codes: Occupation codes (row/column labels)
        alpha: Fitted semantic distance coefficient
        beta: Fitted institutional distance coefficient
        origin_col: Column name for origin occupation
        dest_col: Column name for destination occupation

    Returns:
        probabilities: (n_transitions, n_destinations) probability matrix
        realized_idx: (n_transitions,) index of realized destination
        valid_transitions: (n_transitions,) boolean mask for valid transitions
    """
    occ_to_idx = {occ: i for i, occ in enumerate(occ_codes)}
    n_destinations = len(occ_codes)

    # Pre-allocate
    n_rows = len(transitions_df)
    probabilities = np.zeros((n_rows, n_destinations))
    realized_idx = np.zeros(n_rows, dtype=int)
    valid_mask = np.zeros(n_rows, dtype=bool)

    for row_idx, (_, row) in enumerate(transitions_df.iterrows()):
        origin = int(row[origin_col])
        dest = int(row[dest_col])

        # Skip if codes not in distance matrix
        if origin not in occ_to_idx or dest not in occ_to_idx:
            continue

        origin_idx = occ_to_idx[origin]
        dest_idx = occ_to_idx[dest]

        # Compute utility for all destinations from this origin
        # U_j = α * (-d_sem[i,j]) + β * (-d_inst[i,j])
        utilities = (
            alpha * (-d_sem_matrix[origin_idx, :]) +
            beta * (-d_inst_matrix[origin_idx, :])
        )

        # Softmax to get probabilities
        # Subtract max for numerical stability
        utilities = utilities - np.max(utilities)
        exp_utilities = np.exp(utilities)
        probs = exp_utilities / np.sum(exp_utilities)

        probabilities[row_idx] = probs
        realized_idx[row_idx] = dest_idx
        valid_mask[row_idx] = True

    # Filter to valid transitions only
    probabilities = probabilities[valid_mask]
    realized_idx = realized_idx[valid_mask]

    return probabilities, realized_idx, valid_mask


def compute_performance_battery(
    probabilities: np.ndarray,
    realized_idx: np.ndarray,
    n_eff_threshold: float = 0.3,
) -> PerformanceBatteryResult:
    """
    Compute complete performance battery (MS8 compliant).

    Args:
        probabilities: (n_transitions, n_destinations) probability matrix
        realized_idx: (n_transitions,) index of realized destination
        n_eff_threshold: If n_eff_ratio > threshold, top-k is inappropriate

    Returns:
        PerformanceBatteryResult with all metrics and top-k appropriateness note
    """
    n_transitions, n_destinations = probabilities.shape

    # Compute all metrics
    mpr_results = compute_mean_percentile_rank(probabilities, realized_idx)
    rcm_results = compute_realized_cumulative_mass(probabilities, realized_idx)
    neff_results = compute_effective_consideration_set(probabilities)

    # Determine top-k appropriateness
    n_eff_ratio = neff_results['n_eff_ratio']
    if n_eff_ratio > n_eff_threshold:
        top_k_note = (
            f"Top-k overlap NOT reported: N_eff/J = {n_eff_ratio:.2f} > {n_eff_threshold}. "
            f"Mean effective consideration set = {neff_results['n_eff_mean']:.1f} destinations. "
            f"Distribution too diffuse for top-k to be meaningful."
        )
    else:
        top_k_note = (
            f"Top-k overlap may be reported: N_eff/J = {n_eff_ratio:.2f} ≤ {n_eff_threshold}. "
            f"Mean effective consideration set = {neff_results['n_eff_mean']:.1f} destinations."
        )

    return PerformanceBatteryResult(
        mpr_mean=mpr_results['mpr_mean'],
        mpr_median=mpr_results['mpr_median'],
        mpr_std=mpr_results['mpr_std'],
        mpr_p25=mpr_results['mpr_p25'],
        mpr_p75=mpr_results['mpr_p75'],
        rcm_mean=rcm_results['rcm_mean'],
        rcm_median=rcm_results['rcm_median'],
        rcm_std=rcm_results['rcm_std'],
        n_eff_mean=neff_results['n_eff_mean'],
        n_eff_median=neff_results['n_eff_median'],
        n_eff_std=neff_results['n_eff_std'],
        n_destinations=neff_results['n_destinations'],
        n_eff_ratio=neff_results['n_eff_ratio'],
        n_transitions=n_transitions,
        top_k_note=top_k_note,
    )


# =============================================================================
# Testing utilities
# =============================================================================

def _test_metrics_on_synthetic_data():
    """Test metrics on synthetic data for verification."""
    np.random.seed(42)
    n_transitions = 100
    n_destinations = 50

    # Case 1: Perfect prediction (realized always has highest probability)
    probs_perfect = np.random.rand(n_transitions, n_destinations)
    realized_perfect = np.random.randint(0, n_destinations, n_transitions)
    for i in range(n_transitions):
        probs_perfect[i, realized_perfect[i]] = 1000  # Very high
    probs_perfect = probs_perfect / probs_perfect.sum(axis=1, keepdims=True)

    mpr_perfect = compute_mean_percentile_rank(probs_perfect, realized_perfect)
    assert mpr_perfect['mpr_mean'] > 0.99, "Perfect prediction should have MPR ≈ 1.0"

    # Case 2: Random prediction (uniform probabilities)
    probs_uniform = np.ones((n_transitions, n_destinations)) / n_destinations
    realized_random = np.random.randint(0, n_destinations, n_transitions)

    mpr_random = compute_mean_percentile_rank(probs_uniform, realized_random)
    assert 0.4 < mpr_random['mpr_mean'] < 0.6, "Random prediction should have MPR ≈ 0.5"

    neff_uniform = compute_effective_consideration_set(probs_uniform)
    assert abs(neff_uniform['n_eff_mean'] - n_destinations) < 1, "Uniform should have N_eff ≈ J"

    # Case 3: Concentrated prediction
    probs_concentrated = np.zeros((n_transitions, n_destinations))
    for i in range(n_transitions):
        top_k = 3
        idxs = np.random.choice(n_destinations, top_k, replace=False)
        probs_concentrated[i, idxs] = 1.0 / top_k

    neff_concentrated = compute_effective_consideration_set(probs_concentrated)
    assert neff_concentrated['n_eff_mean'] < 5, "Concentrated should have low N_eff"

    print("All synthetic tests passed!")
    return {
        'perfect': mpr_perfect,
        'random': mpr_random,
        'neff_uniform': neff_uniform,
        'neff_concentrated': neff_concentrated,
    }


if __name__ == "__main__":
    results = _test_metrics_on_synthetic_data()
    print("\nSynthetic test results:")
    print(f"  Perfect MPR: {results['perfect']['mpr_mean']:.4f}")
    print(f"  Random MPR: {results['random']['mpr_mean']:.4f}")
    print(f"  Uniform N_eff: {results['neff_uniform']['n_eff_mean']:.1f}")
    print(f"  Concentrated N_eff: {results['neff_concentrated']['n_eff_mean']:.1f}")
