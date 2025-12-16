"""
Shock integration test infrastructure for AI-exposed occupation analysis.

Evaluates whether task-space geometry improves reallocation prediction
for occupations with high AI exposure (AIOE scores).

Phase 0.7a: Initial validation framework for prospective shock analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import json

import numpy as np
import pandas as pd

from task_space.data.aioe import load_aioe, AIOEData
from task_space.mobility.census_crosswalk import load_census_onet_crosswalk


# =============================================================================
# AIOE Data Functions
# =============================================================================


def get_aioe_by_soc_dataframe(
    use_lm: bool = True,
    aioe_data: Optional[AIOEData] = None,
) -> pd.DataFrame:
    """
    Load AIOE scores mapped to SOC codes.

    Args:
        use_lm: If True, use Language Modeling AIOE (for generative AI).
        aioe_data: Pre-loaded AIOEData. If None, loads fresh.

    Returns:
        DataFrame with columns: [soc_code, aioe_score]
        soc_code is 6-digit SOC (e.g., "11-1011")
    """
    if aioe_data is None:
        aioe_data = load_aioe(include_lm=use_lm)

    col = "lm_aioe" if use_lm and aioe_data.has_lm_aioe else "aioe"

    df = aioe_data.scores[["soc_code", col]].copy()
    df = df.rename(columns={col: "aioe_score"})
    df = df.dropna(subset=["aioe_score"])

    return df


def map_aioe_to_census(
    aioe_df: pd.DataFrame,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """
    Map AIOE scores from SOC-6 to Census 2010 codes.

    Args:
        aioe_df: DataFrame from get_aioe_by_soc_dataframe()
        aggregation: How to aggregate when multiple SOC codes map to
                     one Census code. Options: "mean", "max".

    Returns:
        DataFrame with columns: [census_code, aioe_score, n_soc_codes]
    """
    xwalk = load_census_onet_crosswalk()

    # Build SOC-6 to Census mapping
    # The crosswalk has onet_soc (8-char like "11-1011.00") and soc_6digit
    xwalk_df = xwalk.crosswalk_df.copy()
    xwalk_df = xwalk_df[xwalk_df["matched"] == True]

    # Extract 6-digit SOC from onet_soc or use soc_6digit column
    if "soc_6digit" in xwalk_df.columns:
        soc6_to_census = xwalk_df.groupby("soc_6digit")["census_2010"].first().to_dict()
    else:
        # Extract from onet_soc: "11-1011.00" -> "11-1011"
        xwalk_df["soc_6"] = xwalk_df["onet_soc"].str.slice(0, 7)
        soc6_to_census = xwalk_df.groupby("soc_6")["census_2010"].first().to_dict()

    # Map AIOE scores
    aioe_df = aioe_df.copy()
    aioe_df["census_code"] = aioe_df["soc_code"].map(soc6_to_census)

    # Filter to matched occupations
    matched = aioe_df.dropna(subset=["census_code"]).copy()
    matched["census_code"] = matched["census_code"].astype(int)

    # Aggregate by Census code
    if aggregation == "mean":
        result = matched.groupby("census_code").agg(
            aioe_score=("aioe_score", "mean"),
            n_soc_codes=("soc_code", "count")
        ).reset_index()
    elif aggregation == "max":
        result = matched.groupby("census_code").agg(
            aioe_score=("aioe_score", "max"),
            n_soc_codes=("soc_code", "count")
        ).reset_index()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return result


# =============================================================================
# Geometry-AIOE Correlation
# =============================================================================


def compute_aioe_geometry_correlations(
    aioe_census_df: pd.DataFrame,
    wasserstein_matrix: np.ndarray,
    census_codes: List[int],
) -> dict:
    """
    Preliminary redundancy check: correlation between AIOE and geometry.

    If AIOE is highly correlated with Wasserstein distances, the shock
    integration test may not provide additional signal beyond what
    mobility analysis already captures.

    Args:
        aioe_census_df: DataFrame with census_code and aioe_score
        wasserstein_matrix: (n_occ, n_occ) Wasserstein distance matrix
        census_codes: Census codes (row/column labels for matrix)

    Returns:
        Dict with:
            aioe_wasserstein_corr: Correlation between AIOE and mean distance
            aioe_coverage: Fraction of Census codes with AIOE scores
    """
    census_set = set(census_codes)
    aioe_dict = dict(zip(aioe_census_df["census_code"], aioe_census_df["aioe_score"]))

    # Coverage
    matched_codes = set(aioe_dict.keys()) & census_set
    coverage = len(matched_codes) / len(census_set) if census_set else 0.0

    # For each occupation with AIOE, compute mean Wasserstein distance
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    aioe_scores = []
    mean_distances = []

    for code in matched_codes:
        if code in code_to_idx:
            idx = code_to_idx[code]
            mean_dist = np.mean(wasserstein_matrix[idx, :])
            aioe_scores.append(aioe_dict[code])
            mean_distances.append(mean_dist)

    # Compute correlation
    if len(aioe_scores) > 2:
        corr = np.corrcoef(aioe_scores, mean_distances)[0, 1]
    else:
        corr = np.nan

    return {
        "aioe_wasserstein_corr": float(corr) if not np.isnan(corr) else None,
        "aioe_coverage": float(coverage),
        "n_matched_occupations": len(matched_codes),
    }


# =============================================================================
# Transition Partitioning
# =============================================================================


def partition_transitions_by_exposure(
    transitions_df: pd.DataFrame,
    aioe_census_df: pd.DataFrame,
    quartile: float = 0.75,
    origin_col: str = "origin_occ",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split transitions by origin occupation AIOE exposure.

    Args:
        transitions_df: DataFrame with origin_occ, dest_occ columns
        aioe_census_df: DataFrame with census_code, aioe_score
        quartile: Threshold quartile for "exposed" (default 0.75 = top quartile)
        origin_col: Column name for origin occupation

    Returns:
        Tuple of (exposed_df, unexposed_df) where exposed = top quartile AIOE
    """
    # Create AIOE lookup
    aioe_dict = dict(zip(aioe_census_df["census_code"], aioe_census_df["aioe_score"]))

    # Map AIOE to origins
    transitions_df = transitions_df.copy()
    transitions_df["origin_aioe"] = transitions_df[origin_col].map(aioe_dict)

    # Filter to transitions with AIOE data
    with_aioe = transitions_df.dropna(subset=["origin_aioe"])

    # Compute threshold
    threshold = with_aioe["origin_aioe"].quantile(quartile)

    # Partition
    exposed = with_aioe[with_aioe["origin_aioe"] >= threshold].copy()
    unexposed = with_aioe[with_aioe["origin_aioe"] < threshold].copy()

    return exposed, unexposed


# =============================================================================
# Baseline Computation
# =============================================================================


def compute_historical_baseline(
    transitions_df: pd.DataFrame,
    census_codes: List[int],
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
    smoothing: float = 1e-10,
) -> np.ndarray:
    """
    Compute historical transition probability matrix from training data.

    P[i,j] = count(i->j) / count(i->*)

    Args:
        transitions_df: DataFrame with origin and destination columns
        census_codes: List of Census codes (defines matrix dimensions)
        origin_col: Column name for origin occupation
        dest_col: Column name for destination occupation
        smoothing: Small constant to avoid division by zero

    Returns:
        (n_occ, n_occ) probability matrix where rows sum to 1
    """
    n_occ = len(census_codes)
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    # Count transitions
    count_matrix = np.zeros((n_occ, n_occ))

    for _, row in transitions_df.iterrows():
        origin = int(row[origin_col])
        dest = int(row[dest_col])

        if origin in code_to_idx and dest in code_to_idx:
            i = code_to_idx[origin]
            j = code_to_idx[dest]
            count_matrix[i, j] += 1

    # Normalize rows to get probabilities
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    prob_matrix = count_matrix / (row_sums + smoothing)

    # For origins with no observed transitions, use uniform
    zero_rows = (row_sums.flatten() == 0)
    prob_matrix[zero_rows, :] = 1.0 / n_occ

    return prob_matrix


def compute_uniform_baseline(n_occupations: int) -> np.ndarray:
    """
    Compute uniform probability matrix (naive baseline).

    Args:
        n_occupations: Number of occupations

    Returns:
        (n_occ, n_occ) matrix with all entries = 1/n_occ
    """
    return np.ones((n_occupations, n_occupations)) / n_occupations


# =============================================================================
# Model Evaluation
# =============================================================================


def compute_model_probabilities(
    transitions_df: pd.DataFrame,
    wasserstein_matrix: np.ndarray,
    inst_matrix: np.ndarray,
    census_codes: List[int],
    alpha: float,
    beta: float,
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
) -> np.ndarray:
    """
    Compute choice probabilities from fitted conditional logit model.

    P(j|i) ∝ exp(-α * d_sem(i,j) - β * d_inst(i,j))

    Args:
        transitions_df: DataFrame (used to identify relevant origin-dest pairs)
        wasserstein_matrix: Semantic distance matrix
        inst_matrix: Institutional distance matrix
        census_codes: Census code labels
        alpha: Semantic distance coefficient (from conditional logit)
        beta: Institutional distance coefficient
        origin_col: Column name for origin occupation
        dest_col: Column name for destination occupation

    Returns:
        (n_occ, n_occ) probability matrix
    """
    n_occ = len(census_codes)

    # Compute utility: U[i,j] = -alpha * d_sem[i,j] - beta * d_inst[i,j]
    utility = -alpha * wasserstein_matrix - beta * inst_matrix

    # Convert to probabilities via softmax over destinations
    # P(j|i) = exp(U[i,j]) / sum_k exp(U[i,k])
    exp_utility = np.exp(utility - utility.max(axis=1, keepdims=True))  # Numerical stability
    prob_matrix = exp_utility / exp_utility.sum(axis=1, keepdims=True)

    return prob_matrix


def evaluate_model_on_holdout(
    model_probs: np.ndarray,
    historical_probs: np.ndarray,
    holdout_df: pd.DataFrame,
    census_codes: List[int],
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
) -> dict:
    """
    Compare model predictions to baselines on holdout set.

    Args:
        model_probs: (n_occ, n_occ) geometry-based probability matrix
        historical_probs: (n_occ, n_occ) historical baseline probabilities
        holdout_df: DataFrame with held-out transitions
        census_codes: Census code labels
        origin_col: Column name for origin occupation
        dest_col: Column name for destination occupation

    Returns:
        Dict with:
            geometry_ll: Log-likelihood under geometry model
            baseline_historical_ll: Log-likelihood under historical baseline
            baseline_uniform_ll: Log-likelihood under uniform baseline
            geometry_top5_acc: Top-5 accuracy for geometry model
            baseline_historical_top5_acc: Top-5 accuracy for historical
    """
    code_to_idx = {c: i for i, c in enumerate(census_codes)}
    n_occ = len(census_codes)

    uniform_prob = 1.0 / n_occ

    # Track metrics
    geometry_ll = 0.0
    historical_ll = 0.0
    uniform_ll = 0.0

    geometry_top5_correct = 0
    historical_top5_correct = 0
    n_valid = 0

    for _, row in holdout_df.iterrows():
        origin = int(row[origin_col])
        dest = int(row[dest_col])

        if origin not in code_to_idx or dest not in code_to_idx:
            continue

        i = code_to_idx[origin]
        j = code_to_idx[dest]

        # Log-likelihoods (with floor to avoid log(0))
        eps = 1e-15
        geometry_ll += np.log(max(model_probs[i, j], eps))
        historical_ll += np.log(max(historical_probs[i, j], eps))
        uniform_ll += np.log(uniform_prob)

        # Top-5 accuracy
        geo_top5 = np.argsort(-model_probs[i, :])[:5]
        hist_top5 = np.argsort(-historical_probs[i, :])[:5]

        if j in geo_top5:
            geometry_top5_correct += 1
        if j in hist_top5:
            historical_top5_correct += 1

        n_valid += 1

    if n_valid == 0:
        return {
            "geometry_ll": 0.0,
            "baseline_historical_ll": 0.0,
            "baseline_uniform_ll": 0.0,
            "geometry_top5_acc": 0.0,
            "baseline_historical_top5_acc": 0.0,
            "n_evaluated": 0,
        }

    return {
        "geometry_ll": float(geometry_ll),
        "baseline_historical_ll": float(historical_ll),
        "baseline_uniform_ll": float(uniform_ll),
        "geometry_top5_acc": float(geometry_top5_correct / n_valid),
        "baseline_historical_top5_acc": float(historical_top5_correct / n_valid),
        "n_evaluated": n_valid,
    }


# =============================================================================
# Results Schema
# =============================================================================


@dataclass
class ShockIntegrationResult:
    """
    Results from shock integration test.

    Follows the specified JSON schema for outputs/experiments/.
    """
    version: str
    preliminary_checks: dict
    sample_sizes: dict
    metrics: dict
    deltas: dict
    verdict: str
    aggregation: Optional[str] = None  # Set if SOC-3 aggregation was needed

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "version": self.version,
            "preliminary_checks": self.preliminary_checks,
            "sample_sizes": self.sample_sizes,
            "metrics": self.metrics,
            "deltas": self.deltas,
            "verdict": self.verdict,
        }
        if self.aggregation:
            result["aggregation"] = self.aggregation
        return result

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_verdict(geometry_ll: float, historical_ll: float, uniform_ll: float) -> str:
    """
    Determine verdict based on log-likelihood comparisons.

    Args:
        geometry_ll: Log-likelihood under geometry model
        historical_ll: Log-likelihood under historical baseline
        uniform_ll: Log-likelihood under uniform baseline

    Returns:
        One of: "proceed_strong", "proceed_cautious", "proceed_validated", "stop"

    Logic:
        - "proceed_strong": geometry beats historical by >100 (geometry adds value)
        - "proceed_validated": geometry and historical are close (|delta| < 100)
        - "stop": geometry is close to uniform (uninformative model)
        - "proceed_cautious": historical beats geometry, but geometry beats uniform
    """
    delta_vs_historical = geometry_ll - historical_ll
    delta_vs_uniform = geometry_ll - uniform_ll

    # First check if geometry is uninformative (close to uniform baseline)
    if abs(delta_vs_uniform) < 100:
        return "stop"

    # Geometry beats historical significantly
    if delta_vs_historical > 100:
        return "proceed_strong"

    # Geometry and historical are close
    if abs(delta_vs_historical) < 100:
        return "proceed_validated"

    # Historical beats geometry, but geometry still beats uniform
    if historical_ll > geometry_ll and delta_vs_uniform > 100:
        return "proceed_cautious"

    # Default: if we get here, something unexpected - be cautious
    return "proceed_cautious"
