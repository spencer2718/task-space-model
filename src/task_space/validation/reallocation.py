"""
Counterfactual reallocation predictions for AI-exposed occupations.

Uses task-space geometry to predict where displaced workers would move,
weighted by current employment in exposed occupations.

Phase 0.7c: Counterfactual reallocation framework.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import json

import numpy as np
import pandas as pd

from task_space.validation.shock_integration import (
    get_aioe_by_soc_dataframe,
    map_aioe_to_census,
)
from task_space.data.oes import load_oes_year
from task_space.mobility.census_crosswalk import load_census_onet_crosswalk


# =============================================================================
# Employment Data Functions
# =============================================================================


def load_employment_by_census(
    year: int = 2023,
) -> pd.DataFrame:
    """
    Load OES employment aggregated to Census 2010 codes.

    Args:
        year: OES survey year

    Returns:
        DataFrame with columns: [census_code, tot_emp, n_soc_codes]
    """
    oes_df = load_oes_year(year)

    if "TOT_EMP" not in oes_df.columns:
        raise ValueError(f"Employment data (TOT_EMP) not available for {year}")

    # Map SOC to Census
    xwalk = load_census_onet_crosswalk()
    xwalk_df = xwalk.crosswalk_df[xwalk.crosswalk_df["matched"] == True].copy()
    xwalk_df["soc_6"] = xwalk_df["onet_soc"].str[:7]
    soc_to_census = xwalk_df.groupby("soc_6")["census_2010"].first().to_dict()

    oes_df = oes_df.copy()
    oes_df["census_code"] = oes_df["OCC_CODE"].map(soc_to_census)

    # Filter to mapped occupations
    mapped = oes_df.dropna(subset=["census_code", "TOT_EMP"]).copy()
    mapped["census_code"] = mapped["census_code"].astype(int)

    # Aggregate to Census level
    result = mapped.groupby("census_code").agg(
        tot_emp=("TOT_EMP", "sum"),
        n_soc_codes=("OCC_CODE", "count"),
    ).reset_index()

    return result


def get_exposed_occupations(
    aioe_census_df: pd.DataFrame,
    employment_df: pd.DataFrame,
    quartile: float = 0.75,
) -> pd.DataFrame:
    """
    Get exposed occupations (top quartile AIOE) with employment data.

    Args:
        aioe_census_df: DataFrame with census_code, aioe_score
        employment_df: DataFrame with census_code, tot_emp
        quartile: Threshold quartile for "exposed"

    Returns:
        DataFrame with columns: [census_code, aioe_score, tot_emp]
        Only includes occupations above AIOE threshold with employment data.
    """
    # Compute threshold
    threshold = aioe_census_df["aioe_score"].quantile(quartile)

    # Filter to exposed
    exposed = aioe_census_df[aioe_census_df["aioe_score"] >= threshold].copy()

    # Merge with employment
    merged = exposed.merge(
        employment_df[["census_code", "tot_emp"]],
        on="census_code",
        how="inner",
    )

    return merged, threshold


# =============================================================================
# Destination Probability Computation
# =============================================================================


def compute_destination_probabilities(
    wasserstein_matrix: np.ndarray,
    inst_matrix: np.ndarray,
    census_codes: List[int],
    gamma_sem: float,
    gamma_inst: float,
    exclude_self: bool = True,
) -> np.ndarray:
    """
    Compute choice probabilities P(j|i) from fitted model coefficients.

    Uses conditional logit model:
        P(j|i) proportional to exp(-gamma_sem * d_sem(i,j) - gamma_inst * d_inst(i,j))

    Args:
        wasserstein_matrix: (n_occ, n_occ) semantic distance matrix
        inst_matrix: (n_occ, n_occ) institutional distance matrix
        census_codes: Census codes (row/column labels)
        gamma_sem: Coefficient on semantic distance (from 0.7a)
        gamma_inst: Coefficient on institutional distance (from 0.7a)
        exclude_self: If True, set P(i|i) = 0 (workers must move)

    Returns:
        (n_occ, n_occ) probability matrix where P[i,j] = P(dest=j | origin=i)
    """
    n_occ = len(census_codes)

    # Utility: U[i,j] = -gamma_sem * d_sem[i,j] - gamma_inst * d_inst[i,j]
    utility = -gamma_sem * wasserstein_matrix - gamma_inst * inst_matrix

    # Exclude self-transitions (workers must move to different occupation)
    if exclude_self:
        np.fill_diagonal(utility, -np.inf)

    # Softmax over destinations for each origin
    exp_utility = np.exp(utility - np.nanmax(utility, axis=1, keepdims=True))

    # Handle -inf diagonal
    if exclude_self:
        np.fill_diagonal(exp_utility, 0.0)

    # Normalize
    row_sums = exp_utility.sum(axis=1, keepdims=True)
    prob_matrix = exp_utility / (row_sums + 1e-15)

    return prob_matrix


# =============================================================================
# Reallocation Flow Aggregation
# =============================================================================


def aggregate_reallocation_flows(
    prob_matrix: np.ndarray,
    census_codes: List[int],
    exposed_df: pd.DataFrame,
    displacement_rate: float = 1.0,
) -> pd.DataFrame:
    """
    Compute employment-weighted reallocation flows from exposed occupations.

    Args:
        prob_matrix: (n_occ, n_occ) probability matrix P(j|i)
        census_codes: Census codes (row/column labels)
        exposed_df: DataFrame with census_code, tot_emp for exposed occupations
        displacement_rate: Fraction of exposed workers displaced (default 1.0)

    Returns:
        DataFrame with columns: [origin, dest, flow, origin_emp, prob]
        Where flow = origin_emp * displacement_rate * P(dest|origin)
    """
    code_to_idx = {c: i for i, c in enumerate(census_codes)}

    flows = []
    for _, row in exposed_df.iterrows():
        origin = int(row["census_code"])
        origin_emp = float(row["tot_emp"]) * displacement_rate

        if origin not in code_to_idx:
            continue

        i = code_to_idx[origin]

        for j, dest in enumerate(census_codes):
            prob = prob_matrix[i, j]
            if prob > 1e-10:  # Skip negligible flows
                flows.append({
                    "origin": origin,
                    "dest": dest,
                    "flow": origin_emp * prob,
                    "origin_emp": row["tot_emp"],
                    "prob": prob,
                })

    return pd.DataFrame(flows)


def compute_absorption_ranking(
    flows_df: pd.DataFrame,
    employment_df: pd.DataFrame,
    census_codes: List[int],
    occupation_names: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    Rank destination occupations by total absorption.

    Args:
        flows_df: DataFrame from aggregate_reallocation_flows()
        employment_df: DataFrame with census_code, tot_emp
        census_codes: All Census codes
        occupation_names: Optional mapping of census_code -> occupation name

    Returns:
        DataFrame with columns:
            census_code, total_absorption, current_emp, absorption_rate,
            n_origins, occupation_name (if provided)
        Sorted by total_absorption descending.
    """
    # Aggregate flows to each destination
    absorption = flows_df.groupby("dest").agg(
        total_absorption=("flow", "sum"),
        n_origins=("origin", "nunique"),
    ).reset_index()
    absorption = absorption.rename(columns={"dest": "census_code"})

    # Merge with current employment
    emp_dict = dict(zip(employment_df["census_code"], employment_df["tot_emp"]))
    absorption["current_emp"] = absorption["census_code"].map(emp_dict)

    # Compute absorption rate (new workers / current workers)
    absorption["absorption_rate"] = (
        absorption["total_absorption"] / absorption["current_emp"].fillna(1)
    )

    # Add occupation names if provided
    if occupation_names:
        absorption["occupation_name"] = absorption["census_code"].map(occupation_names)

    # Sort by total absorption
    absorption = absorption.sort_values("total_absorption", ascending=False)
    absorption = absorption.reset_index(drop=True)

    return absorption


# =============================================================================
# Validation Against Holdout
# =============================================================================


def validate_against_holdout(
    flows_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    census_codes: List[int],
    exposed_codes: List[int],
    prob_matrix: np.ndarray,
    occupation_names: Optional[Dict[int, str]] = None,
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
) -> dict:
    """
    Compare predicted flows to actual post-shock transitions.

    Enhanced validation with KL divergence, Spearman correlation, and top-K overlap.

    Args:
        flows_df: DataFrame with predicted flows
        holdout_df: DataFrame with actual transitions
        census_codes: Census codes (defines matrix indices)
        exposed_codes: Census codes of exposed (high-AIOE) occupations
        prob_matrix: (n_occ, n_occ) predicted probability matrix
        occupation_names: Optional mapping of census_code -> occupation name
        origin_col: Column name for origin occupation
        dest_col: Column name for destination occupation

    Returns:
        Dict with comprehensive validation metrics
    """
    from scipy import stats

    code_to_idx = {c: i for i, c in enumerate(census_codes)}
    exposed_set = set(exposed_codes)

    # Filter holdout to transitions from exposed occupations
    holdout_exposed = holdout_df[holdout_df[origin_col].isin(exposed_set)].copy()
    n_observed = len(holdout_exposed)

    if n_observed < 20:
        return {
            "n_observed_transitions": n_observed,
            "warning": "Insufficient exposed transitions in holdout",
            "kl_divergence": None,
            "spearman_correlation": None,
            "top5_overlap": None,
            "top10_overlap": None,
        }

    # Compute observed destination distribution (aggregated across all exposed origins)
    observed_dest_counts = holdout_exposed[dest_col].value_counts()
    observed_total = observed_dest_counts.sum()

    # Compute predicted destination distribution (weighted by origin frequency)
    # First get origin distribution from holdout
    origin_counts = holdout_exposed[origin_col].value_counts()

    # Weighted average of predicted probabilities
    predicted_probs = np.zeros(len(census_codes))
    for origin, count in origin_counts.items():
        if origin in code_to_idx:
            i = code_to_idx[origin]
            predicted_probs += count * prob_matrix[i, :]
    predicted_probs /= predicted_probs.sum() + 1e-15

    # Build observed probability vector aligned with census_codes
    observed_probs = np.zeros(len(census_codes))
    for dest, count in observed_dest_counts.items():
        if dest in code_to_idx:
            j = code_to_idx[dest]
            observed_probs[j] = count / observed_total

    # KL Divergence: KL(observed || predicted)
    # Only compute over destinations with observed data
    mask = observed_probs > 0
    if mask.sum() > 0:
        # Add small epsilon to avoid log(0)
        p = observed_probs[mask]
        q = np.maximum(predicted_probs[mask], 1e-15)
        kl_div = float(np.sum(p * np.log(p / q)))
    else:
        kl_div = None

    # Spearman correlation of destination ranks
    # Compare ranking of destinations by observed vs predicted
    spearman_corr = None
    if mask.sum() >= 5:
        obs_ranks = stats.rankdata(-observed_probs)
        pred_ranks = stats.rankdata(-predicted_probs)
        spearman_corr, _ = stats.spearmanr(obs_ranks, pred_ranks)
        spearman_corr = float(spearman_corr) if not np.isnan(spearman_corr) else None

    # Top-K overlap
    def compute_topk_overlap(k: int) -> Tuple[float, List[str], List[str]]:
        """Compute overlap between top-K observed and predicted destinations."""
        # Get top-K observed
        obs_top_idx = np.argsort(-observed_probs)[:k]
        obs_top_codes = [census_codes[i] for i in obs_top_idx if observed_probs[i] > 0]

        # Get top-K predicted
        pred_top_idx = np.argsort(-predicted_probs)[:k]
        pred_top_codes = [census_codes[i] for i in pred_top_idx]

        overlap = len(set(obs_top_codes) & set(pred_top_codes)) / k if k > 0 else 0

        # Get names if available
        if occupation_names:
            obs_names = [occupation_names.get(c, f"Code {c}") for c in obs_top_codes[:k]]
            pred_names = [occupation_names.get(c, f"Code {c}") for c in pred_top_codes[:k]]
        else:
            obs_names = [str(c) for c in obs_top_codes[:k]]
            pred_names = [str(c) for c in pred_top_codes[:k]]

        return overlap, obs_names, pred_names

    top5_overlap, observed_top5, predicted_top5 = compute_topk_overlap(5)
    top10_overlap, observed_top10, predicted_top10 = compute_topk_overlap(10)

    # Per-origin analysis (for deeper diagnostics)
    per_origin_spearman = []
    for origin in origin_counts.index:
        if origin not in code_to_idx:
            continue
        i = code_to_idx[origin]

        # Observed destinations for this origin
        origin_holdout = holdout_exposed[holdout_exposed[origin_col] == origin]
        origin_dest_counts = origin_holdout[dest_col].value_counts()

        if len(origin_dest_counts) < 5:
            continue

        # Build vectors
        obs_vec = np.zeros(len(census_codes))
        for dest, count in origin_dest_counts.items():
            if dest in code_to_idx:
                obs_vec[code_to_idx[dest]] = count

        pred_vec = prob_matrix[i, :]

        # Spearman on this origin
        if obs_vec.sum() > 0:
            corr, _ = stats.spearmanr(obs_vec, pred_vec)
            if not np.isnan(corr):
                per_origin_spearman.append(corr)

    mean_per_origin_spearman = float(np.mean(per_origin_spearman)) if per_origin_spearman else None

    return {
        "n_observed_transitions": n_observed,
        "kl_divergence": kl_div,
        "spearman_correlation": spearman_corr,
        "mean_per_origin_spearman": mean_per_origin_spearman,
        "top5_overlap": float(top5_overlap),
        "top10_overlap": float(top10_overlap),
        "observed_top5_destinations": observed_top5,
        "predicted_top5_destinations": predicted_top5,
        "observed_top10_destinations": observed_top10,
        "predicted_top10_destinations": predicted_top10,
        "n_origins_evaluated": len(per_origin_spearman),
    }


# =============================================================================
# Capacity and Credential Constraints
# =============================================================================


# Credential-gated occupations (require multi-year licensing/certification)
CREDENTIAL_GATED_OCCUPATIONS = {
    # Teachers (require degree + certification + student teaching)
    2300: "Preschool Teachers",
    2310: "Kindergarten Teachers",
    2320: "Elementary School Teachers",
    2330: "Middle School Teachers",
    2340: "Secondary School Teachers",
    2350: "Special Education Teachers",
    # Healthcare (require degree + licensing)
    3000: "Chiropractors",
    3010: "Dentists",
    3030: "Dietitians and Nutritionists",
    3050: "Pharmacists",
    3060: "Physicians and Surgeons",
    3110: "Physician Assistants",
    3130: "Registered Nurses",
    3140: "Nurse Anesthetists",
    3150: "Nurse Midwives",
    3160: "Nurse Practitioners",
    3200: "Physical Therapists",
    3210: "Occupational Therapists",
    3230: "Speech-Language Pathologists",
    3255: "Registered Nurses",  # Census code
    3256: "Nurse Anesthetists",
    3258: "Nurse Practitioners",
    # Legal (require JD + bar exam)
    2100: "Lawyers",
    2105: "Judicial Law Clerks",
    # Engineering (often require PE license)
    1300: "Architects",
    1360: "Civil Engineers",
    # Accounting (CPA requirement for many roles)
    800: "Accountants and Auditors",
}


def flag_capacity_constraints(
    absorption_df: pd.DataFrame,
    capacity_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Flag destinations with capacity constraints.

    Args:
        absorption_df: DataFrame with absorption metrics
        capacity_threshold: Absorption rate above which capacity is constrained

    Returns:
        DataFrame with capacity_constrained and constraint_type columns
    """
    df = absorption_df.copy()

    # Capacity constraint based on absorption rate
    df["capacity_constrained"] = df["absorption_rate"] > capacity_threshold

    # Credential constraint based on occupation type
    df["credential_gated"] = df["census_code"].isin(CREDENTIAL_GATED_OCCUPATIONS.keys())

    # Combined constraint type
    def get_constraint_type(row):
        constraints = []
        if row["capacity_constrained"]:
            constraints.append("capacity")
        if row["credential_gated"]:
            constraints.append("credential")
        return "|".join(constraints) if constraints else "none"

    df["constraint_type"] = df.apply(get_constraint_type, axis=1)
    df["is_constrained"] = df["constraint_type"] != "none"

    return df


def split_feasible_constrained(
    absorption_df: pd.DataFrame,
    capacity_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split destinations into feasible vs constrained.

    Args:
        absorption_df: DataFrame with absorption metrics
        capacity_threshold: Absorption rate threshold for capacity constraint

    Returns:
        Tuple of (feasible_df, constrained_df)
    """
    df = flag_capacity_constraints(absorption_df, capacity_threshold)

    feasible = df[~df["is_constrained"]].copy()
    constrained = df[df["is_constrained"]].copy()

    return feasible, constrained


def compute_validation_verdict(
    spearman_correlation: Optional[float],
    top5_overlap: Optional[float],
) -> str:
    """
    Determine verdict based on validation metrics.

    Args:
        spearman_correlation: Spearman rank correlation
        top5_overlap: Fraction of top-5 overlap

    Returns:
        "validated", "partial", or "failed"
    """
    if spearman_correlation is None or top5_overlap is None:
        return "insufficient_data"

    if spearman_correlation > 0.3 and top5_overlap >= 0.4:
        return "validated"
    elif spearman_correlation > 0.1 or top5_overlap >= 0.2:
        return "partial"
    else:
        return "failed"


# =============================================================================
# Occupation Name Lookup
# =============================================================================


def load_occupation_names(year: int = 2023) -> Dict[int, str]:
    """
    Load mapping from Census 2010 code to occupation name.

    Uses OES data which has occupation titles mapped through SOC codes.

    Args:
        year: OES year to use for names

    Returns:
        Dict mapping census_code -> occupation name
    """
    from task_space.data.oes import load_oes_year

    # Load OES with titles
    oes_df = load_oes_year(year)

    if "OCC_TITLE" not in oes_df.columns:
        return {}

    # Map SOC to Census
    xwalk = load_census_onet_crosswalk()
    xwalk_df = xwalk.crosswalk_df[xwalk.crosswalk_df["matched"] == True].copy()
    xwalk_df["soc_6"] = xwalk_df["onet_soc"].str[:7]
    soc_to_census = xwalk_df.groupby("soc_6")["census_2010"].first().to_dict()

    # Build Census -> Title mapping (use first matching OES entry)
    oes_df["census_code"] = oes_df["OCC_CODE"].map(soc_to_census)
    mapped = oes_df.dropna(subset=["census_code"]).copy()
    mapped["census_code"] = mapped["census_code"].astype(int)

    name_df = mapped.groupby("census_code")["OCC_TITLE"].first()
    return name_df.to_dict()


# =============================================================================
# Results Schema
# =============================================================================


@dataclass
class ReallocationResult:
    """
    Results from counterfactual reallocation analysis.

    Follows the specified JSON schema for outputs/experiments/.
    """
    version: str
    parameters: dict
    exposed_summary: dict
    top_absorbers: List[dict]  # Top N destination occupations
    flow_statistics: dict
    validation: Optional[dict] = None
    policy_implications: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "version": self.version,
            "parameters": self.parameters,
            "exposed_summary": self.exposed_summary,
            "top_absorbers": self.top_absorbers,
            "flow_statistics": self.flow_statistics,
        }
        if self.validation:
            result["validation"] = self.validation
        if self.policy_implications:
            result["policy_implications"] = self.policy_implications
        return result

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Convenience Function
# =============================================================================


def run_reallocation_analysis(
    wasserstein_matrix: np.ndarray,
    inst_matrix: np.ndarray,
    census_codes: List[int],
    gamma_sem: float,
    gamma_inst: float,
    oes_year: int = 2023,
    aioe_quartile: float = 0.75,
    displacement_rate: float = 1.0,
    top_n: int = 20,
) -> ReallocationResult:
    """
    Run full counterfactual reallocation analysis.

    Args:
        wasserstein_matrix: Semantic distance matrix
        inst_matrix: Institutional distance matrix
        census_codes: Census codes
        gamma_sem: Semantic distance coefficient from 0.7a
        gamma_inst: Institutional distance coefficient from 0.7a
        oes_year: OES survey year for employment data
        aioe_quartile: Quartile threshold for exposed occupations
        displacement_rate: Fraction of exposed workers displaced
        top_n: Number of top absorbers to report

    Returns:
        ReallocationResult with full analysis
    """
    # Load data
    aioe_df = get_aioe_by_soc_dataframe(use_lm=True)
    aioe_census_df = map_aioe_to_census(aioe_df)
    employment_df = load_employment_by_census(year=oes_year)
    occupation_names = load_occupation_names()

    # Get exposed occupations
    exposed_df, aioe_threshold = get_exposed_occupations(
        aioe_census_df, employment_df, quartile=aioe_quartile
    )

    # Compute destination probabilities
    prob_matrix = compute_destination_probabilities(
        wasserstein_matrix, inst_matrix, census_codes,
        gamma_sem=gamma_sem, gamma_inst=gamma_inst,
        exclude_self=True,
    )

    # Aggregate flows
    flows_df = aggregate_reallocation_flows(
        prob_matrix, census_codes, exposed_df,
        displacement_rate=displacement_rate,
    )

    # Compute absorption ranking
    absorption_df = compute_absorption_ranking(
        flows_df, employment_df, census_codes,
        occupation_names=occupation_names,
    )

    # Build results
    total_displaced = exposed_df["tot_emp"].sum() * displacement_rate

    # Top absorbers
    top_absorbers = []
    for _, row in absorption_df.head(top_n).iterrows():
        top_absorbers.append({
            "rank": len(top_absorbers) + 1,
            "census_code": int(row["census_code"]),
            "occupation_name": row.get("occupation_name", "Unknown"),
            "total_absorption": float(row["total_absorption"]),
            "current_employment": float(row["current_emp"]) if pd.notna(row["current_emp"]) else None,
            "absorption_rate": float(row["absorption_rate"]) if pd.notna(row["absorption_rate"]) else None,
            "share_of_displaced": float(row["total_absorption"] / total_displaced),
            "n_source_occupations": int(row["n_origins"]),
        })

    # Flow statistics
    flow_stats = {
        "total_flows": len(flows_df),
        "total_displaced_workers": float(total_displaced),
        "mean_flow": float(flows_df["flow"].mean()),
        "max_flow": float(flows_df["flow"].max()),
        "concentration_top10": float(absorption_df.head(10)["total_absorption"].sum() / total_displaced),
        "concentration_top20": float(absorption_df.head(20)["total_absorption"].sum() / total_displaced),
    }

    return ReallocationResult(
        version="0.7c.0",
        parameters={
            "gamma_sem": gamma_sem,
            "gamma_inst": gamma_inst,
            "oes_year": oes_year,
            "aioe_quartile": aioe_quartile,
            "aioe_threshold": float(aioe_threshold),
            "displacement_rate": displacement_rate,
        },
        exposed_summary={
            "n_exposed_occupations": len(exposed_df),
            "total_exposed_employment": float(exposed_df["tot_emp"].sum()),
            "total_displaced": float(total_displaced),
            "mean_aioe_exposed": float(exposed_df["aioe_score"].mean()),
        },
        top_absorbers=top_absorbers,
        flow_statistics=flow_stats,
    )
