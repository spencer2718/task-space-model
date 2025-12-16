"""
Scaled cost estimation for occupation switching.

Adds destination wages to conditional logit to enable switching cost
estimation in wage-equivalent units.

Phase 0.7b: Integrates wage data with task-space geometry.

Reference: Dix-Carneiro (2014) finds switching costs of 1.4–2.7× annual wages.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import json

import numpy as np
import pandas as pd

from task_space.data.oes import load_oes_year
from task_space.mobility.census_crosswalk import load_census_onet_crosswalk


# =============================================================================
# OES Wage Loading with Census Mapping
# =============================================================================


def load_oes_wages_by_census(year: int = 2023) -> pd.DataFrame:
    """
    Load OES wages mapped to Census occupation codes.

    Args:
        year: OES survey year (default 2023)

    Returns:
        DataFrame with columns: [census_code, mean_annual_wage, median_annual_wage, soc_code]
    """
    # Load OES data
    oes = load_oes_year(year)

    # Load crosswalk
    xwalk = load_census_onet_crosswalk()
    xwalk_df = xwalk.crosswalk_df.copy()
    xwalk_df = xwalk_df[xwalk_df["matched"] == True]

    # Build SOC-6 to Census mapping
    if "soc_6digit" in xwalk_df.columns:
        soc6_to_census = xwalk_df.groupby("soc_6digit")["census_2010"].first().to_dict()
    else:
        xwalk_df["soc_6"] = xwalk_df["onet_soc"].str.slice(0, 7)
        soc6_to_census = xwalk_df.groupby("soc_6")["census_2010"].first().to_dict()

    # Map OES SOC codes to Census
    oes = oes.copy()
    oes["census_code"] = oes["OCC_CODE"].map(soc6_to_census)

    # Filter to matched occupations
    matched = oes.dropna(subset=["census_code"]).copy()
    matched["census_code"] = matched["census_code"].astype(int)

    # Aggregate by Census code (mean if multiple SOC codes map to one Census)
    result = matched.groupby("census_code").agg(
        mean_annual_wage=("A_MEAN", "mean"),
        median_annual_wage=("A_MEDIAN", "mean"),
        soc_code=("OCC_CODE", "first"),  # Keep one for reference
        n_soc_codes=("OCC_CODE", "count"),
    ).reset_index()

    return result


def get_wage_coverage(wages_df: pd.DataFrame, census_codes: List[int]) -> float:
    """Compute fraction of Census codes with wage data."""
    wage_codes = set(wages_df["census_code"])
    census_set = set(census_codes)
    matched = wage_codes & census_set
    return len(matched) / len(census_set) if census_set else 0.0


# =============================================================================
# Choice Dataset with Wages
# =============================================================================


def build_choice_dataset_with_wages(
    transitions_df: pd.DataFrame,
    wasserstein_matrix: np.ndarray,
    inst_matrix: np.ndarray,
    wages_df: pd.DataFrame,
    occupation_codes: List[int],
    n_alternatives: int = 10,
    random_seed: int = 42,
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
) -> pd.DataFrame:
    """
    Build choice dataset including destination wages.

    Args:
        transitions_df: DataFrame with origin_occ, dest_occ columns
        wasserstein_matrix: (n_occ, n_occ) Wasserstein distance matrix
        inst_matrix: (n_occ, n_occ) institutional distance matrix
        wages_df: DataFrame from load_oes_wages_by_census()
        occupation_codes: Census codes (row/column labels for matrices)
        n_alternatives: Number of non-chosen alternatives to sample
        random_seed: Random seed for reproducibility
        origin_col: Column name for origin occupation
        dest_col: Column name for destination occupation

    Returns:
        DataFrame with columns:
            case_id, alt_id, chosen, d_wasserstein, d_inst,
            log_wage_dest, log_wage_ratio, wage_dest
    """
    np.random.seed(random_seed)

    occ_to_idx = {occ: i for i, occ in enumerate(occupation_codes)}
    all_occs = set(occupation_codes)

    # Build wage lookup
    wage_dict = dict(zip(wages_df["census_code"], wages_df["mean_annual_wage"]))

    rows = []
    case_id = 0

    for _, row in transitions_df.iterrows():
        origin = int(row[origin_col])
        dest = int(row[dest_col])

        # Skip if codes not in distance matrix or wage data
        if origin not in occ_to_idx or dest not in occ_to_idx:
            continue
        if origin not in wage_dict or dest not in wage_dict:
            continue

        origin_idx = occ_to_idx[origin]
        dest_idx = occ_to_idx[dest]
        origin_wage = wage_dict[origin]

        # Sample alternatives (excluding chosen destination)
        available = [o for o in all_occs - {dest} if o in wage_dict]
        if len(available) < n_alternatives:
            continue

        sampled_alts = np.random.choice(available, size=n_alternatives, replace=False)

        # Add chosen destination
        dest_wage = wage_dict[dest]
        rows.append({
            "case_id": case_id,
            "alt_id": dest,
            "chosen": 1,
            "d_wasserstein": wasserstein_matrix[origin_idx, dest_idx],
            "d_inst": inst_matrix[origin_idx, dest_idx],
            "wage_dest": dest_wage,
            "log_wage_dest": np.log(dest_wage),
            "log_wage_ratio": np.log(dest_wage / origin_wage),
        })

        # Add sampled alternatives
        for alt in sampled_alts:
            alt_idx = occ_to_idx[alt]
            alt_wage = wage_dict[alt]
            rows.append({
                "case_id": case_id,
                "alt_id": alt,
                "chosen": 0,
                "d_wasserstein": wasserstein_matrix[origin_idx, alt_idx],
                "d_inst": inst_matrix[origin_idx, alt_idx],
                "wage_dest": alt_wage,
                "log_wage_dest": np.log(alt_wage),
                "log_wage_ratio": np.log(alt_wage / origin_wage),
            })

        case_id += 1

    return pd.DataFrame(rows)


# =============================================================================
# Model Estimation
# =============================================================================


@dataclass
class ScaledModelResult:
    """
    Results from conditional logit with wage covariates.

    Attributes:
        gamma_sem: Coefficient on Wasserstein distance (negative expected)
        gamma_inst: Coefficient on institutional distance (negative expected)
        beta_wage: Coefficient on wage measure (positive expected)
        *_se, *_t, *_p: Standard errors, t-stats, p-values
        log_likelihood: Model log-likelihood
        n_cases: Number of choice cases (transitions)
        n_obs: Total observations (cases × alternatives)
        converged: Whether optimization converged
        model_variant: Which wage specification (M1, M2, M3)
    """
    gamma_sem: float
    gamma_sem_se: float
    gamma_sem_t: float
    gamma_sem_p: float
    gamma_inst: float
    gamma_inst_se: float
    gamma_inst_t: float
    gamma_inst_p: float
    beta_wage: float
    beta_wage_se: float
    beta_wage_t: float
    beta_wage_p: float
    log_likelihood: float
    n_cases: int
    n_obs: int
    converged: bool
    model_variant: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "gamma_sem": self.gamma_sem,
            "gamma_sem_se": self.gamma_sem_se,
            "gamma_sem_t": self.gamma_sem_t,
            "gamma_sem_p": self.gamma_sem_p,
            "gamma_inst": self.gamma_inst,
            "gamma_inst_se": self.gamma_inst_se,
            "gamma_inst_t": self.gamma_inst_t,
            "gamma_inst_p": self.gamma_inst_p,
            "beta_wage": self.beta_wage,
            "beta_wage_se": self.beta_wage_se,
            "beta_wage_t": self.beta_wage_t,
            "beta_wage_p": self.beta_wage_p,
            "log_likelihood": self.log_likelihood,
            "n_cases": self.n_cases,
            "n_obs": self.n_obs,
            "converged": self.converged,
            "model_variant": self.model_variant,
        }


def estimate_scaled_model(
    choice_df: pd.DataFrame,
    model_variant: str = "M1",
) -> ScaledModelResult:
    """
    Estimate conditional logit with wages.

    Model variants:
        M1: log destination wage (primary specification)
        M2: level destination wage (in $10k units)
        M3: log wage ratio (dest/origin)

    All models include:
        U_j = gamma_sem * (-d_wasserstein) + gamma_inst * (-d_inst) + beta_wage * wage_measure

    Args:
        choice_df: DataFrame from build_choice_dataset_with_wages()
        model_variant: "M1", "M2", or "M3"

    Returns:
        ScaledModelResult with coefficient estimates.
    """
    from statsmodels.discrete.conditional_models import ConditionalLogit

    # Select wage covariate based on variant
    if model_variant == "M1":
        wage_col = "log_wage_dest"
    elif model_variant == "M2":
        # Scale to $10k units for interpretability
        choice_df = choice_df.copy()
        choice_df["wage_10k"] = choice_df["wage_dest"] / 10000
        wage_col = "wage_10k"
    elif model_variant == "M3":
        wage_col = "log_wage_ratio"
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")

    # Prepare data (negate distances so positive coef = prefer lower distance)
    endog = choice_df["chosen"].values
    exog = np.column_stack([
        -choice_df["d_wasserstein"].values,
        -choice_df["d_inst"].values,
        choice_df[wage_col].values,
    ])
    groups = choice_df["case_id"].values

    # Fit model
    model = ConditionalLogit(endog, exog, groups=groups)
    result = model.fit(disp=False)

    # Extract results
    coefs = result.params
    ses = result.bse
    tstats = result.tvalues
    pvals = result.pvalues

    n_cases = choice_df["case_id"].nunique()
    n_obs = len(choice_df)

    converged = True
    if hasattr(result, "mle_retvals"):
        converged = result.mle_retvals.get("converged", True)

    return ScaledModelResult(
        gamma_sem=float(coefs[0]),
        gamma_sem_se=float(ses[0]),
        gamma_sem_t=float(tstats[0]),
        gamma_sem_p=float(pvals[0]),
        gamma_inst=float(coefs[1]),
        gamma_inst_se=float(ses[1]),
        gamma_inst_t=float(tstats[1]),
        gamma_inst_p=float(pvals[1]),
        beta_wage=float(coefs[2]),
        beta_wage_se=float(ses[2]),
        beta_wage_t=float(tstats[2]),
        beta_wage_p=float(pvals[2]),
        log_likelihood=float(result.llf),
        n_cases=n_cases,
        n_obs=n_obs,
        converged=converged,
        model_variant=model_variant,
    )


# =============================================================================
# Switching Cost Computation
# =============================================================================


def compute_switching_costs(
    result: ScaledModelResult,
    d_wasserstein_median: float,
    d_inst_median: float,
    mean_annual_wage: float,
) -> dict:
    """
    Compute switching costs in wage-equivalent units.

    The model is:
        U_j = gamma_sem * (-d_sem) + gamma_inst * (-d_inst) + beta_wage * log(wage_j)

    Marginal rate of substitution for d_sem:
        MRS = -gamma_sem / beta_wage

    This gives the log-wage change needed to compensate for 1 unit of d_sem.
    Converting to levels: exp(MRS) - 1 ≈ MRS for small MRS.

    For typical transition (median distances):
        sc_typical = MRS_sem * d_sem_median + MRS_inst * d_inst_median

    Args:
        result: ScaledModelResult from estimate_scaled_model()
        d_wasserstein_median: Median Wasserstein distance in sample
        d_inst_median: Median institutional distance in sample
        mean_annual_wage: Mean annual wage for dollar conversion

    Returns:
        Dict with switching cost estimates in various units.
    """
    # For M1 (log wages), MRS gives log-wage equivalent
    # For M2 (level wages), need different interpretation
    # For M3 (log ratio), similar to M1

    if result.model_variant in ["M1", "M3"]:
        # MRS in log-wage units (≈ proportional wage change)
        # Model: U = gamma * (-d) + beta * log(w)
        # gamma > 0 means prefer lower distance
        # beta > 0 means prefer higher wages
        # MRS = d(log_w)/d(d) = gamma / beta (wage increase needed per unit distance)
        if abs(result.beta_wage) < 1e-10:
            mrs_sem = np.inf
            mrs_inst = np.inf
        else:
            mrs_sem = result.gamma_sem / result.beta_wage
            mrs_inst = result.gamma_inst / result.beta_wage

        # Typical switching cost in log-wage units
        sc_typical_log = mrs_sem * d_wasserstein_median + mrs_inst * d_inst_median

        # Convert to wage-years: exp(sc_log) - 1 ≈ sc_log for small values
        # For interpretation: this is the wage premium (as fraction) needed
        # to compensate for the transition cost
        sc_typical_wage_years = sc_typical_log  # Already in log units ≈ fraction

        # Convert to dollars
        sc_typical_dollars = sc_typical_wage_years * mean_annual_wage

    elif result.model_variant == "M2":
        # beta_wage is per $10k, so MRS is in $10k per unit distance
        if abs(result.beta_wage) < 1e-10:
            mrs_sem = np.inf
            mrs_inst = np.inf
        else:
            mrs_sem = result.gamma_sem / result.beta_wage  # in $10k per unit
            mrs_inst = result.gamma_inst / result.beta_wage

        sc_typical_dollars = (mrs_sem * d_wasserstein_median + mrs_inst * d_inst_median) * 10000
        sc_typical_wage_years = sc_typical_dollars / mean_annual_wage

    else:
        raise ValueError(f"Unknown model variant: {result.model_variant}")

    return {
        "sc_sem_per_unit": float(mrs_sem),
        "sc_inst_per_unit": float(mrs_inst),
        "d_wasserstein_median": float(d_wasserstein_median),
        "d_inst_median": float(d_inst_median),
        "sc_typical_wage_years": float(sc_typical_wage_years),
        "sc_typical_dollars": float(sc_typical_dollars),
        "mean_annual_wage": float(mean_annual_wage),
    }


def compute_median_distances(
    transitions_df: pd.DataFrame,
    wasserstein_matrix: np.ndarray,
    inst_matrix: np.ndarray,
    occupation_codes: List[int],
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
) -> Tuple[float, float]:
    """
    Compute median distances from observed transitions.

    Args:
        transitions_df: DataFrame with origin and destination columns
        wasserstein_matrix: (n_occ, n_occ) distance matrix
        inst_matrix: (n_occ, n_occ) institutional distance matrix
        occupation_codes: Census codes (row/column labels)
        origin_col: Column name for origin occupation
        dest_col: Column name for destination occupation

    Returns:
        Tuple of (median_wasserstein, median_institutional)
    """
    occ_to_idx = {occ: i for i, occ in enumerate(occupation_codes)}

    d_sem_list = []
    d_inst_list = []

    for _, row in transitions_df.iterrows():
        origin = int(row[origin_col])
        dest = int(row[dest_col])

        if origin in occ_to_idx and dest in occ_to_idx:
            i = occ_to_idx[origin]
            j = occ_to_idx[dest]
            d_sem_list.append(wasserstein_matrix[i, j])
            d_inst_list.append(inst_matrix[i, j])

    if not d_sem_list:
        return 0.0, 0.0

    return float(np.median(d_sem_list)), float(np.median(d_inst_list))


# =============================================================================
# External Calibration (0.7b-revised)
# =============================================================================


def compute_externally_calibrated_costs(
    d_wasserstein_median: float,
    benchmark_cost_wage_years: float = 2.0,  # Dix-Carneiro midpoint
    mean_annual_wage: float = 71992.0,
) -> dict:
    """
    Calibrate switching costs using external benchmark.

    Since wage coefficient identification fails with occupation-mean wages,
    we anchor to structural estimates from the literature.

    Logic: If typical transition costs benchmark_cost_wage_years,
    and typical transition has d_wasserstein_median distance,
    then cost per unit = benchmark / median_distance

    Args:
        d_wasserstein_median: Median Wasserstein distance in observed transitions
        benchmark_cost_wage_years: External benchmark (Dix-Carneiro 2014 median = 2.0)
        mean_annual_wage: Mean annual wage for dollar conversion

    Returns:
        Dict with calibrated cost parameters
    """
    sc_per_unit_wasserstein = benchmark_cost_wage_years / d_wasserstein_median

    return {
        "calibration_source": "Dix-Carneiro (2014) median",
        "benchmark_wage_years": benchmark_cost_wage_years,
        "d_wasserstein_median": d_wasserstein_median,
        "sc_per_unit_wasserstein_wage_years": sc_per_unit_wasserstein,
        "sc_per_unit_wasserstein_dollars": sc_per_unit_wasserstein * mean_annual_wage,
        "mean_annual_wage": mean_annual_wage,
    }


def compute_example_transition_costs(
    examples: List[Dict],
    sc_per_unit: float,
    mean_wage: float,
) -> List[Dict]:
    """
    Compute costs for illustrative occupation pairs.

    Args:
        examples: List of dicts with 'from', 'to', 'd_wass' keys
        sc_per_unit: Switching cost per unit Wasserstein (in wage-years)
        mean_wage: Mean annual wage for dollar conversion

    Returns:
        List of dicts with added 'cost_wage_years' and 'cost_dollars'
    """
    result = []
    for ex in examples:
        ex_copy = ex.copy()
        ex_copy["cost_wage_years"] = round(ex["d_wass"] * sc_per_unit, 2)
        ex_copy["cost_dollars"] = round(ex["d_wass"] * sc_per_unit * mean_wage, 0)
        result.append(ex_copy)
    return result


def lookup_wasserstein_distance(
    origin_census: int,
    dest_census: int,
    wasserstein_matrix: np.ndarray,
    census_codes: List[int],
) -> Optional[float]:
    """
    Look up Wasserstein distance between two Census occupation codes.

    Args:
        origin_census: Origin Census 2010 code
        dest_census: Destination Census 2010 code
        wasserstein_matrix: Distance matrix
        census_codes: Census codes (row/column labels)

    Returns:
        Distance, or None if codes not found
    """
    try:
        i = census_codes.index(origin_census)
        j = census_codes.index(dest_census)
        return float(wasserstein_matrix[i, j])
    except ValueError:
        return None


# =============================================================================
# Verdict Logic
# =============================================================================


def compute_verdict(
    beta_wage: float,
    gamma_sem: float,
    gamma_inst: float,
    sc_typical_wage_years: float,
) -> str:
    """
    Determine verdict based on coefficient signs and magnitude.

    Args:
        beta_wage: Wage coefficient (should be positive)
        gamma_sem: Semantic distance coefficient (should be positive for neg distances)
        gamma_inst: Institutional distance coefficient (should be positive)
        sc_typical_wage_years: Typical switching cost in wage-years

    Returns:
        One of: "validated", "investigate", "misidentified"

    Logic:
        - "validated": β_wage > 0, γ's > 0, AND sc_typical in [0.5, 5.0] wage-years
        - "investigate": something anomalous but not completely broken
        - "misidentified": β_wage ≤ 0 (wage doesn't attract workers)
    """
    if beta_wage <= 0:
        return "misidentified"

    # Check if signs are correct
    # gamma_sem and gamma_inst should be positive (since we negate distances in exog)
    signs_correct = gamma_sem > 0 and gamma_inst > 0

    # Check if magnitude is reasonable
    in_range = 0.5 <= sc_typical_wage_years <= 5.0

    if signs_correct and in_range:
        return "validated"
    else:
        # beta_wage > 0 but either signs wrong or out of range - investigate
        return "investigate"


# =============================================================================
# Results Schema
# =============================================================================


@dataclass
class ScaledCostsResult:
    """
    Full results from scaled cost estimation.

    Follows the JSON schema for outputs/experiments/.
    """
    version: str
    wage_data: dict
    model_m1: dict
    model_m2: dict
    model_m3: dict
    switching_costs: dict
    benchmark_comparison: dict
    diagnostics: dict
    verdict: str
    external_calibration: Optional[dict] = None
    wage_identification_failure: Optional[dict] = None
    revised_verdict: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "version": self.version,
            "wage_data": self.wage_data,
            "model_m1": self.model_m1,
            "model_m2": self.model_m2,
            "model_m3": self.model_m3,
            "switching_costs": self.switching_costs,
            "benchmark_comparison": self.benchmark_comparison,
            "diagnostics": self.diagnostics,
            "verdict": self.verdict,
        }
        if self.external_calibration:
            result["external_calibration"] = self.external_calibration
        if self.wage_identification_failure:
            result["wage_identification_failure"] = self.wage_identification_failure
        if self.revised_verdict:
            result["revised_verdict"] = self.revised_verdict
        return result

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
