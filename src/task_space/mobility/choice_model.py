"""
Conditional logit model for occupation destination choice.

Implements McFadden's conditional logit to estimate how semantic and
institutional distance affect worker destination choice when switching
occupations.

Model:
    U_{kj} = α * (-d_sem(i,j)) + β * (-d_inst(i,j)) + ε_{kj}

Where worker k switches from origin i, and j indexes potential destinations.
Distances enter negatively: workers prefer closer destinations.

References:
- McFadden (1974) "Conditional Logit Analysis of Qualitative Choice Behavior"
- Train (2009) "Discrete Choice Methods with Simulation"
- paper/main.tex Section 4.4 (CPS Mobility Validation)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple
import json

import numpy as np
import pandas as pd


@dataclass
class ChoiceModelResult:
    """
    Results from conditional logit estimation.

    Attributes:
        alpha: Coefficient on semantic distance (expected positive)
        alpha_se: Standard error for alpha
        alpha_t: t-statistic for alpha
        alpha_p: p-value for alpha
        beta: Coefficient on institutional distance (expected positive)
        beta_se: Standard error for beta
        beta_t: t-statistic for beta
        beta_p: p-value for beta
        log_likelihood: Model log-likelihood
        n_transitions: Number of observed transitions
        n_choice_rows: Total rows in choice dataset (transitions × alternatives)
        n_alternatives: Number of alternatives per choice (including chosen)
        converged: Whether optimization converged
        assumptions: List of modeling assumptions
    """
    alpha: float
    alpha_se: float
    alpha_t: float
    alpha_p: float
    beta: float
    beta_se: float
    beta_t: float
    beta_p: float
    log_likelihood: float
    n_transitions: int
    n_choice_rows: int
    n_alternatives: int
    converged: bool
    assumptions: List[str] = field(default_factory=lambda: [
        "IIA (Independence of Irrelevant Alternatives) assumed",
        "Random sampling of alternatives (10:1 ratio)",
        "Distances enter utility linearly and additively",
        "Homogeneous preferences across workers",
        "Choice set = sampled alternatives, not full occupation space",
    ])

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_transitions": self.n_transitions,
            "n_choice_rows": self.n_choice_rows,
            "n_alternatives": self.n_alternatives,
            "alpha_coef": self.alpha,
            "alpha_se": self.alpha_se,
            "alpha_t": self.alpha_t,
            "alpha_p": self.alpha_p,
            "beta_coef": self.beta,
            "beta_se": self.beta_se,
            "beta_t": self.beta_t,
            "beta_p": self.beta_p,
            "log_likelihood": self.log_likelihood,
            "converged": self.converged,
            "assumptions": self.assumptions,
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ChoiceModelResult":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            alpha=data["alpha_coef"],
            alpha_se=data["alpha_se"],
            alpha_t=data["alpha_t"],
            alpha_p=data["alpha_p"],
            beta=data["beta_coef"],
            beta_se=data["beta_se"],
            beta_t=data["beta_t"],
            beta_p=data["beta_p"],
            log_likelihood=data["log_likelihood"],
            n_transitions=data["n_transitions"],
            n_choice_rows=data["n_choice_rows"],
            n_alternatives=data.get("n_alternatives", 11),
            converged=data.get("converged", True),
            assumptions=data.get("assumptions", cls.__dataclass_fields__["assumptions"].default_factory()),
        )


def build_choice_dataset(
    transitions_df: pd.DataFrame,
    d_sem_matrix: np.ndarray,
    d_inst_matrix: np.ndarray,
    occ_codes: List[int],
    n_alternatives: int = 10,
    random_seed: int = 42,
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
    max_distance: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build choice dataset for conditional logit estimation.

    For each observed transition, create a choice set with:
    - The chosen destination (Y = 1)
    - n_alternatives randomly sampled non-chosen destinations (Y = 0)

    Args:
        transitions_df: DataFrame with origin_occ, dest_occ columns
        d_sem_matrix: (n_occ, n_occ) semantic distance matrix
        d_inst_matrix: (n_occ, n_occ) institutional distance matrix
        occ_codes: Occupation codes (row/column labels for matrices)
        n_alternatives: Number of non-chosen alternatives to sample
        random_seed: Random seed for reproducibility
        origin_col: Column name for origin occupation
        dest_col: Column name for destination occupation
        max_distance: Optional maximum distance threshold. Transitions
                     exceeding this distance (in d_sem_matrix) are excluded.

    Returns:
        DataFrame with columns: transition_id, occ, chosen, neg_d_sem, neg_d_inst
    """
    np.random.seed(random_seed)

    occ_to_idx = {occ: i for i, occ in enumerate(occ_codes)}
    all_occs = set(occ_codes)

    rows = []
    transition_id = 0
    skipped_distance = 0

    for _, row in transitions_df.iterrows():
        origin = int(row[origin_col])
        dest = int(row[dest_col])

        # Skip if codes not in distance matrix
        if origin not in occ_to_idx or dest not in occ_to_idx:
            continue

        origin_idx = occ_to_idx[origin]
        dest_idx = occ_to_idx[dest]

        # Skip transitions exceeding distance threshold
        if max_distance is not None:
            if d_sem_matrix[origin_idx, dest_idx] > max_distance:
                skipped_distance += 1
                continue

        # Sample alternatives (excluding chosen destination)
        available = list(all_occs - {dest})
        if len(available) < n_alternatives:
            continue

        sampled_alts = np.random.choice(available, size=n_alternatives, replace=False)

        # Add chosen destination
        rows.append({
            "transition_id": transition_id,
            "occ": dest,
            "chosen": 1,
            "neg_d_sem": -d_sem_matrix[origin_idx, dest_idx],
            "neg_d_inst": -d_inst_matrix[origin_idx, dest_idx],
        })

        # Add sampled alternatives
        for alt in sampled_alts:
            alt_idx = occ_to_idx[alt]
            rows.append({
                "transition_id": transition_id,
                "occ": alt,
                "chosen": 0,
                "neg_d_sem": -d_sem_matrix[origin_idx, alt_idx],
                "neg_d_inst": -d_inst_matrix[origin_idx, alt_idx],
            })

        transition_id += 1

    return pd.DataFrame(rows)


def fit_conditional_logit(
    choice_df: pd.DataFrame,
    group_col: str = "transition_id",
    outcome_col: str = "chosen",
    sem_col: str = "neg_d_sem",
    inst_col: str = "neg_d_inst",
) -> ChoiceModelResult:
    """
    Fit conditional logit model using statsmodels.

    Estimates:
        P(j | i, switch) ∝ exp(α * neg_d_sem + β * neg_d_inst)

    Where neg_d_* are negated distances, so positive coefficients indicate
    workers prefer destinations with lower distances.

    Args:
        choice_df: DataFrame from build_choice_dataset()
        group_col: Column identifying choice groups (transitions)
        outcome_col: Binary outcome column (1 = chosen)
        sem_col: Semantic distance covariate (negated)
        inst_col: Institutional distance covariate (negated)

    Returns:
        ChoiceModelResult with coefficient estimates and standard errors.
    """
    from statsmodels.discrete.conditional_models import ConditionalLogit

    # Prepare data
    endog = choice_df[outcome_col].values
    exog = choice_df[[sem_col, inst_col]].values
    groups = choice_df[group_col].values

    # Fit model
    model = ConditionalLogit(endog, exog, groups=groups)
    result = model.fit(disp=False)

    # Extract results
    coefs = result.params
    ses = result.bse
    tstats = result.tvalues
    pvals = result.pvalues

    n_transitions = choice_df[group_col].nunique()
    n_choice_rows = len(choice_df)
    n_alternatives = n_choice_rows // n_transitions if n_transitions > 0 else 0

    return ChoiceModelResult(
        alpha=float(coefs[0]),
        alpha_se=float(ses[0]),
        alpha_t=float(tstats[0]),
        alpha_p=float(pvals[0]),
        beta=float(coefs[1]),
        beta_se=float(ses[1]),
        beta_t=float(tstats[1]),
        beta_p=float(pvals[1]),
        log_likelihood=float(result.llf),
        n_transitions=n_transitions,
        n_choice_rows=n_choice_rows,
        n_alternatives=n_alternatives,
        converged=result.mle_retvals.get("converged", True) if hasattr(result, "mle_retvals") else True,
    )


def compute_odds_ratios(result: ChoiceModelResult) -> dict:
    """
    Compute odds ratios for interpretation.

    For a 1-unit increase in distance:
        Odds ratio = exp(-coefficient)

    Since covariates are negated distances, a positive coefficient means
    workers prefer lower distances.

    Args:
        result: ChoiceModelResult from fit_conditional_logit()

    Returns:
        Dictionary with odds ratios and interpretations.
    """
    or_sem = np.exp(-result.alpha)
    or_inst = np.exp(-result.beta)

    return {
        "semantic": {
            "odds_ratio": or_sem,
            "interpretation": f"1-unit increase in d_sem → {(1-or_sem)*100:.1f}% lower odds",
        },
        "institutional": {
            "odds_ratio": or_inst,
            "interpretation": f"1-unit increase in d_inst → {(1-or_inst)*100:.1f}% lower odds",
        },
        "coefficient_ratio": result.alpha / result.beta if result.beta != 0 else np.inf,
    }


def load_canonical_results(
    path: Optional[str] = None,
) -> ChoiceModelResult:
    """
    Load canonical conditional logit results from the paper.

    Args:
        path: Path to results JSON. If None, uses default location.

    Returns:
        ChoiceModelResult with paper's main findings.
    """
    if path is None:
        default_paths = [
            Path("data/processed/mobility/conditional_logit_results.json"),
            Path("killshot/cps_mobility/data/conditional_logit_results.json"),
        ]
        for p in default_paths:
            if p.exists():
                path = str(p)
                break

    if path is None:
        raise FileNotFoundError(
            "Canonical results not found. Check data/processed/mobility/"
        )

    return ChoiceModelResult.load(path)


# ============================================================================
# Asymmetric Choice Model (v0.6.6.0)
# ============================================================================


@dataclass
class AsymmetricChoiceModelResult:
    """
    Conditional logit results with directional institutional coefficients.

    Model:
        U_{kj} = α * (-d_sem) + β_up * (-d_inst_up) + β_down * (-d_inst_down) + ε

    Theory: β_up >> β_down (possibly β_down ≈ 0) if credentials are one-way gates.

    Attributes:
        alpha: Semantic distance coefficient
        alpha_se, alpha_t, alpha_p: Standard error, t-stat, p-value for alpha
        beta_up: Upward institutional barrier coefficient
        beta_up_se, beta_up_t, beta_up_p: SE, t-stat, p-value for beta_up
        beta_down: Downward institutional barrier coefficient
        beta_down_se, beta_down_t, beta_down_p: SE, t-stat, p-value for beta_down
        log_likelihood: Model log-likelihood
        n_transitions: Number of observed transitions
        n_choice_rows: Total rows in choice dataset
        n_alternatives: Alternatives per choice
        converged: Whether optimization converged
        asymmetry_ratio: |beta_up| / |beta_down|
        lr_test_statistic: LR test for H0: beta_up = beta_down
        lr_test_pvalue: p-value for LR test (chi-squared, df=1)
        ll_restricted: Log-likelihood of restricted (symmetric) model
        assumptions: Modeling assumptions
    """
    # Semantic coefficient
    alpha: float
    alpha_se: float
    alpha_t: float
    alpha_p: float

    # Upward institutional coefficient
    beta_up: float
    beta_up_se: float
    beta_up_t: float
    beta_up_p: float

    # Downward institutional coefficient
    beta_down: float
    beta_down_se: float
    beta_down_t: float
    beta_down_p: float

    # Model fit
    log_likelihood: float
    n_transitions: int
    n_choice_rows: int
    n_alternatives: int
    converged: bool

    # Asymmetry tests
    asymmetry_ratio: float  # |beta_up| / |beta_down|
    lr_test_statistic: float  # LR test: 2 * (LL_unrestricted - LL_restricted)
    lr_test_pvalue: float
    ll_restricted: float  # Log-likelihood of symmetric model

    assumptions: List[str] = field(default_factory=lambda: [
        "IIA (Independence of Irrelevant Alternatives) assumed",
        "Random sampling of alternatives (10:1 ratio)",
        "Upward barrier = max(0, Zone_dest - Zone_origin)",
        "Downward barrier = max(0, Zone_origin - Zone_dest)",
        "LR test: H0: beta_up = beta_down",
    ])

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_transitions": self.n_transitions,
            "n_choice_rows": self.n_choice_rows,
            "n_alternatives": self.n_alternatives,
            "alpha": self.alpha,
            "alpha_se": self.alpha_se,
            "alpha_t": self.alpha_t,
            "alpha_p": self.alpha_p,
            "beta_up": self.beta_up,
            "beta_up_se": self.beta_up_se,
            "beta_up_t": self.beta_up_t,
            "beta_up_p": self.beta_up_p,
            "beta_down": self.beta_down,
            "beta_down_se": self.beta_down_se,
            "beta_down_t": self.beta_down_t,
            "beta_down_p": self.beta_down_p,
            "log_likelihood": self.log_likelihood,
            "converged": self.converged,
            "asymmetry_ratio": self.asymmetry_ratio,
            "lr_test_statistic": self.lr_test_statistic,
            "lr_test_pvalue": self.lr_test_pvalue,
            "ll_restricted": self.ll_restricted,
            "assumptions": self.assumptions,
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def build_asymmetric_choice_dataset(
    transitions_df: pd.DataFrame,
    d_sem_matrix: np.ndarray,
    d_up_matrix: np.ndarray,
    d_down_matrix: np.ndarray,
    occ_codes: List[int],
    n_alternatives: int = 10,
    random_seed: int = 42,
    origin_col: str = "origin_occ",
    dest_col: str = "dest_occ",
) -> pd.DataFrame:
    """
    Build choice dataset with asymmetric institutional distances.

    For each observed transition, create a choice set with:
    - The chosen destination (Y = 1)
    - n_alternatives randomly sampled non-chosen destinations (Y = 0)

    Args:
        transitions_df: DataFrame with origin and destination columns
        d_sem_matrix: (n_occ, n_occ) semantic distance matrix
        d_up_matrix: (n_occ, n_occ) upward institutional distance
        d_down_matrix: (n_occ, n_occ) downward institutional distance
        occ_codes: Occupation codes (row/column labels for matrices)
        n_alternatives: Number of non-chosen alternatives to sample
        random_seed: Random seed for reproducibility
        origin_col: Column name for origin occupation
        dest_col: Column name for destination occupation

    Returns:
        DataFrame with columns:
            transition_id, occ, chosen, neg_d_sem, neg_d_inst_up, neg_d_inst_down
    """
    np.random.seed(random_seed)

    occ_to_idx = {occ: i for i, occ in enumerate(occ_codes)}
    all_occs = set(occ_codes)

    rows = []
    transition_id = 0

    for _, row in transitions_df.iterrows():
        origin = int(row[origin_col])
        dest = int(row[dest_col])

        # Skip if codes not in distance matrix
        if origin not in occ_to_idx or dest not in occ_to_idx:
            continue

        origin_idx = occ_to_idx[origin]
        dest_idx = occ_to_idx[dest]

        # Sample alternatives (excluding chosen destination)
        available = list(all_occs - {dest})
        if len(available) < n_alternatives:
            continue

        sampled_alts = np.random.choice(available, size=n_alternatives, replace=False)

        # Add chosen destination
        rows.append({
            "transition_id": transition_id,
            "occ": dest,
            "chosen": 1,
            "neg_d_sem": -d_sem_matrix[origin_idx, dest_idx],
            "neg_d_inst_up": -d_up_matrix[origin_idx, dest_idx],
            "neg_d_inst_down": -d_down_matrix[origin_idx, dest_idx],
        })

        # Add sampled alternatives
        for alt in sampled_alts:
            alt_idx = occ_to_idx[alt]
            rows.append({
                "transition_id": transition_id,
                "occ": alt,
                "chosen": 0,
                "neg_d_sem": -d_sem_matrix[origin_idx, alt_idx],
                "neg_d_inst_up": -d_up_matrix[origin_idx, alt_idx],
                "neg_d_inst_down": -d_down_matrix[origin_idx, alt_idx],
            })

        transition_id += 1

    return pd.DataFrame(rows)


def fit_asymmetric_conditional_logit(
    choice_df: pd.DataFrame,
    group_col: str = "transition_id",
    outcome_col: str = "chosen",
    sem_col: str = "neg_d_sem",
    up_col: str = "neg_d_inst_up",
    down_col: str = "neg_d_inst_down",
) -> AsymmetricChoiceModelResult:
    """
    Fit conditional logit with separate upward/downward institutional barriers.

    Model:
        U_j = α * neg_d_sem + β_up * neg_d_inst_up + β_down * neg_d_inst_down + ε

    Also fits restricted model (β_up = β_down) for LR test.

    Args:
        choice_df: DataFrame from build_asymmetric_choice_dataset()
        group_col: Column identifying choice groups (transitions)
        outcome_col: Binary outcome column (1 = chosen)
        sem_col: Semantic distance covariate (negated)
        up_col: Upward institutional distance (negated)
        down_col: Downward institutional distance (negated)

    Returns:
        AsymmetricChoiceModelResult with coefficient estimates and LR test.
    """
    from statsmodels.discrete.conditional_models import ConditionalLogit
    from scipy import stats

    # Prepare data
    endog = choice_df[outcome_col].values
    exog_unrestricted = choice_df[[sem_col, up_col, down_col]].values
    groups = choice_df[group_col].values

    # Fit unrestricted model (beta_up, beta_down separate)
    model_unrestricted = ConditionalLogit(endog, exog_unrestricted, groups=groups)
    result_unrestricted = model_unrestricted.fit(disp=False)

    # Fit restricted model (beta_up = beta_down, i.e., symmetric)
    # Use sum of up + down as single symmetric covariate
    neg_d_inst_sym = choice_df[up_col].values + choice_df[down_col].values
    exog_restricted = np.column_stack([choice_df[sem_col].values, neg_d_inst_sym])
    model_restricted = ConditionalLogit(endog, exog_restricted, groups=groups)
    result_restricted = model_restricted.fit(disp=False)

    # Extract unrestricted results
    coefs = result_unrestricted.params
    ses = result_unrestricted.bse
    tstats = result_unrestricted.tvalues
    pvals = result_unrestricted.pvalues

    # Compute asymmetry ratio (handle division by zero)
    beta_up = float(coefs[1])
    beta_down = float(coefs[2])
    if abs(beta_down) > 1e-10:
        asymmetry_ratio = abs(beta_up) / abs(beta_down)
    else:
        asymmetry_ratio = float('inf') if abs(beta_up) > 1e-10 else 1.0

    # LR test: 2 * (LL_unrestricted - LL_restricted)
    ll_unrestricted = float(result_unrestricted.llf)
    ll_restricted = float(result_restricted.llf)
    lr_stat = 2 * (ll_unrestricted - ll_restricted)
    lr_pvalue = 1 - stats.chi2.cdf(lr_stat, df=1)  # df=1 for one restriction

    n_transitions = choice_df[group_col].nunique()
    n_choice_rows = len(choice_df)
    n_alternatives = n_choice_rows // n_transitions if n_transitions > 0 else 0

    return AsymmetricChoiceModelResult(
        alpha=float(coefs[0]),
        alpha_se=float(ses[0]),
        alpha_t=float(tstats[0]),
        alpha_p=float(pvals[0]),
        beta_up=beta_up,
        beta_up_se=float(ses[1]),
        beta_up_t=float(tstats[1]),
        beta_up_p=float(pvals[1]),
        beta_down=beta_down,
        beta_down_se=float(ses[2]),
        beta_down_t=float(tstats[2]),
        beta_down_p=float(pvals[2]),
        log_likelihood=ll_unrestricted,
        n_transitions=n_transitions,
        n_choice_rows=n_choice_rows,
        n_alternatives=n_alternatives,
        converged=result_unrestricted.mle_retvals.get("converged", True) if hasattr(result_unrestricted, "mle_retvals") else True,
        asymmetry_ratio=asymmetry_ratio,
        lr_test_statistic=lr_stat,
        lr_test_pvalue=lr_pvalue,
        ll_restricted=ll_restricted,
    )


def compute_asymmetric_odds_ratios(result: AsymmetricChoiceModelResult) -> dict:
    """
    Compute odds ratios for asymmetric model interpretation.

    For a 1-unit increase in distance:
        Odds ratio = exp(-coefficient)

    Args:
        result: AsymmetricChoiceModelResult

    Returns:
        Dictionary with odds ratios and interpretations.
    """
    or_sem = np.exp(-result.alpha)
    or_up = np.exp(-result.beta_up)
    or_down = np.exp(-result.beta_down)

    return {
        "semantic": {
            "odds_ratio": or_sem,
            "interpretation": f"1-unit d_sem increase → {(1-or_sem)*100:.1f}% lower odds",
        },
        "upward": {
            "odds_ratio": or_up,
            "interpretation": f"1-unit d_up increase → {(1-or_up)*100:.1f}% lower odds",
        },
        "downward": {
            "odds_ratio": or_down,
            "interpretation": f"1-unit d_down increase → {(1-or_down)*100:.1f}% lower odds",
        },
        "asymmetry_ratio": result.asymmetry_ratio,
        "asymmetry_significant": result.lr_test_pvalue < 0.05,
    }
