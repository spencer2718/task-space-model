"""
Battery evaluator for retrospective tests.

Implements unified regression and scoring for the interpretation matrix.
Compares discrete (benchmark) vs continuous (geometry) exposure measures.

The key regression specification:

    ΔY = β₀ + β₁·D + β₂·C + β₃·C_resid + ε

Where:
    ΔY = Outcome variable (employment change, wage change, etc.)
    D = Discrete exposure (benchmark from prior literature)
    C = Continuous exposure (raw geometry-based measure)
    C_resid = Continuous exposure orthogonalized to D (captures residual signal)

The interpretation matrix maps (sign of β₃, significance) to verdict:
    β₃ > 0, p < 0.05: Geometry adds signal → "+"
    β₃ < 0, p < 0.05: Geometry contradicts → "-"
    p ≥ 0.05: No significant residual → "0"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import scipy.stats


class Verdict(Enum):
    """Interpretation matrix verdict."""
    POSITIVE = "+"    # Geometry adds significant signal
    NEGATIVE = "-"    # Geometry contradicts benchmark
    NULL = "0"        # No significant residual effect


@dataclass
class RegressionCoefficients:
    """
    Regression coefficients with standard errors.

    Attributes:
        beta: Coefficient estimate
        se: Standard error
        t: t-statistic
        p: p-value (two-sided)
        name: Variable name
    """
    beta: float
    se: float
    t: float
    p: float
    name: str

    @property
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if coefficient is significant at given alpha level."""
        return self.p < alpha


@dataclass
class ModelComparison:
    """
    Model comparison statistics.

    Attributes:
        r2_discrete: R² from discrete-only model
        r2_full: R² from full model (discrete + continuous residual)
        delta_r2: Incremental R² from adding continuous
        f_stat: F-statistic for nested model comparison
        f_p: p-value for F-test
    """
    r2_discrete: float
    r2_full: float
    delta_r2: float
    f_stat: float
    f_p: float


@dataclass
class EvaluatorResult:
    """
    Result from BatteryEvaluator.evaluate().

    Attributes:
        test_name: Name of the test (e.g., "RSH_1980_2005")
        outcome_name: Name of outcome variable
        n_observations: Number of observations in regression
        n_clusters: Number of clusters (for clustered SEs)
        beta_discrete: Coefficient on discrete exposure (β₁)
        beta_continuous: Coefficient on raw continuous exposure (β₂)
        beta_residual: Coefficient on orthogonalized continuous (β₃)
        model_comparison: R² and F-test statistics
        verdict: Interpretation matrix verdict (+, -, 0)
        controls: List of control variable names
        diagnostics: Additional diagnostic information
    """
    test_name: str
    outcome_name: str
    n_observations: int
    n_clusters: Optional[int]
    beta_discrete: RegressionCoefficients
    beta_continuous: RegressionCoefficients
    beta_residual: RegressionCoefficients
    model_comparison: ModelComparison
    verdict: Verdict
    controls: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "test_name": self.test_name,
            "outcome_name": self.outcome_name,
            "n_observations": self.n_observations,
            "n_clusters": self.n_clusters,
            "beta_discrete": {
                "beta": self.beta_discrete.beta,
                "se": self.beta_discrete.se,
                "t": self.beta_discrete.t,
                "p": self.beta_discrete.p,
            },
            "beta_continuous": {
                "beta": self.beta_continuous.beta,
                "se": self.beta_continuous.se,
                "t": self.beta_continuous.t,
                "p": self.beta_continuous.p,
            },
            "beta_residual": {
                "beta": self.beta_residual.beta,
                "se": self.beta_residual.se,
                "t": self.beta_residual.t,
                "p": self.beta_residual.p,
            },
            "model_comparison": {
                "r2_discrete": self.model_comparison.r2_discrete,
                "r2_full": self.model_comparison.r2_full,
                "delta_r2": self.model_comparison.delta_r2,
                "f_stat": self.model_comparison.f_stat,
                "f_p": self.model_comparison.f_p,
            },
            "verdict": self.verdict.value,
            "controls": self.controls,
            "diagnostics": self.diagnostics,
        }


class BatteryEvaluator:
    """
    Unified regression and scoring evaluator for battery tests.

    Computes the interpretation matrix by comparing discrete (benchmark)
    exposure against continuous (geometry-based) exposure for predicting
    outcome changes.

    Example usage:
        >>> evaluator = BatteryEvaluator(
        ...     outcome=delta_employment,  # (n,) outcome array
        ...     discrete_exposure=rti,     # (n,) RTI from Autor-Dorn
        ...     continuous_exposure=semantic_distance,  # (n,) our measure
        ...     unit_ids=occ_codes,
        ...     cluster_ids=industry_codes,  # optional clustering
        ... )
        >>> result = evaluator.evaluate(
        ...     test_name="RSH_1980_2005",
        ...     outcome_name="delta_log_employment",
        ... )
        >>> print(result.verdict)  # Verdict.POSITIVE, Verdict.NEGATIVE, or Verdict.NULL
    """

    def __init__(
        self,
        outcome: np.ndarray,
        discrete_exposure: np.ndarray,
        continuous_exposure: np.ndarray,
        unit_ids: Optional[List[str]] = None,
        cluster_ids: Optional[np.ndarray] = None,
        controls: Optional[np.ndarray] = None,
        control_names: Optional[List[str]] = None,
        alpha: float = 0.05,
    ):
        """
        Initialize the evaluator.

        Args:
            outcome: (n,) outcome variable (ΔY)
            discrete_exposure: (n,) discrete/benchmark exposure (D)
            continuous_exposure: (n,) continuous/geometry exposure (C)
            unit_ids: Optional list of unit identifiers
            cluster_ids: (n,) cluster IDs for clustered SEs (optional)
            controls: (n, k) control variables (optional)
            control_names: Names for control variables
            alpha: Significance level for verdict determination
        """
        self.outcome = np.asarray(outcome).ravel()
        self.discrete = np.asarray(discrete_exposure).ravel()
        self.continuous = np.asarray(continuous_exposure).ravel()
        self.unit_ids = unit_ids
        self.cluster_ids = cluster_ids
        self.controls = controls
        self.control_names = control_names or []
        self.alpha = alpha

        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input dimensions."""
        n = len(self.outcome)
        if len(self.discrete) != n:
            raise ValueError(f"discrete_exposure length {len(self.discrete)} != outcome length {n}")
        if len(self.continuous) != n:
            raise ValueError(f"continuous_exposure length {len(self.continuous)} != outcome length {n}")
        if self.cluster_ids is not None and len(self.cluster_ids) != n:
            raise ValueError(f"cluster_ids length {len(self.cluster_ids)} != outcome length {n}")
        if self.controls is not None and len(self.controls) != n:
            raise ValueError(f"controls length {len(self.controls)} != outcome length {n}")

    def evaluate(
        self,
        test_name: str,
        outcome_name: str,
    ) -> EvaluatorResult:
        """
        Run the evaluation and compute interpretation matrix verdict.

        The procedure:
        1. Standardize exposures (mean=0, std=1)
        2. Compute C_resid by orthogonalizing C to D
        3. Run full regression: ΔY ~ D + C_resid (+ controls)
        4. Run discrete-only regression for model comparison
        5. Determine verdict from β₃ sign and significance

        Args:
            test_name: Name for this test
            outcome_name: Name of outcome variable

        Returns:
            EvaluatorResult with coefficients, comparison, and verdict
        """
        # Standardize exposures
        D_std = self._standardize(self.discrete)
        C_std = self._standardize(self.continuous)

        # Orthogonalize continuous to discrete
        C_resid = self._orthogonalize(C_std, D_std)

        # Build design matrices
        n = len(self.outcome)
        if self.controls is not None:
            X_full = np.column_stack([np.ones(n), D_std, C_resid, self.controls])
            X_discrete = np.column_stack([np.ones(n), D_std, self.controls])
        else:
            X_full = np.column_stack([np.ones(n), D_std, C_resid])
            X_discrete = np.column_stack([np.ones(n), D_std])

        # Run regressions
        beta_full, se_full, residuals_full = self._ols_with_se(X_full, self.outcome)
        beta_disc, se_disc, residuals_disc = self._ols_with_se(X_discrete, self.outcome)

        # Also run regression with raw continuous for reporting
        X_raw = np.column_stack([np.ones(n), D_std, C_std])
        beta_raw, se_raw, _ = self._ols_with_se(X_raw, self.outcome)

        # Compute R² values
        r2_full = self._compute_r2(self.outcome, residuals_full)
        r2_disc = self._compute_r2(self.outcome, residuals_disc)
        delta_r2 = r2_full - r2_disc

        # F-test for nested models
        f_stat, f_p = self._f_test(
            r2_restricted=r2_disc,
            r2_unrestricted=r2_full,
            n=n,
            k_restricted=X_discrete.shape[1],
            k_unrestricted=X_full.shape[1],
        )

        # Degrees of freedom
        if self.cluster_ids is not None:
            n_clusters = len(np.unique(self.cluster_ids))
            df = n_clusters - 1
        else:
            n_clusters = None
            df = n - X_full.shape[1]

        # Build coefficient results
        def make_coef(beta, se, name, idx):
            t = beta[idx] / se[idx] if se[idx] > 0 else 0.0
            p = 2 * (1 - scipy.stats.t.cdf(abs(t), df=df))
            return RegressionCoefficients(
                beta=float(beta[idx]),
                se=float(se[idx]),
                t=float(t),
                p=float(p),
                name=name,
            )

        beta_discrete_coef = make_coef(beta_full, se_full, "discrete", 1)
        beta_residual_coef = make_coef(beta_full, se_full, "continuous_residual", 2)
        beta_continuous_coef = make_coef(beta_raw, se_raw, "continuous_raw", 2)

        # Determine verdict
        verdict = self._compute_verdict(beta_residual_coef)

        # Model comparison
        model_comparison = ModelComparison(
            r2_discrete=float(r2_disc),
            r2_full=float(r2_full),
            delta_r2=float(delta_r2),
            f_stat=float(f_stat),
            f_p=float(f_p),
        )

        # Diagnostics
        diagnostics = {
            "discrete_continuous_correlation": float(np.corrcoef(D_std, C_std)[0, 1]),
            "outcome_mean": float(self.outcome.mean()),
            "outcome_std": float(self.outcome.std()),
        }

        return EvaluatorResult(
            test_name=test_name,
            outcome_name=outcome_name,
            n_observations=n,
            n_clusters=n_clusters,
            beta_discrete=beta_discrete_coef,
            beta_continuous=beta_continuous_coef,
            beta_residual=beta_residual_coef,
            model_comparison=model_comparison,
            verdict=verdict,
            controls=self.control_names,
            diagnostics=diagnostics,
        )

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        """Standardize to mean=0, std=1."""
        std = x.std()
        if std > 0:
            return (x - x.mean()) / std
        return x - x.mean()

    def _orthogonalize(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Orthogonalize x with respect to z.

        Returns residuals from regressing x on z.
        """
        # x = α + β*z + residual
        z_with_const = np.column_stack([np.ones(len(z)), z])
        beta = np.linalg.lstsq(z_with_const, x, rcond=None)[0]
        return x - z_with_const @ beta

    def _ols_with_se(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        OLS regression with clustered or robust standard errors.

        Returns:
            (beta, se, residuals)
        """
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        if self.cluster_ids is not None:
            se = self._clustered_se(X, residuals)
        else:
            # HC1 robust standard errors
            se = self._robust_se(X, residuals)

        return beta, se, residuals

    def _clustered_se(self, X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        """Cluster-robust standard errors (same as validation.regression)."""
        n, k = X.shape
        unique_clusters = np.unique(self.cluster_ids)
        n_clusters = len(unique_clusters)

        XtX_inv = np.linalg.inv(X.T @ X)

        meat = np.zeros((k, k))
        for c in unique_clusters:
            mask = self.cluster_ids == c
            X_c = X[mask]
            e_c = residuals[mask]
            score_c = X_c.T @ e_c
            meat += np.outer(score_c, score_c)

        adjustment = n_clusters / (n_clusters - 1) * (n - 1) / (n - k)
        V = XtX_inv @ meat @ XtX_inv * adjustment

        return np.sqrt(np.diag(V))

    def _robust_se(self, X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        """HC1 heteroskedasticity-robust standard errors."""
        n, k = X.shape
        XtX_inv = np.linalg.inv(X.T @ X)

        # HC1: multiply each row of X by its residual
        meat = np.zeros((k, k))
        for i in range(n):
            score_i = X[i] * residuals[i]
            meat += np.outer(score_i, score_i)

        adjustment = n / (n - k)
        V = XtX_inv @ meat @ XtX_inv * adjustment

        return np.sqrt(np.diag(V))

    def _compute_r2(self, y: np.ndarray, residuals: np.ndarray) -> float:
        """Compute R² from outcome and residuals."""
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def _f_test(
        self,
        r2_restricted: float,
        r2_unrestricted: float,
        n: int,
        k_restricted: int,
        k_unrestricted: int,
    ) -> tuple[float, float]:
        """F-test for nested models."""
        q = k_unrestricted - k_restricted  # Number of restrictions
        if q <= 0 or r2_unrestricted <= r2_restricted:
            return 0.0, 1.0

        f_stat = ((r2_unrestricted - r2_restricted) / q) / \
                 ((1 - r2_unrestricted) / (n - k_unrestricted))

        f_p = 1 - scipy.stats.f.cdf(f_stat, q, n - k_unrestricted)

        return f_stat, f_p

    def _compute_verdict(self, beta_resid: RegressionCoefficients) -> Verdict:
        """
        Determine interpretation matrix verdict.

        Logic:
            If p < alpha and beta > 0: POSITIVE (geometry adds signal)
            If p < alpha and beta < 0: NEGATIVE (geometry contradicts)
            Otherwise: NULL (no significant residual)
        """
        if beta_resid.p < self.alpha:
            if beta_resid.beta > 0:
                return Verdict.POSITIVE
            else:
                return Verdict.NEGATIVE
        return Verdict.NULL
