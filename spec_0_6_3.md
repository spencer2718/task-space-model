---

# Specification v0.6.3: Codebase Consolidation + Phase II Infrastructure

**Status:** APPROVED  
**Prerequisite:** v0.6.2 (Validation Complete)  
**Objective:** Eliminate technical debt and build extensible infrastructure for Phase II experiments  
**Scope:** Infrastructure and architecture—no new empirical results

---

## Executive Summary

This specification consolidates the codebase from organic growth during v0.4–v0.6.2 into a clean, modular architecture. The key additions are:

1. **Registry Pattern** — Extensible shock profiles and similarity measures without modifying runner code
2. **Unified Artifact Store** — Single cache location for embeddings and distance matrices
3. **Package Installation** — Proper `pyproject.toml` for `pip install -e .`
4. **DataFrame Controls** — Regression infrastructure supporting arbitrary control variables
5. **Propagation Module** — Correct implementation of $I_t \to A_t \to E_j$ pipeline

Upon completion, a new experiment (including novel shock profiles) should require ~30 lines of YAML configuration plus a registered Python function, not a 600-line script.

---

## Architectural Clarifications (From Chief Interrogator)

### Overlap vs Exposure

These are distinct mathematical objects:

| Object | Formula | Shape | Meaning |
|--------|---------|-------|---------|
| **Overlap** | $O = \rho K \rho^T$ | $(N_{occ}, N_{occ})$ | Pairwise occupation similarity through kernel |
| **Exposure** | $E = \rho (K I_t)$ | $(N_{occ}, 1)$ | Occupation exposure to a specific shock |

Phase I validated **Overlap**. Phase II will compute **Exposure**.

### Propagation Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Kernel reuse | Same $K$ as Phase I | Manifold geometry doesn't change with shock |
| σ calibration | NN-median (0.223) | Per Phase I findings |
| Row-normalize $K$? | **No** | Row-normalization dilutes shocks in dense regions |
| Exposure formula | $E = \rho \cdot (K \cdot I_t)$ | Accumulation, not averaging |

---

## Task Breakdown

### Task 1: Delete Dead Code

**Objective:** Remove modules that are no longer used.

**Files to delete:**
```
src/task_space/sae.py           # SAE was marginal, abandoned
src/task_space/validation.py    # Replaced by diagnostics_v061.py
src/task_space/diagnostics.py   # Replaced by diagnostics_v061.py
src/task_space/kernel.py        # Row-normalizes by default (the bug)
src/task_space/baseline.py      # Reimplemented in comparison.py
```

**Verification:**
```bash
# After deletion, these must still work:
PYTHONPATH=src python tests/run_phase1_fix.py
PYTHONPATH=src python tests/run_phase2_comparison.py
PYTHONPATH=src python tests/run_phase2_robustness.py
```

**Deliverable:** Commit `chore: remove dead modules`

---

### Task 2: Create Package Structure

**Objective:** Reorganize into logical subpackages.

#### 2.1 Create `data/` subpackage

```
src/task_space/data/
    __init__.py
    onet.py          # From: data.py (O*NET loading functions)
    oes.py           # NEW: OES wage data loading, comovement computation
    crosswalk.py     # From: crosswalk.py (unchanged)
    artifacts.py     # NEW: Unified cache management
```

**`data/__init__.py`:**
```python
from .onet import load_onet_data, get_dwa_titles, get_task_ratings
from .oes import load_oes_panel, compute_wage_comovement
from .crosswalk import build_onet_oes_crosswalk, onet_to_soc
from .artifacts import get_embeddings, get_distance_matrix, clear_cache

__all__ = [
    'load_onet_data', 'get_dwa_titles', 'get_task_ratings',
    'load_oes_panel', 'compute_wage_comovement',
    'build_onet_oes_crosswalk', 'onet_to_soc',
    'get_embeddings', 'get_distance_matrix', 'clear_cache',
]
```

**`data/artifacts.py`:**
```python
"""
Unified artifact store for expensive computations.

IMPORTANT: This is THE canonical location for cached embeddings and distances.
Do not create .npy files in outputs/ or elsewhere.

Cache location: .cache/artifacts/v1/
"""

from pathlib import Path
from typing import Optional
import hashlib
import json
import numpy as np

CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".cache" / "artifacts"
CACHE_VERSION = "v1"


def _hash_texts(texts: list[str]) -> str:
    """Create deterministic hash of text list."""
    h = hashlib.sha256()
    for t in sorted(texts):  # Sort for determinism
        h.update(t.encode('utf-8'))
    return h.hexdigest()[:16]


def _get_cache_path(artifact_type: str, identifier: str) -> Path:
    """Get canonical path for an artifact."""
    cache_dir = CACHE_DIR / CACHE_VERSION / artifact_type
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{identifier}.npz"


def get_embeddings(
    texts: list[str],
    model: str = "all-mpnet-base-v2",
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Get text embeddings, using cache if available.
    
    This is THE canonical way to get embeddings. Do not compute elsewhere.
    
    Args:
        texts: List of texts to embed
        model: Sentence transformer model name
        force_recompute: Bypass cache
        
    Returns:
        (n_texts, embedding_dim) array
    """
    text_hash = _hash_texts(texts)
    cache_path = _get_cache_path("embeddings", f"{model}_{text_hash}")
    
    if not force_recompute and cache_path.exists():
        data = np.load(cache_path)
        return data['embeddings']
    
    # Compute
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(model)
    embeddings = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Cache
    np.savez_compressed(cache_path, embeddings=embeddings, model=model, n_texts=len(texts))
    
    return embeddings


def get_distance_matrix(
    embeddings: np.ndarray,
    metric: str = "cosine",
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Get distance matrix, using cache if available.
    
    Args:
        embeddings: (n, d) embedding matrix
        metric: Distance metric ('cosine' or 'euclidean')
        force_recompute: Bypass cache
        
    Returns:
        (n, n) distance matrix
    """
    # Hash based on embedding shape and first/last values (fast proxy)
    emb_id = f"{embeddings.shape}_{embeddings[0,0]:.6f}_{embeddings[-1,-1]:.6f}"
    emb_hash = hashlib.sha256(emb_id.encode()).hexdigest()[:16]
    cache_path = _get_cache_path("distances", f"{metric}_{emb_hash}")
    
    if not force_recompute and cache_path.exists():
        data = np.load(cache_path)
        return data['distances']
    
    # Compute
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
    if metric == "cosine":
        distances = cosine_distances(embeddings)
    elif metric == "euclidean":
        distances = euclidean_distances(embeddings)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Cache
    np.savez_compressed(cache_path, distances=distances, metric=metric)
    
    return distances


def clear_cache(artifact_type: Optional[str] = None) -> int:
    """
    Clear cached artifacts.
    
    Args:
        artifact_type: 'embeddings', 'distances', or None for all
        
    Returns:
        Number of files deleted
    """
    if artifact_type:
        target = CACHE_DIR / CACHE_VERSION / artifact_type
    else:
        target = CACHE_DIR / CACHE_VERSION
    
    count = 0
    if target.exists():
        for f in target.rglob("*.npz"):
            f.unlink()
            count += 1
    return count
```

#### 2.2 Create `similarity/` subpackage

```
src/task_space/similarity/
    __init__.py
    embeddings.py    # Text embedding computation (uses artifacts.py)
    kernel.py        # Kernel construction with correct defaults
    overlap.py       # Jaccard, kernel overlap, normalized overlap
    distances.py     # Distance matrix computation
    registry.py      # Registry for similarity measures
```

**`similarity/kernel.py`:**
```python
"""
Kernel construction with correct defaults.

CRITICAL LESSONS FROM PHASE 1:
1. σ must be calibrated to nearest-neighbor distances, NOT global distribution
2. Do NOT row-normalize the kernel matrix
3. Discrimination ratio must be > 3x to avoid collapse
"""

import warnings
import numpy as np


def calibrate_sigma(
    dist_matrix: np.ndarray,
    method: str = 'nn_median',
) -> float:
    """
    Calibrate kernel bandwidth to local distance structure.
    
    Args:
        dist_matrix: (n, n) pairwise distance matrix
        method: Only 'nn_median' is supported
        
    Returns:
        Calibrated σ value
        
    Raises:
        ValueError: If method != 'nn_median'
    """
    if method != 'nn_median':
        raise ValueError(
            f"Method '{method}' is not supported. Use 'nn_median'. "
            "Global percentile methods cause kernel collapse (see v0.5.0 postmortem). "
            "The NN-median method is the only validated approach."
        )
    
    dm = dist_matrix.copy()
    np.fill_diagonal(dm, np.inf)
    nn_dists = dm.min(axis=1)
    return float(np.median(nn_dists))


def check_kernel_discrimination(
    dist_matrix: np.ndarray,
    sigma: float,
    min_ratio: float = 3.0,
) -> tuple[float, bool]:
    """
    Check if kernel discriminates between close and distant pairs.
    
    Args:
        dist_matrix: (n, n) pairwise distance matrix
        sigma: Kernel bandwidth
        min_ratio: Minimum acceptable discrimination ratio
        
    Returns:
        (discrimination_ratio, is_acceptable)
    """
    d_flat = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    d_p10 = np.percentile(d_flat, 10)
    d_p90 = np.percentile(d_flat, 90)
    
    w_p10 = np.exp(-d_p10 / sigma)
    w_p90 = np.exp(-d_p90 / sigma)
    
    ratio = w_p10 / w_p90 if w_p90 > 0 else np.inf
    return ratio, ratio >= min_ratio


def build_kernel_matrix(
    dist_matrix: np.ndarray,
    sigma: float = None,
    kernel_type: str = 'exponential',
    row_normalize: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Build kernel matrix from distance matrix.
    
    Args:
        dist_matrix: (n, n) pairwise distance matrix
        sigma: Bandwidth. If None, auto-calibrate via NN-median.
        kernel_type: 'exponential' or 'gaussian'
        row_normalize: Whether to row-normalize. 
            DEFAULT IS FALSE. This is intentional.
            
    Returns:
        (kernel_matrix, sigma_used)
        
    Warnings:
        Emits UserWarning if row_normalize=True
    """
    if sigma is None:
        sigma = calibrate_sigma(dist_matrix)
    
    # Check discrimination
    ratio, ok = check_kernel_discrimination(dist_matrix, sigma)
    if not ok:
        warnings.warn(
            f"Kernel discrimination ratio ({ratio:.2f}) is below 3.0. "
            "This may indicate kernel collapse. Consider smaller σ.",
            UserWarning
        )
    
    # Build kernel
    if kernel_type == 'exponential':
        K = np.exp(-dist_matrix / sigma)
    elif kernel_type == 'gaussian':
        K = np.exp(-dist_matrix**2 / (2 * sigma**2))
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    if row_normalize:
        warnings.warn(
            "Row normalization destroys signal with large activity counts. "
            "This was the root cause of kernel collapse in v0.5.0. "
            "Only use row_normalize=True if you understand the consequences.",
            UserWarning
        )
        row_sums = K.sum(axis=1, keepdims=True)
        K = K / row_sums
    
    return K, sigma
```

**`similarity/overlap.py`:**
```python
"""
Overlap measures between occupation distributions.

Three measures:
1. Jaccard: Binary set overlap (baseline)
2. Kernel overlap: ρ_i^T K ρ_j (unnormalized)
3. Normalized overlap: Kernel overlap / sqrt(self-overlaps) (controls concentration)
"""

import numpy as np


def compute_jaccard_overlap(occ_measures: np.ndarray) -> np.ndarray:
    """
    Compute binary Jaccard overlap between occupations.
    
    Jaccard(i,j) = |A_i ∩ A_j| / |A_i ∪ A_j|
    
    Args:
        occ_measures: (n_occ, n_act) occupation measure matrix
        
    Returns:
        (n_occ, n_occ) Jaccard similarity matrix
    """
    binary = (occ_measures > 0).astype(float)
    
    # Intersection: element-wise min
    intersection = binary @ binary.T
    
    # Union: |A| + |B| - |A ∩ B|
    support_sizes = binary.sum(axis=1)
    union = support_sizes[:, None] + support_sizes[None, :] - intersection
    
    # Avoid division by zero
    union = np.maximum(union, 1e-10)
    
    return intersection / union


def compute_kernel_overlap(
    occ_measures: np.ndarray,
    kernel_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute unnormalized kernel-weighted overlap.
    
    Overlap(i,j) = ρ_i^T K ρ_j
    
    This is RAW overlap. For concentration-controlled overlap,
    use compute_normalized_overlap().
    
    Args:
        occ_measures: (n_occ, n_act) occupation probability measures
        kernel_matrix: (n_act, n_act) kernel matrix (NOT row-normalized)
        
    Returns:
        (n_occ, n_occ) overlap matrix
    """
    return occ_measures @ kernel_matrix @ occ_measures.T


def compute_normalized_overlap(
    occ_measures: np.ndarray,
    kernel_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized (cosine-style) kernel overlap.
    
    NormOverlap(i,j) = (ρ_i^T K ρ_j) / sqrt((ρ_i^T K ρ_i)(ρ_j^T K ρ_j))
    
    This controls for concentration effects (specialist vs generalist).
    Per v0.6.2 findings, normalization IMPROVES R² by ~57%.
    
    Args:
        occ_measures: (n_occ, n_act) occupation probability measures
        kernel_matrix: (n_act, n_act) kernel matrix
        
    Returns:
        (n_occ, n_occ) normalized overlap matrix, values in [0, 1]
    """
    raw = compute_kernel_overlap(occ_measures, kernel_matrix)
    self_overlap = np.diag(raw)
    
    norm_factor = np.sqrt(np.outer(self_overlap, self_overlap))
    norm_factor = np.maximum(norm_factor, 1e-10)
    
    return raw / norm_factor
```

#### 2.3 Create `shocks/` subpackage

```
src/task_space/shocks/
    __init__.py
    registry.py      # Registry pattern for shock profiles
    profiles.py      # Built-in shock profiles (uniform, gaussian, v1, v2, v3)
    propagation.py   # I_t → A_t → E_j pipeline
```

**`shocks/registry.py`:**
```python
"""
Registry pattern for extensible shock profiles.

To add a new shock:
1. Define a function that takes (domain, **kwargs) and returns np.ndarray
2. Decorate with @register_shock(name, description, ...)
3. Use in config: shock: "your_name"
"""

from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

# Type: (domain, **kwargs) -> (n_activities,) shock intensity array
ShockProfileFn = Callable[..., np.ndarray]


@dataclass
class RegisteredShock:
    name: str
    fn: ShockProfileFn
    description: str
    required_args: list[str]
    optional_args: dict[str, Any]


SHOCK_REGISTRY: dict[str, RegisteredShock] = {}


def register_shock(
    name: str,
    description: str = "",
    required_args: list[str] = None,
    optional_args: dict[str, Any] = None,
):
    """Decorator to register a shock profile function."""
    def decorator(fn: ShockProfileFn) -> ShockProfileFn:
        SHOCK_REGISTRY[name] = RegisteredShock(
            name=name,
            fn=fn,
            description=description,
            required_args=required_args or [],
            optional_args=optional_args or {},
        )
        return fn
    return decorator


def get_shock(name: str) -> RegisteredShock:
    """Retrieve registered shock by name."""
    if name not in SHOCK_REGISTRY:
        available = list(SHOCK_REGISTRY.keys())
        raise ValueError(f"Unknown shock '{name}'. Available: {available}")
    return SHOCK_REGISTRY[name]


def list_shocks() -> list[str]:
    """List all registered shock profile names."""
    return list(SHOCK_REGISTRY.keys())
```

**`shocks/profiles.py`:**
```python
"""
Built-in shock profiles.

These implement the candidate profiles from the paper:
- uniform: Baseline uniform shock
- gaussian_directed: Shock centered on specific activity (Example 3.4)
- capability_v1: AI capability-only (positive cognitive, negative manual)
- capability_v2: AI capability + structure (adds routine loading)
- rbtc: Routine-biased technological change
"""

import numpy as np
from .registry import register_shock


@register_shock(
    name="uniform",
    description="Uniform shock intensity across all activities",
    optional_args={"intensity": 1.0},
)
def shock_uniform(domain, intensity: float = 1.0, **kwargs) -> np.ndarray:
    """I(a) = intensity for all a."""
    return np.full(len(domain.activity_ids), intensity)


@register_shock(
    name="gaussian_directed",
    description="Gaussian shock centered on activity (Theory Example 3.4)",
    required_args=["center_idx", "dist_matrix"],
    optional_args={"sigma_shock": 0.1, "intensity": 1.0},
)
def shock_gaussian_directed(
    domain,
    center_idx: int,
    dist_matrix: np.ndarray,
    sigma_shock: float = 0.1,
    intensity: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Gaussian shock centered on specific activity.
    
    I(a) = intensity * exp(-d(a, center)² / 2σ²)
    
    Args:
        domain: Activity domain with activity_ids
        center_idx: Index of center activity in domain
        dist_matrix: (n_act, n_act) distance matrix
        sigma_shock: Shock spread parameter (NOT kernel bandwidth)
        intensity: Peak intensity at center
    """
    distances_from_center = dist_matrix[center_idx, :]
    return intensity * np.exp(-distances_from_center**2 / (2 * sigma_shock**2))


@register_shock(
    name="capability_v1",
    description="AI capability-only: positive cognitive/language, negative manual",
    required_args=["activity_classifications"],
    optional_args={"cognitive_weight": 1.0, "manual_weight": -0.5},
)
def shock_capability_v1(
    domain,
    activity_classifications: dict[str, str],
    cognitive_weight: float = 1.0,
    manual_weight: float = -0.5,
    **kwargs,
) -> np.ndarray:
    """
    v1 shock profile: capability-only.
    
    Positive loading on cognitive/language activities.
    Negative loading on manual/physical activities.
    Neutral (0) on unclassified activities.
    
    Args:
        domain: Activity domain
        activity_classifications: Map activity_id -> classification
            Classifications: 'cognitive', 'language', 'manual', 'physical', etc.
    """
    I = np.zeros(len(domain.activity_ids))
    
    cognitive_tags = {'cognitive', 'language', 'information_processing', 'analytical'}
    manual_tags = {'manual', 'physical', 'motor'}
    
    for i, act_id in enumerate(domain.activity_ids):
        classification = activity_classifications.get(act_id, '').lower()
        if classification in cognitive_tags:
            I[i] = cognitive_weight
        elif classification in manual_tags:
            I[i] = manual_weight
    
    return I


@register_shock(
    name="capability_v2",
    description="AI capability + structure: v1 plus routine amplification",
    required_args=["activity_classifications", "routine_scores"],
    optional_args={"cognitive_weight": 1.0, "manual_weight": -0.5, "routine_amplifier": 0.5},
)
def shock_capability_v2(
    domain,
    activity_classifications: dict[str, str],
    routine_scores: np.ndarray,
    cognitive_weight: float = 1.0,
    manual_weight: float = -0.5,
    routine_amplifier: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """
    v2 shock profile: capability + structure.
    
    v1 loadings plus additional weight on structured/routine activities.
    Hypothesis: structure amplifies AI exposure beyond raw capability.
    """
    I_v1 = shock_capability_v1(
        domain, activity_classifications, cognitive_weight, manual_weight
    )
    
    # Amplify where routine score is high
    routine_normalized = (routine_scores - routine_scores.min()) / (routine_scores.max() - routine_scores.min() + 1e-10)
    
    return I_v1 + routine_amplifier * routine_normalized


@register_shock(
    name="rbtc",
    description="Routine-biased technological change (retrospective evaluation)",
    required_args=["routine_scores"],
    optional_args={"intensity": 1.0},
)
def shock_rbtc(
    domain,
    routine_scores: np.ndarray,
    intensity: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    RBTC shock for retrospective (1990-2007) evaluation.
    
    Positive loading proportional to routine/repetitive score.
    """
    return intensity * routine_scores
```

**`shocks/propagation.py`:**
```python
"""
Shock propagation: I_t → A_t → E_j

Mathematical pipeline (from Theory section):
1. I_t: (n_act,) shock profile over activities
2. A_t = K @ I_t: (n_act,) propagated/displaced field
3. E_j = ρ_j @ A_t: (n_occ,) occupation exposure vector

CRITICAL DESIGN DECISIONS (per Chief Interrogator):
- Use same K as Phase 1 (NN-median σ, exponential, NOT row-normalized)
- Do NOT row-normalize K for propagation (accumulation, not averaging)
- Exposure is ρ @ (K @ I), NOT (ρ @ K) @ I (though mathematically equivalent)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PropagationResult:
    """Results from shock propagation."""
    I_t: np.ndarray          # (n_act,) raw shock profile
    A_t: np.ndarray          # (n_act,) propagated field
    E: np.ndarray            # (n_occ,) occupation exposures
    sigma: float             # Kernel bandwidth used
    shock_name: str          # Name of shock profile


def propagate_shock(
    I_t: np.ndarray,
    kernel_matrix: np.ndarray,
    occ_measures: np.ndarray,
    sigma: float,
    shock_name: str = "unknown",
) -> PropagationResult:
    """
    Propagate shock through task space to occupation exposures.
    
    Pipeline:
        I_t → A_t = K @ I_t → E = ρ @ A_t
    
    Args:
        I_t: (n_act,) shock profile over activities
        kernel_matrix: (n_act, n_act) kernel matrix (NOT row-normalized)
        occ_measures: (n_occ, n_act) occupation probability measures
        sigma: Kernel bandwidth (for metadata)
        shock_name: Name of shock profile (for metadata)
        
    Returns:
        PropagationResult with I_t, A_t, E, and metadata
    """
    # Propagate shock through kernel
    A_t = kernel_matrix @ I_t
    
    # Aggregate to occupation-level exposure
    E = occ_measures @ A_t
    
    return PropagationResult(
        I_t=I_t,
        A_t=A_t,
        E=E,
        sigma=sigma,
        shock_name=shock_name,
    )


def compute_exposure_from_shock(
    domain,
    occ_measures: np.ndarray,
    shock_name: str,
    shock_args: dict,
    kernel_matrix: np.ndarray,
    sigma: float,
) -> PropagationResult:
    """
    Compute occupation exposures from a registered shock profile.
    
    Convenience function that:
    1. Retrieves shock from registry
    2. Computes I_t
    3. Propagates to E
    
    Args:
        domain: Activity domain
        occ_measures: (n_occ, n_act) occupation measures
        shock_name: Registry key for shock profile
        shock_args: Arguments to pass to shock function
        kernel_matrix: (n_act, n_act) kernel
        sigma: Kernel bandwidth
        
    Returns:
        PropagationResult
    """
    from .registry import get_shock
    
    shock = get_shock(shock_name)
    I_t = shock.fn(domain, **shock_args)
    
    return propagate_shock(I_t, kernel_matrix, occ_measures, sigma, shock_name)
```

#### 2.4 Create `validation/` subpackage

```
src/task_space/validation/
    __init__.py
    regression.py    # Single implementation of clustered SE regression
    diagnostics.py   # Kernel diagnostics (from diagnostics_v061.py)
    permutation.py   # Permutation tests, cross-validation
```

**`validation/regression.py`:**
```python
"""
Validation regression utilities.

SINGLE implementation of clustered standard errors.
All validation code must use this module.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import scipy.stats


@dataclass
class RegressionResult:
    """Results from validation regression."""
    beta: np.ndarray           # Coefficients
    se: np.ndarray             # Standard errors (clustered)
    t: np.ndarray              # t-statistics
    p: np.ndarray              # p-values
    r2: float                  # R-squared
    n_pairs: int               # Number of observations
    n_clusters: int            # Number of clusters
    variable_names: list[str]  # Names of variables


def compute_clustered_se(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    """
    Compute cluster-robust standard errors.
    
    THIS IS THE SINGLE IMPLEMENTATION. Use this everywhere.
    
    Args:
        X: (n, k) design matrix (must include constant if desired)
        residuals: (n,) OLS residuals
        cluster_ids: (n,) cluster identifiers
        
    Returns:
        (k,) standard errors
    """
    n, k = X.shape
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    
    # Bread: (X'X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)
    
    # Meat: sum of cluster score outer products
    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = cluster_ids == c
        X_c = X[mask]
        e_c = residuals[mask]
        score_c = X_c.T @ e_c
        meat += np.outer(score_c, score_c)
    
    # Small-sample adjustment (HC1-style)
    adjustment = n_clusters / (n_clusters - 1) * (n - 1) / (n - k)
    
    # Sandwich
    V = XtX_inv @ meat @ XtX_inv * adjustment
    
    return np.sqrt(np.diag(V))


def run_validation_regression(
    similarity: np.ndarray,
    comovement: np.ndarray,
    sim_codes: list[str],
    comove_codes: list[str],
    crosswalk: dict[str, str],
    cluster_by: str = 'origin',
    controls: Optional[pd.DataFrame] = None,
) -> RegressionResult:
    """
    Run validation regression of comovement on similarity.
    
    Args:
        similarity: (n_sim, n_sim) similarity/overlap matrix
        comovement: (n_comove, n_comove) comovement matrix
        sim_codes: O*NET-SOC codes for similarity rows/cols
        comove_codes: SOC codes for comovement rows/cols
        crosswalk: O*NET-SOC → SOC mapping
        cluster_by: 'origin' or 'destination'
        controls: Optional DataFrame with control variables
            Must have 'origin_soc', 'dest_soc' columns for merging
            
    Returns:
        RegressionResult
    """
    # Build pair dataset
    pairs = _build_pair_dataset(similarity, comovement, sim_codes, comove_codes, crosswalk)
    
    # Merge controls if provided
    control_cols = []
    if controls is not None:
        pairs = pairs.merge(controls, on=['origin_soc', 'dest_soc'], how='left')
        control_cols = [c for c in controls.columns if c not in ['origin_soc', 'dest_soc']]
    
    # Design matrix
    y = pairs['y'].values
    X_data = pairs[['x'] + control_cols].values
    X = np.column_stack([np.ones(len(y)), X_data])
    variable_names = ['const', 'similarity'] + control_cols
    
    # OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    
    # Clustered SEs
    cluster_ids = pairs['origin_soc' if cluster_by == 'origin' else 'dest_soc'].values
    se = compute_clustered_se(X, residuals, cluster_ids)
    
    # Statistics
    t = beta / se
    p = 2 * (1 - scipy.stats.t.cdf(np.abs(t), df=len(y) - len(beta)))
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    
    return RegressionResult(
        beta=beta,
        se=se,
        t=t,
        p=p,
        r2=r2,
        n_pairs=len(y),
        n_clusters=len(np.unique(cluster_ids)),
        variable_names=variable_names,
    )


def _build_pair_dataset(
    similarity: np.ndarray,
    comovement: np.ndarray,
    sim_codes: list[str],
    comove_codes: list[str],
    crosswalk: dict[str, str],
) -> pd.DataFrame:
    """Build DataFrame of occupation pairs with similarity and comovement."""
    # Create mappings
    sim_code_to_idx = {c: i for i, c in enumerate(sim_codes)}
    comove_code_to_idx = {c: i for i, c in enumerate(comove_codes)}
    
    pairs = []
    for i, onet_i in enumerate(sim_codes):
        soc_i = crosswalk.get(onet_i)
        if soc_i is None or soc_i not in comove_code_to_idx:
            continue
        ci = comove_code_to_idx[soc_i]
        
        for j, onet_j in enumerate(sim_codes):
            if i >= j:
                continue
            soc_j = crosswalk.get(onet_j)
            if soc_j is None or soc_j not in comove_code_to_idx:
                continue
            cj = comove_code_to_idx[soc_j]
            
            pairs.append({
                'origin_soc': soc_i,
                'dest_soc': soc_j,
                'x': similarity[i, j],
                'y': comovement[ci, cj],
            })
    
    return pd.DataFrame(pairs)
```

#### 2.5 Create `experiments/` subpackage

```
src/task_space/experiments/
    __init__.py
    config.py        # Experiment configuration schema
    runner.py        # Generic experiment execution
```

**`experiments/config.py`:**
```python
"""
Experiment configuration.

Configs are YAML files that specify:
- Data sources
- Similarity measure (registry-based)
- Shock profile (registry-based, optional)
- Validation parameters
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
import yaml


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    
    # Identity
    name: str
    version: str = "0.6.3"
    description: str = ""
    
    # Data
    onet_path: Path = field(default_factory=lambda: Path("data/onet/db_30_0_excel"))
    oes_path: Path = field(default_factory=lambda: Path("data/external/oes"))
    output_dir: Path = field(default_factory=lambda: Path("outputs/experiments"))
    
    # Similarity (registry key + args)
    similarity: str = "normalized_kernel"
    similarity_args: dict[str, Any] = field(default_factory=dict)
    
    # Shock (registry key + args, optional for validation-only)
    shock: Optional[str] = None
    shock_args: dict[str, Any] = field(default_factory=dict)
    
    # Validation
    target: str = "wage_comovement"
    oes_years: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023)
    cluster_by: str = "origin"
    
    # Controls
    controls: list[str] = field(default_factory=list)
    
    # Robustness
    run_permutation: bool = True
    n_permutations: int = 1000
    run_cv: bool = True
    n_folds: int = 5
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'ExperimentConfig':
        with open(path) as f:
            data = yaml.safe_load(f)
        
        # Path conversion
        for key in ('onet_path', 'oes_path', 'output_dir'):
            if key in data and data[key]:
                data[key] = Path(data[key])
        
        # Tuple conversion
        if 'oes_years' in data:
            data['oes_years'] = tuple(data['oes_years'])
        
        return cls(**data)
    
    def to_yaml(self, path: Path) -> None:
        data = asdict(self)
        for key in ('onet_path', 'oes_path', 'output_dir'):
            data[key] = str(data[key])
        data['oes_years'] = list(data['oes_years'])
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
```

**`experiments/runner.py`:**
```python
"""
Generic experiment runner.

Executes experiments defined by YAML configs.
"""

import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import ExperimentConfig


def run_experiment(config: ExperimentConfig) -> dict:
    """
    Execute an experiment from configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Results dictionary
    """
    from ..domain import build_dwa_occupation_measures
    from ..data import load_oes_panel, compute_wage_comovement, build_onet_oes_crosswalk
    from ..data.artifacts import get_embeddings, get_distance_matrix
    from ..similarity.kernel import build_kernel_matrix
    from ..similarity.overlap import compute_jaccard_overlap, compute_kernel_overlap, compute_normalized_overlap
    from ..validation.regression import run_validation_regression
    from ..validation.permutation import run_permutation_test, run_cross_validation
    
    results = {
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        'git_commit': _get_git_commit(),
        'timestamp': datetime.utcnow().isoformat(),
    }
    
    # 1. Load data
    print(f"[1/6] Loading data...")
    measures = build_dwa_occupation_measures(config.onet_path)
    oes_panel = load_oes_panel(config.oes_path, list(config.oes_years))
    comovement = compute_wage_comovement(oes_panel)
    crosswalk = build_onet_oes_crosswalk(measures.occupation_codes, comovement.columns.tolist())
    
    results['data'] = {
        'n_occupations': len(measures.occupation_codes),
        'n_activities': len(measures.activity_ids),
        'n_comovement_codes': len(comovement.columns),
        'crosswalk_coverage': len(crosswalk),
    }
    
    # 2. Compute similarity
    print(f"[2/6] Computing similarity ({config.similarity})...")
    if config.similarity == 'jaccard':
        similarity = compute_jaccard_overlap(measures.occupation_matrix)
        sigma = None
    else:
        embeddings = get_embeddings(measures.activity_titles)
        dist_matrix = get_distance_matrix(embeddings)
        K, sigma = build_kernel_matrix(dist_matrix)
        
        if config.similarity == 'kernel':
            similarity = compute_kernel_overlap(measures.occupation_matrix, K)
        elif config.similarity == 'normalized_kernel':
            similarity = compute_normalized_overlap(measures.occupation_matrix, K)
        else:
            raise ValueError(f"Unknown similarity: {config.similarity}")
    
    results['similarity'] = {
        'type': config.similarity,
        'sigma': sigma,
    }
    
    # 3. Compute shock exposure if specified
    if config.shock:
        print(f"[3/6] Computing shock exposure ({config.shock})...")
        from ..shocks.propagation import compute_exposure_from_shock
        
        prop_result = compute_exposure_from_shock(
            measures, measures.occupation_matrix,
            config.shock, config.shock_args,
            K, sigma,
        )
        results['shock'] = {
            'type': config.shock,
            'args': config.shock_args,
            'exposure_stats': {
                'min': float(prop_result.E.min()),
                'max': float(prop_result.E.max()),
                'mean': float(prop_result.E.mean()),
                'std': float(prop_result.E.std()),
            }
        }
    else:
        print(f"[3/6] No shock specified (validation-only mode)")
    
    # 4. Run regression
    print(f"[4/6] Running validation regression...")
    reg_result = run_validation_regression(
        similarity, comovement.values,
        measures.occupation_codes, comovement.columns.tolist(),
        crosswalk, cluster_by=config.cluster_by,
    )
    results['regression'] = {
        'beta': float(reg_result.beta[1]),  # similarity coefficient
        'se': float(reg_result.se[1]),
        't': float(reg_result.t[1]),
        'p': float(reg_result.p[1]),
        'r2': float(reg_result.r2),
        'n_pairs': reg_result.n_pairs,
        'n_clusters': reg_result.n_clusters,
    }
    
    # 5. Robustness
    if config.run_permutation:
        print(f"[5/6] Running permutation test (n={config.n_permutations})...")
        perm_result = run_permutation_test(
            similarity, comovement.values,
            measures.occupation_codes, comovement.columns.tolist(),
            crosswalk, n_permutations=config.n_permutations, seed=config.seed,
        )
        results['permutation'] = asdict(perm_result)
    
    if config.run_cv:
        print(f"[6/6] Running cross-validation (k={config.n_folds})...")
        cv_result = run_cross_validation(
            similarity, comovement.values,
            measures.occupation_codes, comovement.columns.tolist(),
            crosswalk, n_folds=config.n_folds, seed=config.seed,
        )
        results['cross_validation'] = asdict(cv_result)
    
    # Save results
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / f"{config.name}_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    print(f"  R² = {results['regression']['r2']:.5f}")
    print(f"  t  = {results['regression']['t']:.2f}")
    
    return results


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return 'unknown'
```

---

### Task 3: Package Installation

**Objective:** Enable `pip install -e .` for notebook imports.

**`pyproject.toml`:**
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "task_space"
version = "0.6.3"
description = "Geometric framework for labor market exposure measurement"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "Research Use Only"}

dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "openpyxl>=3.1",
    "torch>=2.0",
    "sentence-transformers>=2.2",
    "transformers>=4.30",
    "pyyaml>=6.0",
    "tqdm>=4.65",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
]
notebooks = [
    "jupyter>=1.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

**Verification:**
```bash
pip install -e ".[dev,notebooks]"
python -c "from task_space import build_dwa_occupation_measures; print('OK')"
```

---

### Task 4: Unit Tests

**Objective:** Add real pytest tests for critical functionality.

```
tests/
    unit/
        test_kernel.py
        test_overlap.py
        test_regression.py
        test_propagation.py
    integration/
        test_validation_pipeline.py
```

**`tests/unit/test_kernel.py`:**
```python
import numpy as np
import pytest
from task_space.similarity.kernel import calibrate_sigma, build_kernel_matrix, check_kernel_discrimination


class TestCalibrateSigma:
    def test_nn_median(self):
        """σ = median of nearest-neighbor distances."""
        dist = np.array([
            [0.0, 0.1, 0.5],
            [0.1, 0.0, 0.4],
            [0.5, 0.4, 0.0],
        ])
        # NN: [0.1, 0.1, 0.4] → median = 0.1
        sigma = calibrate_sigma(dist)
        assert sigma == pytest.approx(0.1)
    
    def test_rejects_global(self):
        """Reject non-NN methods."""
        dist = np.random.rand(5, 5)
        with pytest.raises(ValueError, match="not supported"):
            calibrate_sigma(dist, method='global_median')


class TestBuildKernelMatrix:
    def test_exponential(self):
        """K_ij = exp(-d_ij / σ)."""
        dist = np.array([[0.0, 1.0], [1.0, 0.0]])
        K, sigma = build_kernel_matrix(dist, sigma=1.0)
        
        assert K[0, 0] == pytest.approx(1.0)
        assert K[0, 1] == pytest.approx(np.exp(-1))
    
    def test_auto_sigma(self):
        """Auto-calibrates when sigma=None."""
        dist = np.random.rand(10, 10)
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        
        K, sigma = build_kernel_matrix(dist, sigma=None)
        assert sigma > 0
    
    def test_row_normalize_warning(self):
        """Warns when row_normalize=True."""
        dist = np.random.rand(5, 5)
        with pytest.warns(UserWarning, match="destroy signal"):
            build_kernel_matrix(dist, sigma=0.1, row_normalize=True)


class TestDiscrimination:
    def test_good_discrimination(self):
        """High ratio when σ is small relative to distances."""
        dist = np.random.rand(100, 100)
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        
        ratio, ok = check_kernel_discrimination(dist, sigma=0.1)
        assert ratio > 3.0
        assert ok
    
    def test_collapsed(self):
        """Low ratio when σ is too large."""
        dist = np.random.rand(100, 100) * 0.5 + 0.5  # distances in [0.5, 1.0]
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        
        ratio, ok = check_kernel_discrimination(dist, sigma=5.0)
        assert ratio < 2.0
        assert not ok
```

**`tests/unit/test_propagation.py`:**
```python
import numpy as np
import pytest
from task_space.shocks.propagation import propagate_shock


class TestPropagate:
    def test_shapes(self):
        """Output shapes are correct."""
        n_act, n_occ = 100, 50
        I_t = np.random.rand(n_act)
        K = np.random.rand(n_act, n_act)
        rho = np.random.rand(n_occ, n_act)
        rho = rho / rho.sum(axis=1, keepdims=True)
        
        result = propagate_shock(I_t, K, rho, sigma=0.2)
        
        assert result.I_t.shape == (n_act,)
        assert result.A_t.shape == (n_act,)
        assert result.E.shape == (n_occ,)
    
    def test_uniform_shock(self):
        """Uniform shock gives exposure proportional to measure mass."""
        n_act, n_occ = 10, 3
        I_t = np.ones(n_act)  # Uniform shock
        K = np.eye(n_act)     # Identity kernel (no propagation)
        rho = np.array([
            [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ])
        
        result = propagate_shock(I_t, K, rho, sigma=1.0)
        
        # With identity kernel and uniform shock, E_j = sum_a ρ_j(a) = 1 for all j
        np.testing.assert_array_almost_equal(result.E, [1.0, 1.0, 1.0])
    
    def test_localized_shock(self):
        """Localized shock affects nearby occupations more."""
        n_act = 5
        I_t = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Shock only on first activity
        K = np.eye(n_act)  # No propagation
        
        rho = np.array([
            [1.0, 0, 0, 0, 0],    # Occ 0: only uses activity 0
            [0, 1.0, 0, 0, 0],    # Occ 1: only uses activity 1
        ])
        
        result = propagate_shock(I_t, K, rho, sigma=1.0)
        
        assert result.E[0] == pytest.approx(1.0)  # Fully exposed
        assert result.E[1] == pytest.approx(0.0)  # Not exposed
```

---

### Task 5: Directory Structure

**Create directories:**
```bash
mkdir -p notebooks/prototyping
mkdir -p notebooks/analysis
mkdir -p experiments/configs
mkdir -p experiments/scripts
```

**`notebooks/prototyping/README.md`:**
```markdown
# Prototyping Notebooks

Exploratory work for shock profile design and manifold visualization.

## Setup

```bash
pip install -e ".[notebooks]"
```

## Quick Start

```python
from task_space import build_dwa_occupation_measures
from task_space.data.artifacts import get_embeddings, get_distance_matrix
from task_space.similarity.kernel import build_kernel_matrix
from task_space.shocks.registry import list_shocks, get_shock
from task_space.shocks.propagation import propagate_shock

# Load
measures = build_dwa_occupation_measures()
embeddings = get_embeddings(measures.activity_titles)
dist_matrix = get_distance_matrix(embeddings)
K, sigma = build_kernel_matrix(dist_matrix)

# Define shock
I_t = np.zeros(len(measures.activity_ids))
I_t[100] = 1.0  # Shock on activity 100

# Propagate
result = propagate_shock(I_t, K, measures.occupation_matrix, sigma)

# Visualize
import matplotlib.pyplot as plt
plt.hist(result.E, bins=50)
plt.xlabel('Occupation Exposure')
plt.show()
```

## Guidelines

1. Import from package, not sys.path
2. Use artifact store for embeddings
3. Move working shocks to `src/task_space/shocks/profiles.py`
```

**`experiments/scripts/run_experiment.py`:**
```python
#!/usr/bin/env python
"""Run experiment from YAML config."""

import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    from task_space.experiments import ExperimentConfig, run_experiment
    
    config = ExperimentConfig.from_yaml(args.config)
    
    if args.dry_run:
        print(f"Config: {config.name}")
        print(f"  Similarity: {config.similarity}")
        print(f"  Shock: {config.shock or 'None'}")
        return
    
    run_experiment(config)

if __name__ == '__main__':
    main()
```

---

### Task 6: Update Documentation

#### 6.1 Update `CLAUDE.md`

**Replace status section with:**
```markdown
## Current Status: v0.6.3 (Infrastructure Complete)

**Phase I validation complete. Infrastructure ready for Phase II.**

| Measure | t-stat (clustered) | R² | Status |
|---------|-------------------|-----|--------|
| Normalized kernel | 7.14 | 0.00485 | Primary |
| Unnormalized kernel | 5.90 | 0.00310 | Robustness |
| Binary Jaccard | 8.00 | 0.00167 | Baseline |

**What Phase I established:**
- Continuous semantic structure predicts wage comovement
- 191% R² improvement over binary Jaccard
- Robust to concentration controls (t = 5.29)
- Not artifact (100th percentile vs random)

**What Phase II will test:**
- Shock profile construction (v1, v2, v3)
- Retrospective evaluation (1990–2007)
- Prospective evaluation (2022–present)
- RBTC vs AI stability
```

**Add architecture section:**
```markdown
## Architecture (v0.6.3)

### Package Structure

```
src/task_space/
    domain.py              # Activity domain, occupation measures
    data/                  # Data loading
        onet.py, oes.py, crosswalk.py, artifacts.py
    similarity/            # Similarity computation
        kernel.py, overlap.py, embeddings.py, distances.py
    shocks/                # Shock profiles (Phase II)
        registry.py, profiles.py, propagation.py
    validation/            # Validation utilities
        regression.py, diagnostics.py, permutation.py
    experiments/           # Experiment infrastructure
        config.py, runner.py
```

### Key Concepts

**Overlap vs Exposure:**
- Overlap: O = ρ K ρ^T — pairwise occupation similarity (Phase I)
- Exposure: E = ρ (K I_t) — occupation exposure to shock (Phase II)

**Registry Pattern:**
- Shocks: `@register_shock("name")` decorator
- Add new shocks without modifying runner

**Artifact Store:**
- Cache: `.cache/artifacts/v1/`
- Use `get_embeddings()`, `get_distance_matrix()`
- Never compute embeddings elsewhere
```

**Add lessons learned:**
```markdown
### From v0.6.3 (Consolidation)

16. **Single source of truth** — Duplication caused v0.5.0 bug to persist. One implementation, tested.

17. **Explicit rejection of bad methods** — `calibrate_sigma()` throws error for non-NN methods, not silent fallback.

18. **Config-driven experiments** — New experiments = YAML config + registered function, not new script.

19. **Overlap ≠ Exposure** — Phase I validated overlap (O = ρKρ^T). Phase II computes exposure (E = ρKI).
```

#### 6.2 Update `README.md`

**Update structure section:**
```markdown
## Repository Structure (v0.6.3)

```
src/task_space/
    domain.py                 # Core: Activity domain, occupation measures
    data/                     # Data loading and caching
    similarity/               # Kernel, overlap, embeddings
    shocks/                   # Shock profiles and propagation
    validation/               # Regression, diagnostics, tests
    experiments/              # Config-driven experiment runner

tests/
    unit/                     # pytest unit tests
    integration/              # Pipeline tests

experiments/
    configs/                  # YAML experiment configurations
    scripts/                  # CLI runners

notebooks/
    prototyping/              # Exploratory notebooks
    analysis/                 # Result analysis
```

## Quick Start (v0.6.3)

```bash
# Install
pip install -e ".[dev,notebooks]"

# Run tests
pytest tests/unit/ -v

# Run experiment
python experiments/scripts/run_experiment.py experiments/configs/phase1_validation.yaml
```
```

---

## Deliverables Checklist

| # | Task | Deliverable | Verification |
|---|------|-------------|--------------|
| 1 | Delete dead code | 5 files removed | Old scripts still run |
| 2.1 | `data/` subpackage | 4 modules | `from task_space.data import get_embeddings` |
| 2.2 | `similarity/` subpackage | 4 modules | `from task_space.similarity import build_kernel_matrix` |
| 2.3 | `shocks/` subpackage | 3 modules | `list_shocks()` returns 5 shocks |
| 2.4 | `validation/` subpackage | 3 modules | `run_validation_regression()` works |
| 2.5 | `experiments/` subpackage | 2 modules | Config loads from YAML |
| 3 | Package install | pyproject.toml | `pip install -e .` succeeds |
| 4 | Unit tests | 4 test files | `pytest tests/unit/ -v` passes |
| 5 | Directories | notebooks/, experiments/ | Directories exist |
| 6.1 | CLAUDE.md | Updated | Reflects v0.6.3 |
| 6.2 | README.md | Updated | Structure matches reality |

---

## Verification Protocol

After all tasks complete:

```bash
# 1. Install
pip install -e ".[dev,notebooks]"

# 2. Unit tests
pytest tests/unit/ -v
# Expected: All pass

# 3. Import check
python -c "
from task_space import build_dwa_occupation_measures
from task_space.data.artifacts import get_embeddings
from task_space.similarity.kernel import build_kernel_matrix
from task_space.shocks.registry import list_shocks
from task_space.shocks.propagation import propagate_shock
print('All imports OK')
print('Registered shocks:', list_shocks())
"
# Expected: ['uniform', 'gaussian_directed', 'capability_v1', 'capability_v2', 'rbtc']

# 4. Reproduce Phase 1 results
python experiments/scripts/run_experiment.py experiments/configs/phase1_normalized.yaml
# Expected: t ≈ 7.14, R² ≈ 0.00485

# 5. Cache check
ls .cache/artifacts/v1/embeddings/
# Expected: One .npz file
```

---

## v0.6.3.1: Architecture Tests

**Objective:** Validate new infrastructure with targeted experiments before full Phase II.

### Test 1: Registry Extensibility

**Goal:** Verify new shocks can be added without modifying runner.

**Procedure:**
1. Create `notebooks/prototyping/custom_shock.ipynb`
2. Define new shock function with `@register_shock`
3. Run via config without changing `runner.py`

**Config (`experiments/configs/test_custom_shock.yaml`):**
```yaml
name: test_custom_shock
similarity: normalized_kernel
shock: gaussian_directed
shock_args:
  center_idx: 500
  sigma_shock: 0.15
```

**Success criterion:** Experiment runs, produces exposure vector.

---

### Test 2: Artifact Cache Consistency

**Goal:** Verify embeddings are loaded from cache, not recomputed.

**Procedure:**
1. Run experiment (computes embeddings)
2. Delete model from memory, run again
3. Verify cache hit (no progress bar)

**Success criterion:** Second run is <5 seconds (vs ~2 minutes for computation).

---

### Test 3: Propagation Correctness

**Goal:** Verify E = ρ(KI) produces expected results.

**Procedure:**
1. Create synthetic shock: I_t = 1 on activity 0, 0 elsewhere
2. Create synthetic K: identity matrix
3. Create synthetic ρ: occupation 0 has all mass on activity 0
4. Verify E[0] = 1.0, E[others] = 0.0

**Success criterion:** Unit test passes.

---

### Test 4: Regression Consistency

**Goal:** Verify new regression module matches v0.6.2 results.

**Procedure:**
1. Run `phase1_normalized.yaml` config
2. Compare to stored v0.6.2 results

**Success criterion:** t-stat within 0.01, R² within 0.00001.

---

### Test 5: Control Variables

**Goal:** Verify entropy control produces t ≈ 5.29.

**Procedure:**
1. Add entropy control to config
2. Run experiment
3. Check coefficient on overlap

**Config:**
```yaml
name: test_entropy_control
similarity: normalized_kernel
controls:
  - entropy
```

**Success criterion:** t_overlap ≈ 5.29 (±0.1).

---

### Test 6: End-to-End Shock Exposure

**Goal:** Compute and inspect AI exposure (v1 profile).

**Procedure:**
1. Construct activity classifications (manual/cognitive)
2. Run capability_v1 shock
3. Inspect top-10 exposed occupations

**Success criterion:** 
- Cognitive occupations (e.g., "Accountant") have positive exposure
- Manual occupations (e.g., "Carpenter") have negative/zero exposure
- Results are face-valid

---

### Test 7: Notebook Workflow

**Goal:** Verify Jupyter can import and visualize.

**Procedure:**
1. Start `jupyter notebook`
2. Open `notebooks/prototyping/shock_visualization.ipynb`
3. Run all cells

**Success criterion:** Notebook executes without import errors, produces plots.

---

## Post-v0.6.3.1: Phase II Roadmap

After architecture tests pass:

| Version | Objective |
|---------|-----------|
| v0.6.4 | Construct activity classifications (manual/cognitive/routine) |
| v0.6.5 | Implement v1, v2, v3 shock profiles with real data |
| v0.6.6 | Retrospective evaluation (1990–2007, RBTC shock) |
| v0.6.7 | Prospective evaluation (2022–present, AI shock) |
| v0.7.0 | Stability test: RBTC vs AI exposure correlation |

---

**Status:** Ready for implementation.