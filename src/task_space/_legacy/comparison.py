"""
Phase 2: Systematic Representation Comparison (v0.6.2)

Implements the full comparison protocol from spec_0.6.2.md.
Compares discrete, continuous, and hybrid representations of task space.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RepresentationResult:
    """
    Result for a single representation's validation.
    """
    rep_id: str
    name: str
    category: str  # 'discrete', 'continuous_text', 'continuous_onet', 'hybrid'
    beta: float
    se: float
    t_stat: float
    pvalue: float
    r_squared: float
    n_pairs: int
    n_clusters: int
    sigma: Optional[float] = None  # For continuous representations


@dataclass
class PermutationResult:
    """
    Permutation test result.
    """
    rep_id: str
    observed_t: float
    perm_t_mean: float
    perm_t_std: float
    perm_t_max: float
    p_value: float
    n_permutations: int


@dataclass
class CrossValidationResult:
    """
    Cross-validation result.
    """
    rep_id: str
    cv_r2_mean: float
    cv_r2_std: float
    cv_r2_folds: list[float]
    full_sample_r2: float
    overfit_ratio: float


@dataclass
class BandwidthSensitivityResult:
    """
    Bandwidth sensitivity analysis for continuous representations.
    """
    rep_id: str
    sigma_values: dict[str, float]  # percentile label -> sigma value
    t_stats: dict[str, float]  # percentile label -> t-stat
    r2_values: dict[str, float]  # percentile label -> R²
    t_range: float  # max_t - min_t
    t_cv: float  # coefficient of variation of t-stats
    is_fragile: bool  # True if t varies by >50%


@dataclass
class HybridModelResult:
    """
    Hybrid model (multiple representations) result.
    """
    model_id: str
    components: list[str]
    r_squared: float
    n_pairs: int
    component_betas: dict[str, float]
    component_ses: dict[str, float]
    component_ts: dict[str, float]
    component_ps: dict[str, float]


# =============================================================================
# Representation Names
# =============================================================================

REPRESENTATION_NAMES = {
    'D1': 'Binary Jaccard',
    'D2': 'Weighted Jaccard',
    'D3': 'Cosine (binary)',
    'C1': 'MPNet',
    'C2': 'JobBERT',
    'C3': 'E5-large',
    'C4': 'Abilities',
    'C5': 'Skills',
    'C6': 'Knowledge',
    'C7': 'Combined (A+S+K)',
}


# =============================================================================
# Discrete Representations
# =============================================================================

def compute_binary_jaccard(occ_measures: np.ndarray) -> np.ndarray:
    """
    D1: Binary Jaccard similarity.

    |A ∩ B| / |A ∪ B| where A, B are sets of activities with positive weight.

    Args:
        occ_measures: (n_occ, n_activities) raw importance matrix

    Returns:
        (n_occ, n_occ) Jaccard similarity matrix
    """
    # Binarize
    B = (occ_measures > 0).astype(np.float64)

    # Intersection: B[i] · B[j]
    intersection = B @ B.T

    # Union: |A_i| + |A_j| - |A_i ∩ A_j|
    activity_counts = B.sum(axis=1, keepdims=True)
    union = activity_counts + activity_counts.T - intersection

    # Avoid division by zero
    union = np.maximum(union, 1e-10)

    return intersection / union


def compute_weighted_jaccard(occ_measures: np.ndarray) -> np.ndarray:
    """
    D2: Weighted Jaccard similarity.

    Σ min(ρ_i(a), ρ_j(a)) / Σ max(ρ_i(a), ρ_j(a))

    Args:
        occ_measures: (n_occ, n_activities) importance matrix

    Returns:
        (n_occ, n_occ) weighted Jaccard similarity matrix
    """
    n_occ = occ_measures.shape[0]
    jaccard = np.zeros((n_occ, n_occ))

    for i in range(n_occ):
        for j in range(i, n_occ):
            min_sum = np.sum(np.minimum(occ_measures[i], occ_measures[j]))
            max_sum = np.sum(np.maximum(occ_measures[i], occ_measures[j]))

            if max_sum > 0:
                jaccard[i, j] = min_sum / max_sum
                jaccard[j, i] = jaccard[i, j]

    return jaccard


def compute_cosine_binary(occ_measures: np.ndarray) -> np.ndarray:
    """
    D3: Cosine similarity on binary vectors.

    Args:
        occ_measures: (n_occ, n_activities) importance matrix

    Returns:
        (n_occ, n_occ) cosine similarity matrix
    """
    B = (occ_measures > 0).astype(np.float64)
    return cosine_similarity(B)


# =============================================================================
# Text Embedding Representations
# =============================================================================

def compute_mpnet_embeddings(
    texts: list[str],
    device: str = "cuda",
    show_progress: bool = True,
) -> np.ndarray:
    """
    C1: MPNet embeddings (all-mpnet-base-v2).

    Args:
        texts: List of activity text descriptions
        device: Device to use ('cuda', 'cpu')
        show_progress: Show progress bar

    Returns:
        (n_texts, 768) embedding matrix
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-mpnet-base-v2', device=device)
    embeddings = model.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)
    return embeddings


def compute_jobbert_embeddings(
    texts: list[str],
    device: str = "cuda",
    batch_size: int = 32,
) -> np.ndarray:
    """
    C2: JobBERT embeddings (TechWolf/JobBERT-v2).

    Uses mean pooling over tokens.

    Args:
        texts: List of activity text descriptions
        device: Device to use
        batch_size: Batch size for inference

    Returns:
        (n_texts, 768) embedding matrix
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("TechWolf/JobBERT-v2")
    model = AutoModel.from_pretrained("TechWolf/JobBERT-v2").to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)

            outputs = model(**inputs)

            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            embeddings.append(mean_pooled.cpu().numpy())

    return np.vstack(embeddings)


def compute_e5_embeddings(
    texts: list[str],
    device: str = "cuda",
    show_progress: bool = True,
) -> np.ndarray:
    """
    C3: E5-large embeddings (intfloat/e5-large-v2).

    Note: E5 requires "query: " prefix for queries.

    Args:
        texts: List of activity text descriptions
        device: Device to use
        show_progress: Show progress bar

    Returns:
        (n_texts, 1024) embedding matrix
    """
    from sentence_transformers import SentenceTransformer

    # E5 requires prefix
    prefixed_texts = [f"query: {t}" for t in texts]

    model = SentenceTransformer('intfloat/e5-large-v2', device=device)
    embeddings = model.encode(prefixed_texts, show_progress_bar=show_progress, convert_to_numpy=True)
    return embeddings


def embeddings_to_kernel_overlap(
    occ_measures: np.ndarray,
    activity_embeddings: np.ndarray,
    sigma: Optional[float] = None,
    kernel_type: str = 'exponential',
    normalize: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Convert activity embeddings to kernel-weighted occupation overlap.

    Uses the Phase 1 fix:
    - σ = median of nearest-neighbor distances (if not specified)
    - No row normalization (per Phase 1 findings)

    Args:
        occ_measures: (n_occ, n_activities) occupation measure matrix
        activity_embeddings: (n_activities, embedding_dim) embeddings
        sigma: Kernel bandwidth. If None, use NN median.
        kernel_type: 'exponential' or 'gaussian'
        normalize: Whether to row-normalize kernel (default False)

    Returns:
        Tuple of (overlap_matrix, sigma_used)
    """
    # Compute activity distance matrix (cosine distance)
    dist_matrix = cosine_distances(activity_embeddings)

    # Compute sigma from NN distances if not provided
    if sigma is None:
        dm = dist_matrix.copy()
        np.fill_diagonal(dm, np.inf)
        nn_dists = dm.min(axis=1)
        sigma = float(np.median(nn_dists))

    # Compute kernel
    if kernel_type == 'exponential':
        K = np.exp(-dist_matrix / sigma)
    elif kernel_type == 'gaussian':
        K = np.exp(-dist_matrix**2 / (2 * sigma**2))
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    if normalize:
        K = K / K.sum(axis=1, keepdims=True)

    # Compute overlap: rho @ K @ rho^T
    overlap = occ_measures @ K @ occ_measures.T

    # Symmetrize
    overlap = (overlap + overlap.T) / 2

    return overlap, sigma


# =============================================================================
# O*NET Structured Representations
# =============================================================================

def load_onet_structured_dimension(
    onet_path: str,
    dimension: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load O*NET structured dimension (Abilities, Skills, or Knowledge).

    Uses Importance × Level composite scoring.

    Args:
        onet_path: Path to O*NET database directory
        dimension: 'abilities', 'skills', or 'knowledge'

    Returns:
        Tuple of (matrix, occ_codes, element_ids)
        - matrix: (n_occ, n_elements) standardized composite scores
        - occ_codes: List of O*NET-SOC codes
        - element_ids: List of element IDs
    """
    file_map = {
        'abilities': 'Abilities.xlsx',
        'skills': 'Skills.xlsx',
        'knowledge': 'Knowledge.xlsx',
    }

    filepath = Path(onet_path) / file_map[dimension]
    df = pd.read_excel(filepath)

    # Get Importance ratings (Scale ID = 'IM')
    df_im = df[df['Scale ID'] == 'IM'][['O*NET-SOC Code', 'Element ID', 'Data Value']].copy()
    df_im = df_im.rename(columns={'Data Value': 'Importance'})

    # Get Level ratings (Scale ID = 'LV')
    df_lv = df[df['Scale ID'] == 'LV'][['O*NET-SOC Code', 'Element ID', 'Data Value']].copy()
    df_lv = df_lv.rename(columns={'Data Value': 'Level'})

    # Merge and compute composite
    df_merged = df_im.merge(df_lv, on=['O*NET-SOC Code', 'Element ID'], how='inner')
    df_merged['Composite'] = df_merged['Importance'] * df_merged['Level']

    # Pivot to matrix form
    matrix = df_merged.pivot_table(
        index='O*NET-SOC Code',
        columns='Element ID',
        values='Composite',
        aggfunc='first',
    ).fillna(0)

    # Standardize columns (z-score)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix_std = (matrix - matrix.mean()) / matrix.std()
        matrix_std = matrix_std.fillna(0)

    return matrix_std.values, list(matrix_std.index), list(matrix_std.columns)


def compute_structured_similarity(
    occ_vectors: np.ndarray,
    metric: str = 'euclidean',
) -> np.ndarray:
    """
    Compute occupation similarity from structured O*NET dimensions.

    Converts distance to similarity: sim = 1 / (1 + dist)

    Args:
        occ_vectors: (n_occ, n_dim) standardized dimension vectors
        metric: Distance metric ('euclidean', 'cosine')

    Returns:
        (n_occ, n_occ) similarity matrix
    """
    from sklearn.metrics.pairwise import pairwise_distances

    dist_matrix = pairwise_distances(occ_vectors, metric=metric)

    # Convert to similarity (bounded [0, 1])
    sim_matrix = 1 / (1 + dist_matrix)

    return sim_matrix


# =============================================================================
# Validation Functions
# =============================================================================

def _cluster_se(
    X: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray,
) -> tuple[float, float, float, int]:
    """
    OLS with clustered standard errors.

    Returns (beta, se, r_squared, n_clusters)
    """
    n = len(y)

    # Add constant
    X_mat = np.column_stack([np.ones(n), X])

    # OLS
    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    beta_vec = XtX_inv @ (X_mat.T @ y)
    resid = y - X_mat @ beta_vec

    # R-squared
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Clustered SEs
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((2, 2))
    for c in unique_clusters:
        mask = cluster_ids == c
        cluster_resid = resid[mask]
        cluster_X = X_mat[mask]
        score = cluster_X.T @ cluster_resid
        meat += np.outer(score, score)

    # Small-sample correction
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - 2))
    var_beta = correction * XtX_inv @ meat @ XtX_inv
    se_beta = np.sqrt(np.maximum(np.diag(var_beta), 1e-20))

    return float(beta_vec[1]), float(se_beta[1]), float(r_squared), int(n_clusters)


def run_validation_regression(
    similarity_matrix: np.ndarray,
    wage_comovement: np.ndarray,
    sim_codes: list[str],
    comovement_codes: list[str],
    crosswalk_map: dict[str, str],
    rep_id: str,
    name: str,
    category: str,
    sigma: Optional[float] = None,
) -> RepresentationResult:
    """
    Run validation regression: WageComovement ~ Similarity.

    Handles O*NET → SOC aggregation.

    Args:
        similarity_matrix: Occupation similarity matrix
        wage_comovement: Wage comovement matrix (SOC level)
        sim_codes: Occupation codes for similarity matrix (may be O*NET or SOC)
        comovement_codes: SOC codes for comovement matrix
        crosswalk_map: Dict mapping O*NET-SOC to SOC (empty if already SOC level)
        rep_id: Representation ID (e.g., 'D1')
        name: Human-readable name
        category: Category ('discrete', 'continuous_text', etc.)
        sigma: Sigma used (for continuous representations)

    Returns:
        RepresentationResult
    """
    # Check if we need to aggregate (O*NET → SOC)
    needs_aggregation = len(crosswalk_map) > 0

    if needs_aggregation:
        # Aggregate similarity to SOC level
        onet_to_soc = crosswalk_map
        soc_codes = list(set(onet_to_soc.values()))
        soc_codes = [soc for soc in soc_codes if soc in comovement_codes]
        soc_codes = sorted(soc_codes)

        n_soc = len(soc_codes)
        soc_to_idx = {soc: i for i, soc in enumerate(soc_codes)}
        comovement_idx = {soc: i for i, soc in enumerate(comovement_codes)}
        sim_to_idx = {code: i for i, code in enumerate(sim_codes)}

        # Aggregate
        soc_sim = np.zeros((n_soc, n_soc))
        soc_counts = np.zeros((n_soc, n_soc))

        for onet_i, soc_i in onet_to_soc.items():
            if soc_i not in soc_to_idx or onet_i not in sim_to_idx:
                continue
            for onet_j, soc_j in onet_to_soc.items():
                if soc_j not in soc_to_idx or onet_j not in sim_to_idx:
                    continue
                if onet_i >= onet_j:
                    continue

                i_idx = sim_to_idx[onet_i]
                j_idx = sim_to_idx[onet_j]
                sim_val = similarity_matrix[i_idx, j_idx]

                si = soc_to_idx[soc_i]
                sj = soc_to_idx[soc_j]
                if si > sj:
                    si, sj = sj, si

                soc_sim[si, sj] += sim_val
                soc_counts[si, sj] += 1

        soc_counts[soc_counts == 0] = 1
        soc_sim = soc_sim / soc_counts
    else:
        # Already at SOC level
        soc_codes = [soc for soc in sim_codes if soc in comovement_codes]
        soc_codes = sorted(soc_codes)

        n_soc = len(soc_codes)
        soc_to_idx = {soc: i for i, soc in enumerate(soc_codes)}
        comovement_idx = {soc: i for i, soc in enumerate(comovement_codes)}
        sim_to_idx = {code: i for i, code in enumerate(sim_codes)}

        soc_sim = np.zeros((n_soc, n_soc))
        for i, soc_i in enumerate(soc_codes):
            for j, soc_j in enumerate(soc_codes):
                if j <= i:
                    continue
                si = sim_to_idx[soc_i]
                sj = sim_to_idx[soc_j]
                soc_sim[i, j] = similarity_matrix[si, sj]

    # Build pair dataset
    pairs_x, pairs_y, pairs_cluster = [], [], []
    for i in range(n_soc):
        for j in range(i + 1, n_soc):
            soc_i = soc_codes[i]
            soc_j = soc_codes[j]

            ci = comovement_idx.get(soc_i)
            cj = comovement_idx.get(soc_j)

            if ci is None or cj is None:
                continue

            comove_val = wage_comovement[ci, cj]

            if np.isnan(comove_val):
                continue

            pairs_x.append(soc_sim[i, j])
            pairs_y.append(comove_val)
            pairs_cluster.append(soc_i)

    if len(pairs_x) == 0:
        raise ValueError(f"No valid pairs found for {rep_id}")

    X = np.array(pairs_x)
    y = np.array(pairs_y)
    clusters = np.array(pairs_cluster)

    _, cluster_ids = np.unique(clusters, return_inverse=True)

    beta, se, r_squared, n_clusters = _cluster_se(X, y, cluster_ids)
    t_stat = beta / se if se > 0 else 0.0
    pvalue = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=max(n_clusters - 1, 1)))

    return RepresentationResult(
        rep_id=rep_id,
        name=name,
        category=category,
        beta=beta,
        se=se,
        t_stat=t_stat,
        pvalue=pvalue,
        r_squared=r_squared,
        n_pairs=len(pairs_x),
        n_clusters=n_clusters,
        sigma=sigma,
    )


def _build_pair_dataset(
    similarity_matrix: np.ndarray,
    wage_comovement: np.ndarray,
    sim_codes: list[str],
    comovement_codes: list[str],
    crosswalk_map: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build pair-level dataset for permutation tests.

    Returns (x, y, clusters) arrays.
    """
    needs_aggregation = len(crosswalk_map) > 0

    if needs_aggregation:
        onet_to_soc = crosswalk_map
        soc_codes = list(set(onet_to_soc.values()))
        soc_codes = [soc for soc in soc_codes if soc in comovement_codes]
        soc_codes = sorted(soc_codes)

        n_soc = len(soc_codes)
        soc_to_idx = {soc: i for i, soc in enumerate(soc_codes)}
        comovement_idx = {soc: i for i, soc in enumerate(comovement_codes)}
        sim_to_idx = {code: i for i, code in enumerate(sim_codes)}

        soc_sim = np.zeros((n_soc, n_soc))
        soc_counts = np.zeros((n_soc, n_soc))

        for onet_i, soc_i in onet_to_soc.items():
            if soc_i not in soc_to_idx or onet_i not in sim_to_idx:
                continue
            for onet_j, soc_j in onet_to_soc.items():
                if soc_j not in soc_to_idx or onet_j not in sim_to_idx:
                    continue
                if onet_i >= onet_j:
                    continue

                i_idx = sim_to_idx[onet_i]
                j_idx = sim_to_idx[onet_j]
                sim_val = similarity_matrix[i_idx, j_idx]

                si = soc_to_idx[soc_i]
                sj = soc_to_idx[soc_j]
                if si > sj:
                    si, sj = sj, si

                soc_sim[si, sj] += sim_val
                soc_counts[si, sj] += 1

        soc_counts[soc_counts == 0] = 1
        soc_sim = soc_sim / soc_counts
    else:
        soc_codes = [soc for soc in sim_codes if soc in comovement_codes]
        soc_codes = sorted(soc_codes)
        n_soc = len(soc_codes)
        soc_to_idx = {soc: i for i, soc in enumerate(soc_codes)}
        comovement_idx = {soc: i for i, soc in enumerate(comovement_codes)}
        sim_to_idx = {code: i for i, code in enumerate(sim_codes)}

        soc_sim = np.zeros((n_soc, n_soc))
        for i, soc_i in enumerate(soc_codes):
            for j, soc_j in enumerate(soc_codes):
                if j <= i:
                    continue
                if soc_i in sim_to_idx and soc_j in sim_to_idx:
                    si = sim_to_idx[soc_i]
                    sj = sim_to_idx[soc_j]
                    soc_sim[i, j] = similarity_matrix[si, sj]

    # Build pairs
    pairs_x, pairs_y, pairs_cluster = [], [], []
    for i in range(n_soc):
        for j in range(i + 1, n_soc):
            ci = comovement_idx.get(soc_codes[i])
            cj = comovement_idx.get(soc_codes[j])
            if ci is None or cj is None:
                continue
            comove_val = wage_comovement[ci, cj]
            if np.isnan(comove_val):
                continue
            pairs_x.append(soc_sim[i, j])
            pairs_y.append(comove_val)
            pairs_cluster.append(soc_codes[i])

    return np.array(pairs_x), np.array(pairs_y), np.array(pairs_cluster)


def run_permutation_test(
    similarity_matrix: np.ndarray,
    wage_comovement: np.ndarray,
    sim_codes: list[str],
    comovement_codes: list[str],
    crosswalk_map: dict[str, str],
    rep_id: str,
    n_permutations: int = 1000,
    seed: int = 42,
) -> PermutationResult:
    """
    Permutation test for statistical significance.

    Shuffles y values (comovement) at pair level to break the
    relationship between similarity and comovement.

    Uses simple correlation-based t-stat for speed (clustered SEs
    are for inference, permutation test just needs null distribution).
    """
    np.random.seed(seed)

    # Build pair-level dataset once
    x, y, clusters = _build_pair_dataset(
        similarity_matrix, wage_comovement,
        sim_codes, comovement_codes, crosswalk_map
    )

    if len(x) == 0:
        return PermutationResult(
            rep_id=rep_id,
            observed_t=0.0,
            perm_t_mean=0.0,
            perm_t_std=0.0,
            perm_t_max=0.0,
            p_value=1.0,
            n_permutations=0,
        )

    n = len(x)

    def simple_t(x_arr, y_arr):
        """Fast t-stat from correlation."""
        r = np.corrcoef(x_arr, y_arr)[0, 1]
        if abs(r) >= 1.0:
            return 0.0
        t = r * np.sqrt((n - 2) / (1 - r**2))
        return t

    # Compute observed t-stat (use simple for consistency with permutation)
    observed_t = simple_t(x, y)

    # Permutation distribution: shuffle y values
    perm_ts = np.zeros(n_permutations)
    for i in range(n_permutations):
        y_perm = np.random.permutation(y)
        perm_ts[i] = simple_t(x, y_perm)

    p_value = np.mean(np.abs(perm_ts) >= np.abs(observed_t))

    return PermutationResult(
        rep_id=rep_id,
        observed_t=float(observed_t),
        perm_t_mean=float(np.mean(perm_ts)),
        perm_t_std=float(np.std(perm_ts)),
        perm_t_max=float(np.max(np.abs(perm_ts))),
        p_value=float(p_value),
        n_permutations=n_permutations,
    )


def run_cross_validation(
    similarity_matrix: np.ndarray,
    wage_comovement: np.ndarray,
    sim_codes: list[str],
    comovement_codes: list[str],
    crosswalk_map: dict[str, str],
    rep_id: str,
    n_folds: int = 5,
    seed: int = 42,
) -> CrossValidationResult:
    """
    K-fold cross-validation for R².
    """
    from sklearn.model_selection import KFold

    # First build the full pair dataset
    # (Need to do aggregation once)
    needs_aggregation = len(crosswalk_map) > 0

    if needs_aggregation:
        onet_to_soc = crosswalk_map
        soc_codes = list(set(onet_to_soc.values()))
        soc_codes = [soc for soc in soc_codes if soc in comovement_codes]
        soc_codes = sorted(soc_codes)

        n_soc = len(soc_codes)
        soc_to_idx = {soc: i for i, soc in enumerate(soc_codes)}
        comovement_idx = {soc: i for i, soc in enumerate(comovement_codes)}
        sim_to_idx = {code: i for i, code in enumerate(sim_codes)}

        soc_sim = np.zeros((n_soc, n_soc))
        soc_counts = np.zeros((n_soc, n_soc))

        for onet_i, soc_i in onet_to_soc.items():
            if soc_i not in soc_to_idx or onet_i not in sim_to_idx:
                continue
            for onet_j, soc_j in onet_to_soc.items():
                if soc_j not in soc_to_idx or onet_j not in sim_to_idx:
                    continue
                if onet_i >= onet_j:
                    continue

                i_idx = sim_to_idx[onet_i]
                j_idx = sim_to_idx[onet_j]
                sim_val = similarity_matrix[i_idx, j_idx]

                si = soc_to_idx[soc_i]
                sj = soc_to_idx[soc_j]
                if si > sj:
                    si, sj = sj, si

                soc_sim[si, sj] += sim_val
                soc_counts[si, sj] += 1

        soc_counts[soc_counts == 0] = 1
        soc_sim = soc_sim / soc_counts
    else:
        soc_codes = [soc for soc in sim_codes if soc in comovement_codes]
        soc_codes = sorted(soc_codes)
        n_soc = len(soc_codes)
        soc_to_idx = {soc: i for i, soc in enumerate(soc_codes)}
        comovement_idx = {soc: i for i, soc in enumerate(comovement_codes)}
        sim_to_idx = {code: i for i, code in enumerate(sim_codes)}

        soc_sim = np.zeros((n_soc, n_soc))
        for i, soc_i in enumerate(soc_codes):
            for j, soc_j in enumerate(soc_codes):
                if j <= i:
                    continue
                si = sim_to_idx[soc_i]
                sj = sim_to_idx[soc_j]
                soc_sim[i, j] = similarity_matrix[si, sj]

    # Build pairs
    pairs_x, pairs_y = [], []
    for i in range(n_soc):
        for j in range(i + 1, n_soc):
            ci = comovement_idx.get(soc_codes[i])
            cj = comovement_idx.get(soc_codes[j])
            if ci is None or cj is None:
                continue
            comove_val = wage_comovement[ci, cj]
            if np.isnan(comove_val):
                continue
            pairs_x.append(soc_sim[i, j])
            pairs_y.append(comove_val)

    x = np.array(pairs_x)
    y = np.array(pairs_y)

    # Full sample R²
    X_full = np.column_stack([np.ones(len(x)), x])
    beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
    y_pred_full = X_full @ beta_full
    ss_res_full = np.sum((y - y_pred_full)**2)
    ss_tot_full = np.sum((y - y.mean())**2)
    full_r2 = 1 - ss_res_full / ss_tot_full if ss_tot_full > 0 else 0.0

    # Cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_r2s = []

    for train_idx, test_idx in kf.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = np.column_stack([np.ones(len(x_train)), x_train])
        beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

        X_test = np.column_stack([np.ones(len(x_test)), x_test])
        y_pred = X_test @ beta

        ss_res = np.sum((y_test - y_pred)**2)
        ss_tot = np.sum((y_test - y_test.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        cv_r2s.append(r2)

    cv_mean = float(np.mean(cv_r2s))
    overfit_ratio = full_r2 / cv_mean if cv_mean > 0 else float('inf')

    return CrossValidationResult(
        rep_id=rep_id,
        cv_r2_mean=cv_mean,
        cv_r2_std=float(np.std(cv_r2s)),
        cv_r2_folds=cv_r2s,
        full_sample_r2=full_r2,
        overfit_ratio=overfit_ratio,
    )


def run_hybrid_regression(
    similarity_matrices: dict[str, np.ndarray],
    wage_comovement: np.ndarray,
    codes_map: dict[str, list[str]],  # rep_id -> codes
    comovement_codes: list[str],
    crosswalk_maps: dict[str, dict[str, str]],  # rep_id -> crosswalk
    components: list[str],
    model_id: str,
) -> HybridModelResult:
    """
    Multiple regression with multiple similarity measures.

    Y_ij = α + β₁*Sim1 + β₂*Sim2 + ... + ε
    """
    # Build pair datasets for each component
    # Need to aggregate each separately, then join

    # Use first component to determine pair structure
    ref_component = components[0]
    ref_sim = similarity_matrices[ref_component]
    ref_codes = codes_map[ref_component]
    ref_crosswalk = crosswalk_maps[ref_component]

    needs_aggregation = len(ref_crosswalk) > 0

    if needs_aggregation:
        soc_codes = list(set(ref_crosswalk.values()))
        soc_codes = [soc for soc in soc_codes if soc in comovement_codes]
        soc_codes = sorted(soc_codes)
    else:
        soc_codes = [soc for soc in ref_codes if soc in comovement_codes]
        soc_codes = sorted(soc_codes)

    n_soc = len(soc_codes)
    soc_to_idx = {soc: i for i, soc in enumerate(soc_codes)}
    comovement_idx = {soc: i for i, soc in enumerate(comovement_codes)}

    # Aggregate each component
    component_soc_sims = {}
    for comp in components:
        sim = similarity_matrices[comp]
        codes = codes_map[comp]
        crosswalk = crosswalk_maps[comp]

        if len(crosswalk) > 0:
            sim_to_idx = {code: i for i, code in enumerate(codes)}
            soc_sim = np.zeros((n_soc, n_soc))
            soc_counts = np.zeros((n_soc, n_soc))

            for onet_i, soc_i in crosswalk.items():
                if soc_i not in soc_to_idx or onet_i not in sim_to_idx:
                    continue
                for onet_j, soc_j in crosswalk.items():
                    if soc_j not in soc_to_idx or onet_j not in sim_to_idx:
                        continue
                    if onet_i >= onet_j:
                        continue

                    i_idx = sim_to_idx[onet_i]
                    j_idx = sim_to_idx[onet_j]
                    sim_val = sim[i_idx, j_idx]

                    si = soc_to_idx[soc_i]
                    sj = soc_to_idx[soc_j]
                    if si > sj:
                        si, sj = sj, si

                    soc_sim[si, sj] += sim_val
                    soc_counts[si, sj] += 1

            soc_counts[soc_counts == 0] = 1
            soc_sim = soc_sim / soc_counts
        else:
            sim_to_idx = {code: i for i, code in enumerate(codes)}
            soc_sim = np.zeros((n_soc, n_soc))
            for i, soc_i in enumerate(soc_codes):
                for j, soc_j in enumerate(soc_codes):
                    if j <= i:
                        continue
                    if soc_i in sim_to_idx and soc_j in sim_to_idx:
                        si = sim_to_idx[soc_i]
                        sj = sim_to_idx[soc_j]
                        soc_sim[i, j] = sim[si, sj]

        component_soc_sims[comp] = soc_sim

    # Build pair dataset
    pairs_data = {comp: [] for comp in components}
    pairs_y = []
    pairs_cluster = []

    for i in range(n_soc):
        for j in range(i + 1, n_soc):
            ci = comovement_idx.get(soc_codes[i])
            cj = comovement_idx.get(soc_codes[j])
            if ci is None or cj is None:
                continue
            comove_val = comovement_codes[ci] if isinstance(comovement_codes, np.ndarray) else wage_comovement[ci, cj]
            comove_val = wage_comovement[ci, cj]
            if np.isnan(comove_val):
                continue

            for comp in components:
                pairs_data[comp].append(component_soc_sims[comp][i, j])
            pairs_y.append(comove_val)
            pairs_cluster.append(soc_codes[i])

    y = np.array(pairs_y)
    clusters = np.array(pairs_cluster)
    _, cluster_ids = np.unique(clusters, return_inverse=True)

    # Build design matrix
    X_cols = [np.ones(len(y))]
    for comp in components:
        X_cols.append(np.array(pairs_data[comp]))
    X = np.column_stack(X_cols)

    # OLS
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_vec = XtX_inv @ (X.T @ y)
    resid = y - X @ beta_vec

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Clustered SEs
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    n = len(y)
    k = len(components) + 1

    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = cluster_ids == c
        cluster_resid = resid[mask]
        cluster_X = X[mask]
        score = cluster_X.T @ cluster_resid
        meat += np.outer(score, score)

    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    var_beta = correction * XtX_inv @ meat @ XtX_inv
    se_beta = np.sqrt(np.maximum(np.diag(var_beta), 1e-20))

    # Extract component results
    betas = {}
    ses = {}
    ts = {}
    ps = {}

    for i, comp in enumerate(components):
        betas[comp] = float(beta_vec[i + 1])
        ses[comp] = float(se_beta[i + 1])
        ts[comp] = float(beta_vec[i + 1] / se_beta[i + 1]) if se_beta[i + 1] > 0 else 0.0
        ps[comp] = float(2 * (1 - sp_stats.t.cdf(abs(ts[comp]), df=max(n_clusters - 1, 1))))

    return HybridModelResult(
        model_id=model_id,
        components=components,
        r_squared=float(r_squared),
        n_pairs=len(y),
        component_betas=betas,
        component_ses=ses,
        component_ts=ts,
        component_ps=ps,
    )


# =============================================================================
# Output Generation
# =============================================================================

def generate_phase2_summary(
    primary_results: dict[str, RepresentationResult],
    perm_results: dict[str, PermutationResult],
    cv_results: dict[str, CrossValidationResult],
    hybrid_results: dict[str, HybridModelResult],
    output_path: Path,
) -> str:
    """
    Generate phase2_summary.md.
    """
    lines = [
        "# Phase 2 Summary: Representation Comparison",
        "",
        "## Primary Validation Results",
        "",
        "| ID | Name | t-stat | R² | Perm p | CV R² | Overfit | Tier |",
        "|----|------|--------|-----|--------|-------|---------|------|",
    ]

    def get_tier(t, perm_p, overfit):
        if t > 20 and perm_p < 0.001 and overfit < 1.2:
            return "Tier 1"
        elif t > 10 and perm_p < 0.01 and overfit < 1.5:
            return "Tier 2"
        elif t > 5 and perm_p < 0.05:
            return "Tier 3"
        else:
            return "Reject"

    for rep_id in sorted(primary_results.keys(), key=lambda k: -primary_results[k].t_stat):
        res = primary_results[rep_id]
        perm = perm_results.get(rep_id)
        cv = cv_results.get(rep_id)

        perm_p = perm.p_value if perm else 1.0
        cv_r2 = cv.cv_r2_mean if cv else 0.0
        overfit = cv.overfit_ratio if cv else float('inf')

        tier = get_tier(res.t_stat, perm_p, overfit)

        lines.append(
            f"| {rep_id} | {res.name} | {res.t_stat:.2f} | {res.r_squared:.5f} | "
            f"{perm_p:.4f} | {cv_r2:.5f} | {overfit:.2f} | {tier} |"
        )

    lines.extend([
        "",
        "## Hybrid Model Results",
        "",
        "| Model | Components | R² | Component t-stats |",
        "|-------|------------|-----|-------------------|",
    ])

    for h_id, h_res in hybrid_results.items():
        comps = '+'.join(h_res.components)
        t_stats = ', '.join([f"{c}={h_res.component_ts[c]:.2f}" for c in h_res.components])
        lines.append(f"| {h_id} | {comps} | {h_res.r_squared:.5f} | {t_stats} |")

    lines.extend([
        "",
        "## Decision Criteria",
        "",
        "- **Tier 1:** t > 20, perm p < 0.001, overfit < 1.2",
        "- **Tier 2:** t > 10, perm p < 0.01, overfit < 1.5",
        "- **Tier 3:** t > 5, perm p < 0.05",
        "- **Reject:** Does not meet Tier 3 criteria",
        "",
    ])

    # Recommendation
    best_id = max(primary_results.keys(), key=lambda k: primary_results[k].t_stat)
    best = primary_results[best_id]

    lines.extend([
        "## Recommendation",
        "",
        f"**Best representation:** {best_id} ({best.name})",
        f"- t-stat: {best.t_stat:.2f}",
        f"- R²: {best.r_squared:.5f}",
    ])

    summary = '\n'.join(lines)

    with open(output_path, 'w') as f:
        f.write(summary)

    return summary


def save_phase2_results(
    output_dir: Path,
    primary_results: dict[str, RepresentationResult],
    perm_results: dict[str, PermutationResult],
    cv_results: dict[str, CrossValidationResult],
    hybrid_results: dict[str, HybridModelResult],
    similarity_matrices: Optional[dict[str, np.ndarray]] = None,
) -> None:
    """
    Save all Phase 2 results to output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Primary validation
    with open(output_dir / "primary_validation.json", 'w') as f:
        json.dump({k: asdict(v) for k, v in primary_results.items()}, f, indent=2)

    # Permutation tests
    with open(output_dir / "permutation_tests.json", 'w') as f:
        json.dump({k: asdict(v) for k, v in perm_results.items()}, f, indent=2)

    # Cross-validation
    with open(output_dir / "cross_validation.json", 'w') as f:
        json.dump({k: asdict(v) for k, v in cv_results.items()}, f, indent=2)

    # Hybrid models
    with open(output_dir / "hybrid_models.json", 'w') as f:
        json.dump({k: asdict(v) for k, v in hybrid_results.items()}, f, indent=2)

    # Similarity matrices (if provided)
    if similarity_matrices:
        np.savez_compressed(
            output_dir / "similarity_matrices.npz",
            **similarity_matrices
        )

    # Generate summary
    generate_phase2_summary(
        primary_results, perm_results, cv_results, hybrid_results,
        output_dir / "phase2_summary.md"
    )
