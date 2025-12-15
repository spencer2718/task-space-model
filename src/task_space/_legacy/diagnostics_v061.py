"""
Phase 1 Diagnostics for v0.6.1: Implementation Verification.

This module diagnoses whether the "random > semantic" anomaly from v0.5.0
is due to implementation bugs or substantive findings.

Diagnostic Tasks:
- 1.1.1: Distance distribution diagnostics
- 1.1.2: Similarity orientation check
- 1.1.3: Kernel weight distribution
- 1.1.4: Jaccard-semantic correlation
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as sp_stats
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DistanceDistributionResult:
    """
    Result of distance distribution diagnostics (Task 1.1.1).

    Attributes:
        min, max, mean, std, median: Basic statistics
        percentiles: Dict of percentile values
        effective_range: p95 - p5
        is_degenerate: True if std < 0.01 * mean (distances effectively constant)
        is_clustered: True if effective_range < 0.1
        diagnosis: Human-readable interpretation
    """
    min: float
    max: float
    mean: float
    std: float
    median: float
    percentiles: dict[str, float]
    effective_range: float
    coefficient_of_variation: float
    is_degenerate: bool
    is_clustered: bool
    diagnosis: str


@dataclass
class SimilarityOrientationResult:
    """
    Result of similarity orientation check (Task 1.1.2).

    Attributes:
        similar_pairs: List of (pair_text, similarity_score)
        dissimilar_pairs: List of (pair_text, similarity_score)
        mean_similar: Mean similarity of known-similar pairs
        mean_dissimilar: Mean similarity of known-dissimilar pairs
        orientation_correct: True if similar > dissimilar by at least 0.1
        gap: mean_similar - mean_dissimilar
        diagnosis: Human-readable interpretation
    """
    similar_pairs: list[tuple[str, float]]
    dissimilar_pairs: list[tuple[str, float]]
    mean_similar: float
    mean_dissimilar: float
    orientation_correct: bool
    gap: float
    diagnosis: str


@dataclass
class KernelWeightResult:
    """
    Result for a single sigma value in kernel weight diagnostics.
    """
    sigma: float
    weight_mean: float
    weight_std: float
    weight_min: float
    weight_max: float
    weight_p5: float
    weight_p95: float
    weight_range: float
    is_collapsed: bool


@dataclass
class KernelWeightDiagnosticsResult:
    """
    Result of kernel weight distribution diagnostics (Task 1.1.3).

    Attributes:
        sigma_results: Dict mapping sigma label to KernelWeightResult
        recommended_sigma: Median of non-zero distances
        all_collapsed: True if all sigmas produce collapsed weights
        diagnosis: Human-readable interpretation
    """
    sigma_results: dict[str, KernelWeightResult]
    recommended_sigma: float
    all_collapsed: bool
    diagnosis: str


@dataclass
class JaccardSemanticCorrelationResult:
    """
    Result of Jaccard-semantic correlation analysis (Task 1.1.4).

    Attributes:
        pearson_r: Pearson correlation coefficient
        spearman_rho: Spearman rank correlation
        n_pairs: Number of occupation pairs analyzed
        interpretation: Category of result
        diagnosis: Human-readable interpretation
    """
    pearson_r: float
    spearman_rho: float
    pearson_pvalue: float
    spearman_pvalue: float
    n_pairs: int
    interpretation: str  # "sign_flip", "orthogonal", "aligned"
    diagnosis: str


# =============================================================================
# Task 1.1.1: Distance Distribution Diagnostics
# =============================================================================

def diagnose_distance_distribution(dist_matrix: np.ndarray) -> DistanceDistributionResult:
    """
    Compute distribution statistics for distance matrix.

    Args:
        dist_matrix: Square distance matrix (n x n)

    Returns:
        DistanceDistributionResult with statistics and diagnosis.

    Decision criteria:
        - If std < 0.01 * mean → distances are effectively constant → BUG
        - If effective_range < 0.1 → distances clustered → needs recalibration
        - If distribution looks reasonable → proceed to next diagnostic
    """
    n = dist_matrix.shape[0]

    # Get upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(n, k=1)
    pairwise_dists = dist_matrix[triu_indices]

    # Basic statistics
    min_val = float(np.min(pairwise_dists))
    max_val = float(np.max(pairwise_dists))
    mean_val = float(np.mean(pairwise_dists))
    std_val = float(np.std(pairwise_dists))
    median_val = float(np.median(pairwise_dists))

    # Percentiles
    percentiles = {
        "p5": float(np.percentile(pairwise_dists, 5)),
        "p10": float(np.percentile(pairwise_dists, 10)),
        "p25": float(np.percentile(pairwise_dists, 25)),
        "p50": float(np.percentile(pairwise_dists, 50)),
        "p75": float(np.percentile(pairwise_dists, 75)),
        "p90": float(np.percentile(pairwise_dists, 90)),
        "p95": float(np.percentile(pairwise_dists, 95)),
    }

    effective_range = percentiles["p95"] - percentiles["p5"]

    # Coefficient of variation (robust to scale)
    cv = std_val / mean_val if mean_val > 0 else 0.0

    # Check for degenerate/clustered distributions
    is_degenerate = cv < 0.01
    is_clustered = effective_range < 0.1

    # Diagnosis
    if is_degenerate:
        diagnosis = (
            f"BUG CONFIRMED: Distances are effectively constant. "
            f"CV = {cv:.4f} < 0.01. Check embedding computation."
        )
    elif is_clustered:
        diagnosis = (
            f"WARNING: Distances are clustered in narrow band. "
            f"Effective range = {effective_range:.4f} < 0.1. "
            f"Kernel bandwidth needs recalibration. Consider normalizing embeddings."
        )
    else:
        diagnosis = (
            f"OK: Distance distribution appears reasonable. "
            f"CV = {cv:.4f}, effective range = {effective_range:.4f}. "
            f"Proceed to similarity orientation check."
        )

    return DistanceDistributionResult(
        min=min_val,
        max=max_val,
        mean=mean_val,
        std=std_val,
        median=median_val,
        percentiles=percentiles,
        effective_range=effective_range,
        coefficient_of_variation=cv,
        is_degenerate=is_degenerate,
        is_clustered=is_clustered,
        diagnosis=diagnosis,
    )


# =============================================================================
# Task 1.1.2: Similarity Orientation Check
# =============================================================================

# Hardcoded test pairs from spec
SIMILAR_PAIRS = [
    ("Analyze data to identify trends", "Interpret statistical data"),
    ("Write reports", "Document findings"),
    ("Operate machinery", "Control equipment"),
]

DISSIMILAR_PAIRS = [
    ("Analyze data to identify trends", "Lift heavy objects"),
    ("Write reports", "Operate forklifts"),
    ("Counsel patients", "Weld metal parts"),
]


def verify_similarity_orientation(
    model_name: str = "all-mpnet-base-v2",
    device: str = "cuda",
) -> SimilarityOrientationResult:
    """
    Test that similar activities have HIGH similarity scores.

    Uses hardcoded test pairs to verify the embedding model produces
    sensible similarity rankings.

    Args:
        model_name: Sentence transformer model to test
        device: Device to use ("cuda" for ROCm/CUDA, "cpu" otherwise)

    Returns:
        SimilarityOrientationResult with pair similarities and diagnosis.

    Decision criteria:
        - If mean(similar) < mean(dissimilar) → SIGN FLIP CONFIRMED
        - If mean(similar) > mean(dissimilar) by at least 0.1 → orientation correct
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers required. pip install sentence-transformers")

    # Load model
    print(f"Loading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device)

    def get_similarity(text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        embeddings = model.encode([text1, text2], convert_to_numpy=True)
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0, 0]
        return float(sim)

    # Compute similarities for test pairs
    similar_results = []
    for t1, t2 in SIMILAR_PAIRS:
        sim = get_similarity(t1, t2)
        similar_results.append((f"{t1} <-> {t2}", sim))

    dissimilar_results = []
    for t1, t2 in DISSIMILAR_PAIRS:
        sim = get_similarity(t1, t2)
        dissimilar_results.append((f"{t1} <-> {t2}", sim))

    # Statistics
    mean_similar = float(np.mean([s for _, s in similar_results]))
    mean_dissimilar = float(np.mean([s for _, s in dissimilar_results]))
    gap = mean_similar - mean_dissimilar

    orientation_correct = gap >= 0.1

    # Diagnosis
    if mean_similar < mean_dissimilar:
        diagnosis = (
            f"BUG CONFIRMED: Sign flip detected! "
            f"Similar pairs ({mean_similar:.3f}) < Dissimilar pairs ({mean_dissimilar:.3f}). "
            f"Use distance = 1 - similarity."
        )
    elif gap < 0.1:
        diagnosis = (
            f"WARNING: Weak discrimination. "
            f"Similar ({mean_similar:.3f}) vs Dissimilar ({mean_dissimilar:.3f}), gap = {gap:.3f}. "
            f"Model may not capture semantic similarity well for work activities."
        )
    else:
        diagnosis = (
            f"OK: Orientation correct. "
            f"Similar ({mean_similar:.3f}) > Dissimilar ({mean_dissimilar:.3f}), gap = {gap:.3f}. "
            f"Proceed to kernel weight check."
        )

    return SimilarityOrientationResult(
        similar_pairs=similar_results,
        dissimilar_pairs=dissimilar_results,
        mean_similar=mean_similar,
        mean_dissimilar=mean_dissimilar,
        orientation_correct=orientation_correct,
        gap=gap,
        diagnosis=diagnosis,
    )


# =============================================================================
# Task 1.1.3: Kernel Weight Distribution
# =============================================================================

def diagnose_kernel_weights(
    dist_matrix: np.ndarray,
    sigma_grid: Optional[list[float]] = None,
) -> KernelWeightDiagnosticsResult:
    """
    For each sigma, compute kernel weight distribution.

    Tests whether the kernel produces meaningful variation in weights
    or whether all weights collapse to similar values.

    Args:
        dist_matrix: Square distance matrix (n x n)
        sigma_grid: List of sigma values to test. If None, uses percentiles.

    Returns:
        KernelWeightDiagnosticsResult with weight statistics and diagnosis.

    Decision criteria:
        - If p95_weight - p5_weight < 0.01 for all sigma → KERNEL COLLAPSE
        - If weights vary meaningfully for at least one sigma → kernel is functional
    """
    n = dist_matrix.shape[0]

    # Get upper triangle distances for computing percentiles
    triu_indices = np.triu_indices(n, k=1)
    pairwise_dists = dist_matrix[triu_indices]
    nonzero_dists = pairwise_dists[pairwise_dists > 0]

    # Recommended sigma: median of non-zero distances
    recommended_sigma = float(np.median(nonzero_dists)) if len(nonzero_dists) > 0 else 1.0

    # Build sigma grid from percentiles if not provided
    if sigma_grid is None:
        sigma_grid = [
            float(np.percentile(nonzero_dists, p)) if len(nonzero_dists) > 0 else 1.0
            for p in [10, 25, 50, 75, 90]
        ]

    sigma_labels = ["p10", "p25", "p50", "p75", "p90"]

    sigma_results = {}
    collapsed_count = 0

    for label, sigma in zip(sigma_labels, sigma_grid):
        if sigma <= 0:
            sigma = 1e-6  # Avoid division by zero

        # Compute exponential kernel (unnormalized)
        K_raw = np.exp(-dist_matrix / sigma)

        # Row-normalize
        row_sums = K_raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        K_norm = K_raw / row_sums

        # Get off-diagonal weights
        K_offdiag = K_norm[triu_indices]

        # Statistics
        weight_mean = float(np.mean(K_offdiag))
        weight_std = float(np.std(K_offdiag))
        weight_min = float(np.min(K_offdiag))
        weight_max = float(np.max(K_offdiag))
        weight_p5 = float(np.percentile(K_offdiag, 5))
        weight_p95 = float(np.percentile(K_offdiag, 95))
        weight_range = weight_p95 - weight_p5

        is_collapsed = weight_range < 0.01
        if is_collapsed:
            collapsed_count += 1

        sigma_results[label] = KernelWeightResult(
            sigma=sigma,
            weight_mean=weight_mean,
            weight_std=weight_std,
            weight_min=weight_min,
            weight_max=weight_max,
            weight_p5=weight_p5,
            weight_p95=weight_p95,
            weight_range=weight_range,
            is_collapsed=is_collapsed,
        )

    all_collapsed = collapsed_count == len(sigma_grid)

    # Diagnosis
    if all_collapsed:
        diagnosis = (
            f"BUG CONFIRMED: Kernel weights collapsed for all sigma values. "
            f"All weight ranges < 0.01. Set sigma = median(distances) = {recommended_sigma:.4f}."
        )
    elif collapsed_count > 0:
        diagnosis = (
            f"WARNING: Kernel weights collapsed for {collapsed_count}/{len(sigma_grid)} sigmas. "
            f"Some bandwidth values work. Recommended sigma = {recommended_sigma:.4f}."
        )
    else:
        diagnosis = (
            f"OK: Kernel weights show variation for all sigma values. "
            f"Kernel is functional. Recommended sigma = {recommended_sigma:.4f}."
        )

    return KernelWeightDiagnosticsResult(
        sigma_results=sigma_results,
        recommended_sigma=recommended_sigma,
        all_collapsed=all_collapsed,
        diagnosis=diagnosis,
    )


# =============================================================================
# Task 1.1.4: Jaccard-Semantic Correlation
# =============================================================================

def correlate_jaccard_semantic(
    jaccard_matrix: np.ndarray,
    semantic_sim_matrix: np.ndarray,
    occupation_codes: Optional[list[str]] = None,
) -> JaccardSemanticCorrelationResult:
    """
    Compute correlation between binary Jaccard and semantic similarity.

    A negative correlation would indicate a sign flip bug.
    Near-zero correlation suggests measures capture different structure.
    Positive correlation suggests measures are aligned.

    Args:
        jaccard_matrix: Binary Jaccard overlap matrix (n_occ x n_occ)
        semantic_sim_matrix: Semantic similarity matrix (n_occ x n_occ)
        occupation_codes: Optional list of occupation codes for labels

    Returns:
        JaccardSemanticCorrelationResult with correlation and interpretation.

    Decision criteria:
        - If r < -0.1 → SIGN FLIP (negative correlation)
        - If r ∈ [-0.1, 0.1] → Orthogonal (substantive finding)
        - If r > 0.3 → Aligned (failure is elsewhere)
    """
    n = jaccard_matrix.shape[0]

    # Verify dimensions match
    if semantic_sim_matrix.shape[0] != n:
        raise ValueError(
            f"Dimension mismatch: Jaccard {n}x{n}, Semantic {semantic_sim_matrix.shape}"
        )

    # Get upper triangle values (excluding diagonal)
    triu_indices = np.triu_indices(n, k=1)
    jaccard_vals = jaccard_matrix[triu_indices]
    semantic_vals = semantic_sim_matrix[triu_indices]

    # Remove any NaN pairs
    valid_mask = ~(np.isnan(jaccard_vals) | np.isnan(semantic_vals))
    jaccard_vals = jaccard_vals[valid_mask]
    semantic_vals = semantic_vals[valid_mask]

    n_pairs = len(jaccard_vals)

    # Compute correlations
    pearson_r, pearson_p = sp_stats.pearsonr(jaccard_vals, semantic_vals)
    spearman_rho, spearman_p = sp_stats.spearmanr(jaccard_vals, semantic_vals)

    # Interpretation
    if pearson_r < -0.1:
        interpretation = "sign_flip"
        diagnosis = (
            f"BUG CONFIRMED: Negative correlation (r = {pearson_r:.3f}). "
            f"Jaccard and semantic measures are anti-correlated. "
            f"Check distance/similarity conversion."
        )
    elif abs(pearson_r) <= 0.1:
        interpretation = "orthogonal"
        diagnosis = (
            f"SUBSTANTIVE: Measures are orthogonal (r = {pearson_r:.3f}). "
            f"Jaccard and semantic capture different structure. "
            f"This may explain why semantic fails—it's measuring something different."
        )
    elif pearson_r <= 0.3:
        interpretation = "weak_positive"
        diagnosis = (
            f"WEAK ALIGNMENT: Modest positive correlation (r = {pearson_r:.3f}). "
            f"Measures partially overlap but capture different aspects. "
            f"Proceed to Phase 2 for alternative measures."
        )
    else:
        interpretation = "aligned"
        diagnosis = (
            f"ALIGNED: Strong positive correlation (r = {pearson_r:.3f}). "
            f"Jaccard and semantic capture similar structure. "
            f"Failure is in kernel/regression, not in measure construction."
        )

    return JaccardSemanticCorrelationResult(
        pearson_r=float(pearson_r),
        spearman_rho=float(spearman_rho),
        pearson_pvalue=float(pearson_p),
        spearman_pvalue=float(spearman_p),
        n_pairs=n_pairs,
        interpretation=interpretation,
        diagnosis=diagnosis,
    )


# =============================================================================
# Helper: Compute semantic similarity matrix from embeddings
# =============================================================================

def compute_semantic_similarity_matrix(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise cosine similarity from embeddings.

    Args:
        embeddings: Embedding matrix (n_samples x embedding_dim)

    Returns:
        Similarity matrix (n_samples x n_samples)
    """
    return cosine_similarity(embeddings)


def compute_occupation_semantic_overlap(
    occupation_matrix: np.ndarray,
    activity_similarity: np.ndarray,
) -> np.ndarray:
    """
    Compute occupation-pair semantic overlap through activity similarities.

    Overlap(i,j) = rho_i^T @ S @ rho_j
    where S is the activity-activity similarity matrix.

    Args:
        occupation_matrix: (n_occ, n_activities) occupation measure matrix
        activity_similarity: (n_activities, n_activities) similarity matrix

    Returns:
        (n_occ, n_occ) semantic overlap matrix
    """
    # Overlap = rho @ S @ rho^T
    overlap = occupation_matrix @ activity_similarity @ occupation_matrix.T

    # Symmetrize (S may not be perfectly symmetric due to numerical issues)
    overlap = (overlap + overlap.T) / 2

    return overlap


# =============================================================================
# Save/Load Results
# =============================================================================

def save_phase1_results(
    output_dir: Path,
    distance_result: DistanceDistributionResult,
    orientation_result: SimilarityOrientationResult,
    kernel_result: KernelWeightDiagnosticsResult,
    correlation_result: JaccardSemanticCorrelationResult,
) -> None:
    """
    Save all Phase 1 results to JSON files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Distance distribution
    with open(output_dir / "distance_distribution.json", "w") as f:
        json.dump(asdict(distance_result), f, indent=2)

    # Similarity orientation
    with open(output_dir / "similarity_orientation.json", "w") as f:
        json.dump(asdict(orientation_result), f, indent=2)

    # Kernel weights
    kernel_dict = {
        "sigma_results": {k: asdict(v) for k, v in kernel_result.sigma_results.items()},
        "recommended_sigma": kernel_result.recommended_sigma,
        "all_collapsed": kernel_result.all_collapsed,
        "diagnosis": kernel_result.diagnosis,
    }
    with open(output_dir / "kernel_weights.json", "w") as f:
        json.dump(kernel_dict, f, indent=2)

    # Jaccard-semantic correlation
    with open(output_dir / "jaccard_semantic_correlation.json", "w") as f:
        json.dump(asdict(correlation_result), f, indent=2)


# =============================================================================
# Phase 1 Fix: Nearest-Neighbor Based Sigma Selection
# =============================================================================

@dataclass
class NNDistanceResult:
    """
    Nearest-neighbor distance statistics for sigma calibration.
    """
    min: float
    max: float
    mean: float
    median: float
    std: float
    p5: float
    p10: float
    p25: float
    p75: float
    p90: float
    p95: float
    sigma_candidates: dict[str, float]


@dataclass
class SigmaDiscriminationResult:
    """
    Kernel discrimination test for a single sigma value.
    """
    sigma: float
    weight_at_p10_dist: float
    weight_at_p90_dist: float
    discrimination_ratio: float
    normalized_weight_range: float
    collapsed: bool


@dataclass
class KernelValidationResult:
    """
    Validation regression result for kernel-weighted overlap.
    """
    sigma: float
    normalized: bool
    beta: float
    se: float
    t_stat: float
    pvalue: float
    r_squared: float
    n_pairs: int
    n_clusters: int
    label: str


def diagnose_nearest_neighbor_distances(dist_matrix: np.ndarray) -> NNDistanceResult:
    """
    Analyze NN distance distribution to inform sigma selection.

    The key insight: sigma should be calibrated to nearest-neighbor distances,
    not the overall distance distribution. This ensures the kernel gives
    high weight to similar activities and low weight to dissimilar ones.

    Args:
        dist_matrix: Square distance matrix (n x n)

    Returns:
        NNDistanceResult with NN statistics and sigma candidates
    """
    n = dist_matrix.shape[0]

    # Temporarily set diagonal to inf to exclude self
    dm = dist_matrix.copy()
    np.fill_diagonal(dm, np.inf)

    nn_distances = dm.min(axis=1)

    stats = NNDistanceResult(
        min=float(nn_distances.min()),
        max=float(nn_distances.max()),
        mean=float(nn_distances.mean()),
        median=float(np.median(nn_distances)),
        std=float(nn_distances.std()),
        p5=float(np.percentile(nn_distances, 5)),
        p10=float(np.percentile(nn_distances, 10)),
        p25=float(np.percentile(nn_distances, 25)),
        p75=float(np.percentile(nn_distances, 75)),
        p90=float(np.percentile(nn_distances, 90)),
        p95=float(np.percentile(nn_distances, 95)),
        sigma_candidates={
            'nn_p10': float(np.percentile(nn_distances, 10)),
            'nn_p25': float(np.percentile(nn_distances, 25)),
            'nn_median': float(np.median(nn_distances)),
            'nn_p75': float(np.percentile(nn_distances, 75)),
        }
    )

    return stats


def test_sigma_discrimination(
    dist_matrix: np.ndarray,
    sigma_candidates: list[float],
) -> dict[str, SigmaDiscriminationResult]:
    """
    Test kernel weight discrimination for each sigma candidate.

    Reports: weight ratio between p10 and p90 of distances.
    Want ratio > 10x for meaningful discrimination.

    Args:
        dist_matrix: Square distance matrix
        sigma_candidates: List of sigma values to test

    Returns:
        Dict mapping sigma to SigmaDiscriminationResult
    """
    # Get p10 and p90 of distances (excluding zeros/diagonal)
    triu_idx = np.triu_indices(dist_matrix.shape[0], k=1)
    dists = dist_matrix[triu_idx]

    p10_dist = float(np.percentile(dists, 10))
    p90_dist = float(np.percentile(dists, 90))

    results = {}

    for sigma in sigma_candidates:
        weight_p10 = np.exp(-p10_dist / sigma)
        weight_p90 = np.exp(-p90_dist / sigma)
        ratio = weight_p10 / weight_p90 if weight_p90 > 0 else float('inf')

        # Compute actual kernel weight distribution
        K = np.exp(-dist_matrix / sigma)
        K_normalized = K / K.sum(axis=1, keepdims=True)

        # Get off-diagonal weights only
        K_offdiag = K_normalized[triu_idx]
        weight_range = float(np.percentile(K_offdiag, 95) - np.percentile(K_offdiag, 5))

        results[f"{sigma:.4f}"] = SigmaDiscriminationResult(
            sigma=sigma,
            weight_at_p10_dist=float(weight_p10),
            weight_at_p90_dist=float(weight_p90),
            discrimination_ratio=float(ratio),
            normalized_weight_range=weight_range,
            collapsed=weight_range < 0.001,  # More lenient threshold
        )

    return results


def compute_kernel_overlap(
    occ_measures: np.ndarray,
    dist_matrix: np.ndarray,
    sigma: float,
    normalize_kernel: bool = True,
) -> np.ndarray:
    """
    Compute kernel-weighted occupation overlap.

    Args:
        occ_measures: (n_occ, n_activities) occupation measure matrix
        dist_matrix: (n_activities, n_activities) distance matrix
        sigma: Kernel bandwidth
        normalize_kernel: If True, row-normalize the kernel

    Returns:
        (n_occ, n_occ) kernel-weighted overlap matrix
    """
    # Compute kernel
    K = np.exp(-dist_matrix / sigma)

    if normalize_kernel:
        K = K / K.sum(axis=1, keepdims=True)

    # Compute overlap: rho @ K @ rho^T
    overlap = occ_measures @ K @ occ_measures.T

    # Symmetrize (K may not be symmetric after normalization)
    overlap = (overlap + overlap.T) / 2

    return overlap


def run_kernel_validation(
    overlap_matrix: np.ndarray,
    comovement_matrix: np.ndarray,
    overlap_codes: list[str],
    comovement_codes: list[str],
    crosswalk_map: dict[str, str],
    sigma: float,
    normalized: bool,
    label: str,
) -> KernelValidationResult:
    """
    Run validation regression: WageComovement ~ KernelOverlap.

    Args:
        overlap_matrix: Kernel-weighted overlap matrix (O*NET occupations)
        comovement_matrix: Wage comovement matrix (SOC occupations)
        overlap_codes: O*NET-SOC codes for overlap matrix
        comovement_codes: SOC codes for comovement matrix
        crosswalk_map: Dict mapping O*NET-SOC to SOC
        sigma: Sigma value used
        normalized: Whether kernel was normalized
        label: Label for this test

    Returns:
        KernelValidationResult with regression statistics
    """
    import pandas as pd

    # Map O*NET occupations to SOC codes
    onet_to_soc = {}
    for onet_code in overlap_codes:
        soc = crosswalk_map.get(onet_code)
        if soc:
            onet_to_soc[onet_code] = soc

    # Find common SOC codes
    soc_codes = list(set(onet_to_soc.values()))
    soc_codes = [soc for soc in soc_codes if soc in comovement_codes]
    soc_codes = sorted(soc_codes)

    n_soc = len(soc_codes)
    soc_to_idx = {soc: i for i, soc in enumerate(soc_codes)}
    comovement_idx = {soc: i for i, soc in enumerate(comovement_codes)}

    # Aggregate overlap to SOC level
    onet_to_overlap_idx = {code: i for i, code in enumerate(overlap_codes)}

    soc_overlap = np.zeros((n_soc, n_soc))
    soc_counts = np.zeros((n_soc, n_soc))

    for onet_i, soc_i in onet_to_soc.items():
        if soc_i not in soc_to_idx:
            continue
        for onet_j, soc_j in onet_to_soc.items():
            if soc_j not in soc_to_idx:
                continue
            if onet_i >= onet_j:
                continue

            i_idx = onet_to_overlap_idx[onet_i]
            j_idx = onet_to_overlap_idx[onet_j]
            overlap_val = overlap_matrix[i_idx, j_idx]

            si = soc_to_idx[soc_i]
            sj = soc_to_idx[soc_j]
            if si > sj:
                si, sj = sj, si

            soc_overlap[si, sj] += overlap_val
            soc_counts[si, sj] += 1

    # Average
    soc_counts[soc_counts == 0] = 1
    soc_overlap = soc_overlap / soc_counts

    # Build pair-level dataset
    pairs = []
    for i in range(n_soc):
        for j in range(i + 1, n_soc):
            soc_i = soc_codes[i]
            soc_j = soc_codes[j]

            ci = comovement_idx[soc_i]
            cj = comovement_idx[soc_j]
            comove_val = comovement_matrix[ci, cj]

            if np.isnan(comove_val):
                continue

            overlap_val = soc_overlap[i, j]

            pairs.append({
                "occ_i": soc_i,
                "occ_j": soc_j,
                "overlap": overlap_val,
                "y": comove_val,
            })

    if len(pairs) == 0:
        raise ValueError("No valid occupation pairs found")

    df = pd.DataFrame(pairs)

    # Regression with clustered SEs
    X_raw = df["overlap"].values
    y = df["y"].values
    clusters = df["occ_i"].values

    # Standardize X to avoid numerical issues (overlap values can be tiny)
    X_mean = X_raw.mean()
    X_std = X_raw.std()
    if X_std < 1e-10:
        X_std = 1.0  # Avoid division by zero
    X = (X_raw - X_mean) / X_std

    unique_clusters, cluster_ids = np.unique(clusters, return_inverse=True)
    n = len(y)
    n_clusters = len(unique_clusters)

    # Add constant
    X_mat = np.column_stack([np.ones(n), X])

    # OLS on standardized X
    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    beta_vec = XtX_inv @ (X_mat.T @ y)
    resid = y - X_mat @ beta_vec

    # R-squared (same regardless of standardization)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Clustered standard errors
    meat = np.zeros((2, 2))
    for c in unique_clusters:
        mask = cluster_ids == c
        cluster_resid = resid[mask]
        cluster_X = X_mat[mask]
        score = cluster_X.T @ cluster_resid
        meat += np.outer(score, score)

    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - 2))
    var_beta = correction * XtX_inv @ meat @ XtX_inv
    se_beta = np.sqrt(np.maximum(np.diag(var_beta), 1e-20))  # Avoid sqrt of negative

    # Convert back to original scale
    beta_standardized = beta_vec[1]
    se_standardized = se_beta[1]

    # Beta in original scale: beta_orig = beta_std / X_std
    beta = beta_standardized / X_std
    se = se_standardized / X_std

    t_stat = beta_standardized / se_standardized  # t-stat is scale-invariant
    pvalue = 2 * (1 - sp_stats.t.cdf(np.abs(t_stat), df=n_clusters - 1))

    return KernelValidationResult(
        sigma=sigma,
        normalized=normalized,
        beta=float(beta),
        se=float(se),
        t_stat=float(t_stat),
        pvalue=float(pvalue),
        r_squared=float(r_squared),
        n_pairs=len(df),
        n_clusters=n_clusters,
        label=label,
    )


def generate_phase1_summary(
    output_dir: Path,
    distance_result: DistanceDistributionResult,
    orientation_result: SimilarityOrientationResult,
    kernel_result: KernelWeightDiagnosticsResult,
    correlation_result: JaccardSemanticCorrelationResult,
) -> str:
    """
    Generate human-readable Phase 1 summary markdown.
    """
    # Determine overall diagnosis
    bugs_found = []
    if distance_result.is_degenerate:
        bugs_found.append("Distance distribution degenerate")
    if distance_result.is_clustered:
        bugs_found.append("Distance distribution clustered")
    if not orientation_result.orientation_correct:
        if orientation_result.mean_similar < orientation_result.mean_dissimilar:
            bugs_found.append("Similarity orientation sign flip")
        else:
            bugs_found.append("Weak similarity discrimination")
    if kernel_result.all_collapsed:
        bugs_found.append("Kernel weights collapsed")
    if correlation_result.interpretation == "sign_flip":
        bugs_found.append("Jaccard-semantic negative correlation")

    overall_status = "BUG FOUND" if bugs_found else "NO BUGS FOUND"

    summary = f"""# Phase 1 Diagnostic Summary

**Status:** {overall_status}

---

## Task 1.1.1: Distance Distribution

| Metric | Value |
|--------|-------|
| Min | {distance_result.min:.4f} |
| Max | {distance_result.max:.4f} |
| Mean | {distance_result.mean:.4f} |
| Std | {distance_result.std:.4f} |
| CV | {distance_result.coefficient_of_variation:.4f} |
| Effective Range (p95-p5) | {distance_result.effective_range:.4f} |

**Diagnosis:** {distance_result.diagnosis}

---

## Task 1.1.2: Similarity Orientation

### Similar Pairs
| Pair | Similarity |
|------|------------|
"""
    for pair, sim in orientation_result.similar_pairs:
        summary += f"| {pair} | {sim:.4f} |\n"

    summary += f"""
### Dissimilar Pairs
| Pair | Similarity |
|------|------------|
"""
    for pair, sim in orientation_result.dissimilar_pairs:
        summary += f"| {pair} | {sim:.4f} |\n"

    summary += f"""
| Metric | Value |
|--------|-------|
| Mean Similar | {orientation_result.mean_similar:.4f} |
| Mean Dissimilar | {orientation_result.mean_dissimilar:.4f} |
| Gap | {orientation_result.gap:.4f} |
| Orientation Correct | {orientation_result.orientation_correct} |

**Diagnosis:** {orientation_result.diagnosis}

---

## Task 1.1.3: Kernel Weight Distribution

| Sigma | Weight Mean | Weight Std | Range (p95-p5) | Collapsed? |
|-------|-------------|------------|----------------|------------|
"""
    for label, result in kernel_result.sigma_results.items():
        collapsed = "YES" if result.is_collapsed else "no"
        summary += f"| {label} ({result.sigma:.4f}) | {result.weight_mean:.4f} | {result.weight_std:.4f} | {result.weight_range:.4f} | {collapsed} |\n"

    summary += f"""
**Recommended sigma:** {kernel_result.recommended_sigma:.4f}

**Diagnosis:** {kernel_result.diagnosis}

---

## Task 1.1.4: Jaccard-Semantic Correlation

| Metric | Value |
|--------|-------|
| Pearson r | {correlation_result.pearson_r:.4f} |
| Spearman ρ | {correlation_result.spearman_rho:.4f} |
| Pearson p-value | {correlation_result.pearson_pvalue:.2e} |
| Spearman p-value | {correlation_result.spearman_pvalue:.2e} |
| N pairs | {correlation_result.n_pairs:,} |
| Interpretation | {correlation_result.interpretation} |

**Diagnosis:** {correlation_result.diagnosis}

---

## Overall Decision

"""
    if bugs_found:
        summary += "### Bugs Found:\n"
        for bug in bugs_found:
            summary += f"- {bug}\n"
        summary += "\n**ACTION:** Fix bugs before proceeding to Phase 2.\n"
    else:
        summary += """### No Implementation Bugs Found

The implementation appears correct. The failure of kernel-weighted overlap is likely **substantive**, not technical.

**ACTION:** Proceed to Phase 2 (Alternative Distances) to test whether O*NET structured dimensions or domain-specific embeddings perform better.
"""

    return summary
