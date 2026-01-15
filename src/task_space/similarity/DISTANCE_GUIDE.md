# Distance Metric Decision Guide

This guide helps researchers choose the appropriate distance metric for their application. The key finding from our validation is that **embedding choice drives 75-96% of improvement**, while distributional treatment adds only 3%.

## Quick Decision Tree

```
Need occupation distance metric?
├─ New application/research? → Use **wasserstein** (primary metric)
├─ Need computational speed? → Use **cosine_embed** (ρ = 0.95 with wasserstein)
├─ Comparing to prior literature? → Match their metric (likely O*NET-based)
└─ Using normalized_kernel? → STOP - deprecated per HC1, use wasserstein
```

## Metric Specifications

### 1. Wasserstein Distance (Embedding-Based) ✓ RECOMMENDED

**When to use:** 
- Individual transition prediction
- Pathway analysis 
- Policy evaluation (automation impact, retraining programs)
- Any new research application

**Advantages:**
- Strong theoretical grounding as minimum task transformation cost
- Best predictive performance (14.5% pseudo-R²)
- Captures semantic task similarity via MPNet embeddings
- Economic interpretation: effort to transform task distributions

**Registry key:** `"wasserstein"`

### 2. Cosine Distance on Centroids (Embedding-Based) ✓ ACCEPTABLE

**When to use:**
- Computational speed is critical (100x faster than Wasserstein)
- Large-scale screening or initial exploration
- When ρ = 0.95 correlation with Wasserstein is sufficient

**Advantages:**
- Nearly identical rankings to Wasserstein (ρ = 0.95)
- Much faster computation (no optimal transport)
- Still leverages MPNet embeddings

**Registry key:** `"cosine_embed"`

### 3. O*NET Cosine Distance ⚠️ LEGACY ONLY

**When to use:**
- Replicating prior literature
- Baseline comparisons
- When required for methodological consistency

**Limitations:**
- 78% of occupation pairs at maximum distance (sparsity problem)
- Misses semantic similarity (forklift ≠ delivery truck in O*NET)
- Poor individual prediction (8% pseudo-R²)

**Registry key:** `"cosine_onet"`

### 4. O*NET Euclidean Distance (DWA-based) ⚠️ NOT RECOMMENDED

**When to use:**
- Only for exact replication of specific prior work

**Limitations:**
- Worst performance across all tests
- No meaningful economic interpretation
- Treats all activities as orthogonal

**Registry key:** `"euclidean_dwa"`

### 5. Normalized Kernel Overlap ❌ DEPRECATED

**When to use:** Never for distance applications

**Status:** Deprecated per Hard Constraint HC1. The kernel overlap method was designed for density estimation, not distance measurement. Any code using `normalized_kernel` for distances should be updated.

**Migration:** Replace with `"wasserstein"` or `"cosine_embed"`

## Key Findings Summary

From the 2×2 methodology comparison (n=89,329 transitions):

| Method | Embedding | Aggregation | Pseudo-R² | Verdict |
|--------|-----------|-------------|-----------|---------|
| wasserstein | MPNet | Optimal transport | 14.51% | **Best** |
| cosine_embed | MPNet | Centroid | 14.08% | Good approximation |
| cosine_onet | O*NET | Centroid | 8.05% | Legacy only |
| euclidean_dwa | O*NET | Raw DWA vectors | 6.06% | Avoid |

**Core insight:** The embedding space (MPNet vs O*NET) matters far more than the aggregation method (Wasserstein vs centroid). MPNet embeddings capture that "operating forklift" ≈ "driving delivery vehicle"—a semantic similarity invisible to O*NET's formal activity taxonomy.

## Implementation Example

```python
from task_space.similarity import get_distance_computer

# Recommended approach
computer = get_distance_computer("wasserstein")
distances = computer.compute_matrix(occupations)

# Fast approximation
computer_fast = get_distance_computer("cosine_embed")
distances_fast = computer_fast.compute_matrix(occupations)

# Legacy comparison
computer_legacy = get_distance_computer("cosine_onet")
distances_legacy = computer_legacy.compute_matrix(occupations)
```

## References

- Hard Constraint HC1: Wasserstein is primary geometry (ΔLL = +9,576)
- Validation results: LEDGER.md Section "T Module: 2×2 Methodology Comparison"
- Theoretical foundation: paper/main.tex Section 3 (Wasserstein Geometry)