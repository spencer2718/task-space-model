# Distance Metric Decision Guide

This guide helps researchers choose the appropriate distance metric. The key finding from our validation is that **the embedding representation drives a 74.9% improvement** over O*NET baselines, while the aggregation method (Wasserstein vs. centroid) makes negligible difference.

## Quick Decision Tree
```
Need occupation distance metric?
├─ New application/research? → Use **cosine_embed** (primary metric)
├─ Need transport plan (which tasks map)? → Use **wasserstein** (theoretical grounding)
├─ Comparing to prior literature? → Match their metric (likely O*NET-based)
└─ Using normalized_kernel? → STOP - deprecated per HC1
```

## Metric Specifications

### 1. Cosine Distance on Centroids (Embedding-Based) ✓ PRIMARY

**When to use:**
- Individual transition prediction
- Pathway analysis
- Policy evaluation (automation impact, retraining programs)
- Any new research application

**Advantages:**
- Best predictive performance (14.1% pseudo-R²)
- Captures semantic task similarity via MPNet embeddings
- Fast computation (no optimal transport needed)
- Nearly identical rankings to Wasserstein (ρ = 0.95)

**Registry key:** `"cosine_embed"`

### 2. Wasserstein Distance (Embedding-Based) ✓ THEORETICAL GROUNDING

**When to use:**
- When transport plan interpretation is needed (which tasks map to which)
- Theoretical analysis requiring metric space properties
- Triangle inequality is required

**Advantages:**
- Economic interpretation as minimum task transformation cost
- True metric (satisfies triangle inequality at O*NET level)
- Transport plan reveals task-to-task mapping

**Limitations:**
- Marginally underperforms centroid after diagonal correction (13.8% vs 14.1%)
- 100x slower computation than centroid
- Census-level aggregation introduces nonzero diagonal

**Registry key:** `"wasserstein"`

### 3. O*NET Cosine Distance ⚠️ LEGACY ONLY

**When to use:**
- Replicating prior literature
- Baseline comparisons

**Limitations:**
- 78% of occupation pairs at maximum distance (sparsity problem)
- Misses semantic similarity (forklift ≠ delivery truck in O*NET)
- 8.1% pseudo-R² (vs 14.1% for centroid)

**Registry key:** `"cosine_onet"`

### 4. O*NET Euclidean Distance ⚠️ NOT RECOMMENDED

**When to use:** Only for exact replication of specific prior work.

**Registry key:** `"euclidean_dwa"`

### 5. Normalized Kernel Overlap ❌ DEPRECATED

Deprecated per HC1. Replace with `"cosine_embed"`.

## Key Findings Summary

From the 2×2 methodology comparison (n = 89,329 transitions, J = 11 sampled alternatives):

| Method | Embedding | Aggregation | Pseudo-R² | Verdict |
|--------|-----------|-------------|-----------|---------|
| cosine_embed | MPNet | Centroid | 14.08% | **Primary** |
| wasserstein | MPNet | Optimal transport | 13.76% | Theoretical grounding |
| cosine_onet | O*NET | Cosine | 8.05% | Legacy only |
| euclidean_dwa | O*NET | Euclidean | 6.06% | Avoid |

**Core insight:** The embedding space (MPNet vs O*NET) matters far more than the aggregation method (Wasserstein vs centroid). MPNet embeddings capture that "operating forklift" ≈ "driving delivery vehicle" — a semantic similarity invisible to O*NET's formal taxonomy.

Ground metric validation: embedding ground metric improves pseudo-R² by +83% over identity ground metric (7.52% → 13.76%), confirming that semantic task similarity is the mechanism.

## References

- Hard Constraint HC1: Centroid is primary specification; Wasserstein provides theoretical grounding
- Diagonal correction: v0.7.7.0 (nonzero diagonal from SOC→Census aggregation)
- Validation results: LEDGER.md, `outputs/experiments/distance_head_to_head_v0732.json`
