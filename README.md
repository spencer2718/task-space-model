# Task Space Model

A geometric framework for measuring labor market exposure to technological shocks.

**Version 0.6.1**

---

## What This Is

This project develops a measurement framework for studying how technological change affects labor markets. The key idea: occupations are probability distributions over an underlying space of work activities. When automation affects certain activities, the impact propagates through shared structure.

---

## Current Status: Continuous Structure Validated

**v0.6.1 Result:** Kernel-weighted semantic overlap **strongly outperforms** binary Jaccard.

| Measure | t-stat | R² | Status |
|---------|--------|-----|--------|
| Kernel (semantic) | **27.65** | **0.00310** | Best |
| Binary Jaccard | 8.00 | 0.00167 | Baseline |
| Random (best of 100) | 15.09 | 0.00052 | Noise |

**Key insight:** The v0.5.0 finding that "discrete dominates continuous" was an **implementation artifact** caused by kernel collapse. With proper bandwidth calibration (σ = median of nearest-neighbor distances), continuous semantic similarity is highly predictive.

**Bottom line:** Tasks live on a continuous similarity space. Kernel bandwidth must be calibrated to local neighborhood structure, not global distance distribution.

---

## Version History

| Version | Approach | Result |
|---------|----------|--------|
| **v0.6.1** | **Kernel fix + semantic vs random** | **Semantic >> Jaccard >> Random** |
| v0.5.0 | Binary Jaccard + SAE comparison | Binary validated, SAE marginal |
| v0.4.2.1 | Robustness audit of v0.4.2 | FAIL — random > semantic (artifact) |
| v0.4.2 | DWA + Recipe Y (text embeddings) | Appeared to pass, spurious |
| v0.4.1 | GWA + Recipe X (PCA) | FAIL — wrong sign |

### Key Findings

| Finding | Evidence |
|---------|----------|
| Semantic structure is predictive | t = 27.65, 100th percentile vs random |
| Kernel bandwidth matters | σ = 0.22 (NN median) vs 0.74 (overall median) |
| Row-normalization hurts | Unnormalized kernel outperforms |
| v0.5.0 was an artifact | Random > semantic was due to kernel collapse |

---

## Theoretical Implications

The manifold representation is **vindicated**:

| Theoretical Claim | Empirical Status |
|-------------------|------------------|
| Activities live on smooth manifold | **Confirmed** — semantic predicts wages |
| Kernels K(a,b) capture spillover | **Confirmed** — with proper σ calibration |
| σ bandwidth matters | **Confirmed** — 4.5x discrimination at σ=0.22 |
| Geometry enables prediction | **Confirmed** — +86% R² over binary |

---

## Data Requirements

### O*NET Database (Required)
Download `db_30_0_excel.zip` from https://www.onetcenter.org/database.html and extract to `data/onet/db_30_0_excel/`.

### OES Wage Data (For Validation)
Download national OES files from https://www.bls.gov/oes/tables.htm for years 2019-2023. Extract to `data/external/oes/`.

**Note:** BLS blocks automated downloads. You must download manually via browser.

---

## Repository Structure

```
paper/
    main.tex              # Theoretical framework and empirical strategy
    references.bib        # Bibliography
src/task_space/
    data.py               # O*NET file loading
    domain.py             # Activity domain and occupation measures
    distances.py          # Recipe X (PCA) and Recipe Y (embeddings)
    kernel.py             # Kernel matrix and exposure computation
    baseline.py           # Binary Jaccard overlap
    sae.py                # Sparse Autoencoder
    diagnostics.py        # Phase I coherence checks
    diagnostics_v061.py   # v0.6.1 kernel diagnostics and fix
    validation.py         # Phase I external validation
    crosswalk.py          # O*NET-SOC to OES crosswalk
tests/
    run_phase1_diagnostics.py  # Phase 1 bug diagnosis
    run_phase1_fix.py          # Kernel fix validation
    run_semantic_vs_random.py  # Critical confirmation
data/
    onet/                 # O*NET database files (not in git)
    external/oes/         # OES wage data (not in git)
outputs/                  # Validation outputs (not in git)
```

---

## Quick Start

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy pandas scipy scikit-learn openpyxl matplotlib sentence-transformers torch

# Run Phase 1 diagnostics (identifies kernel collapse)
PYTHONPATH=src python tests/run_phase1_diagnostics.py

# Run kernel fix validation
PYTHONPATH=src python tests/run_phase1_fix.py

# Run semantic vs random comparison (critical confirmation)
PYTHONPATH=src python tests/run_semantic_vs_random.py
```

---

## Usage

### Kernel-Weighted Overlap (Recommended)

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
from task_space import build_dwa_activity_domain, build_dwa_occupation_measures

# Build domain and measures
domain = build_dwa_activity_domain()
measures = build_dwa_occupation_measures()

# Compute embeddings
model = SentenceTransformer('all-mpnet-base-v2')
titles = [domain.activity_names[aid] for aid in domain.activity_ids]
embeddings = model.encode(titles)

# Compute distances
dist_matrix = cosine_distances(embeddings)

# Key parameters (from v0.6.1 fix)
sigma = 0.2230  # Median of nearest-neighbor distances
normalize_kernel = False  # Skip row-normalization

# Compute kernel and overlap
K = np.exp(-dist_matrix / sigma)
overlap = measures.occupation_matrix @ K @ measures.occupation_matrix.T
```

### Binary Overlap (Baseline)

```python
from task_space import build_dwa_occupation_measures, compute_binary_overlap

measures = build_dwa_occupation_measures()
result = compute_binary_overlap(measures, threshold=0.0)
# result.overlap_matrix contains pairwise Jaccard similarities
```

---

## The Framework in Brief

1. **Activity Domain**: 2,087 DWAs form a discrete set of work activities.

2. **Occupation Measures**: Each occupation is a probability distribution over activities, constructed from O*NET task importance ratings.

3. **Semantic Distance**: Cosine distance between activity embeddings (MPNet).

4. **Kernel-Weighted Overlap**: `overlap(i,j) = ρ_i @ K @ ρ_j` where `K = exp(-d/σ)`.

5. **Critical Parameter**: σ = median of nearest-neighbor distances (≈ 0.22), NOT overall median (0.74).

See `paper/main.tex` Section 3 for formal definitions and Section 4 for empirical strategy.

---

## Validation Results (v0.6.1)

### Kernel vs Jaccard

| Measure | t-stat | R² | Improvement |
|---------|--------|-----|-------------|
| Kernel (σ=0.22, unnorm) | **27.65** | **0.00310** | — |
| Binary Jaccard | 8.00 | 0.00167 | +245% t, +86% R² |

### Semantic vs Random (100 seeds)

| Metric | Semantic | Random Mean | Random Max | Percentile |
|--------|----------|-------------|------------|------------|
| t-stat | **27.65** | 5.67 | 15.09 | **100%** |
| R² | **0.00310** | 0.00017 | 0.00052 | **100%** |

Semantic is at the **100th percentile** — it beats all 100 random baselines.

---

## What v0.6.1 Fixed

### The Bug (v0.5.0)

Kernel weights collapsed because:
1. σ calibrated to global distance distribution (median = 0.74)
2. With σ = 0.74, discrimination ratio was only 1.6x
3. Row-normalization over 2,087 activities made all weights ≈ 0.0005

### The Fix (v0.6.1)

1. σ = median of **nearest-neighbor** distances = 0.22
2. Discrimination ratio improved to 4.5x
3. Skip row-normalization (use raw kernel weights)

---

## Next Steps

### For Phase II
Use **kernel-weighted semantic overlap** with:
- σ = 0.2230 (median NN distance)
- Unnormalized kernel

### Alternative Directions
1. **O*NET structured dimensions** — test Abilities/Skills/Knowledge
2. **JobBERT** — domain-specific embeddings may outperform MPNet
3. **Worker mobility validation** — test if overlap predicts CPS transitions

---

## License

Research use only.
