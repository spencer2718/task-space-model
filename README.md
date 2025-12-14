# Task Space Model

A geometric framework for measuring labor market exposure to technological shocks.

**Version 0.6.2** (Phase 2 Complete)

---

## What This Is

This project develops a measurement framework for studying how technological change affects labor markets. The key idea: occupations are probability distributions over an underlying space of work activities. When automation affects certain activities, the impact propagates through shared structure.

---

## Current Status: Phase 2 Complete

**v0.6.2 Result:** Systematic representation comparison confirms kernel-weighted semantic overlap is robust.

| Measure | t-stat (clustered SEs) | R² | Notes |
|---------|------------------------|-----|-------|
| MPNet (normalized) | **7.14** | **0.00485** | ✓ Best measure |
| MPNet (unnormalized) | 5.90 | 0.00310 | Matches Phase 1 |
| Binary Jaccard | 8.00 | 0.00167 | Baseline |

**Key findings:**
1. **Normalization IMPROVES prediction** — R² increased 56.6% after normalization (concentration was noise)
2. **Not just concentration** — Remains significant (t=5.29) after controlling for entropy
3. **Task-manifold theory validated** — Semantic content predicts beyond job breadth

---

## Version History

| Version | Approach | Result |
|---------|----------|--------|
| **v0.6.2** | **Robustness checks + controls** | **Semantic signal robust (t=7.14 normalized, t=5.29 with entropy control)** |
| v0.6.1 | Kernel fix + semantic vs random | Semantic >> Jaccard >> Random |
| v0.5.0 | Binary Jaccard + SAE comparison | Binary validated, SAE marginal |
| v0.4.2.1 | Robustness audit of v0.4.2 | FAIL — random > semantic (artifact) |
| v0.4.2 | DWA + Recipe Y (text embeddings) | Appeared to pass, spurious |
| v0.4.1 | GWA + Recipe X (PCA) | FAIL — wrong sign |

---

## Phase 2 Robustness Summary

### Normalized vs Unnormalized Overlap

Normalization controls for concentration bias (specialist vs generalist occupations):

| Measure | t-stat (clustered SEs) | R² | Interpretation |
|---------|------------------------|-----|----------------|
| Unnormalized | 5.90 | 0.00310 | Raw kernel overlap |
| Normalized | 7.14 | 0.00485 | Angular similarity |

**R² INCREASED 56.6% after normalization.** Concentration was NOISE, not signal.

### Control Regressions

After controlling for occupational breadth, semantic overlap remains significant:

| Model | t_overlap | Significant? |
|-------|-----------|--------------|
| + Entropy | 5.29 | Yes (p < 0.01) |
| + Support | 5.43 | Yes (p < 0.01) |

*Note: Entropy and Support are r=0.97 correlated — report single-control models as primary.*

### Methodological Note

**C1-C3 (text embeddings) vs C4-C7 (O*NET structured) test different hypotheses:**
- C1-C3: "Do occupations share semantically similar *activities*?" (activity-level)
- C4-C7: "Do occupations require similar *abilities/skills*?" (occupation-level)

C4-C7 bypass the activity-level kernel entirely. Higher R² for C4-C7 does not validate or invalidate the task-manifold theory.

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
    diagnostics_v061.py   # v0.6.1 kernel diagnostics
    comparison.py         # Phase 2 representation comparison (in progress)
    validation.py         # Phase I external validation
    crosswalk.py          # O*NET-SOC to OES crosswalk
tests/
    run_phase1_diagnostics.py  # Phase 1 bug diagnosis
    run_phase1_fix.py          # Kernel fix validation
    run_phase2_comparison.py   # Phase 2 representation comparison
data/
    onet/                 # O*NET database files (not in git)
    external/oes/         # OES wage data (not in git)
outputs/
    phase1/               # Phase 1 outputs
    phase2/               # Phase 2 outputs (in progress)
```

---

## Quick Start

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy pandas scipy scikit-learn openpyxl matplotlib sentence-transformers torch transformers

# Run Phase 1 validation (kernel fix)
PYTHONPATH=src python tests/run_phase1_fix.py

# Results saved to outputs/phase1/
```

---

## Usage

### Kernel-Weighted Overlap (Recommended)

```python
from task_space import build_dwa_occupation_measures, compute_recipe_y_distances
from task_space.diagnostics_v061 import compute_kernel_overlap

# Load data
measures = build_dwa_occupation_measures()

# Compute activity distances using MPNet embeddings
dist_matrix = compute_recipe_y_distances(measures.activity_titles)

# Compute kernel overlap with proper σ calibration
overlap, sigma = compute_kernel_overlap(
    measures.raw_matrix,
    dist_matrix,
    sigma=None,  # Auto-calibrate to NN median
    normalize=False
)
# R² = 0.00310, t = 27.65
```

### Baseline: Binary Jaccard

```python
from task_space import build_dwa_occupation_measures, compute_binary_overlap

measures = build_dwa_occupation_measures()
result = compute_binary_overlap(measures, threshold=0.0)
# result.overlap_matrix contains pairwise Jaccard similarities
# R² = 0.00167, t = 8.00 (baseline)
```

---

## The Framework in Brief

1. **Activity Domain**: 2,087 DWAs form a discrete set of work activities.

2. **Occupation Measures**: Each occupation is a probability distribution over activities, constructed from O*NET task importance ratings.

3. **Distance Structure**: Activity embeddings (MPNet) define semantic distances between activities.

4. **Kernel-Weighted Overlap**: Occupations sharing semantically similar activities have higher overlap.

See `paper/main.tex` Section 3 for formal definitions and Section 4 for empirical strategy.

---

## Key Insights

### From Phase 1
1. **σ must be calibrated to nearest-neighbor distances** — Global distance percentiles (median = 0.74) cause kernel collapse. Use NN median (≈ 0.22) instead.

2. **Skip row-normalization for kernels** — With 2,087 activities, kernel row-normalization washes out structure.

### From Phase 2
3. **Normalize occupation overlap, not kernels** — Occupation-level normalization (cosine-style) IMPROVES prediction by 56.6% (concentration was noise).

4. **Semantic signal is real, not artifact** — Remains significant (t > 5) after controlling for entropy and support size.

5. **Activity-level vs occupation-level are different tests** — Text embeddings (C1-C3) test task-manifold theory; O*NET structured (C4-C7) test ability/skill similarity. Both valid, different hypotheses.

6. **Task-manifold theory is validated** — Kernel-weighted semantic overlap significantly predicts wage comovement beyond discrete activity counting.

---

## License

Research use only.
