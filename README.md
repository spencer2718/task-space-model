# Task Space Model

A geometric framework for measuring labor market exposure to technological shocks.

**Version 0.5.0**

---

## What This Is

This project develops a measurement framework for studying how technological change affects labor markets. The key idea: occupations are probability distributions over an underlying space of work activities. When automation affects certain activities, the impact propagates through shared structure.

---

## Current Status: Binary Overlap Validated

**v0.5.0 Result:** The labor market signal is **discrete, not continuous**.

Binary activity overlap (Jaccard similarity) strongly predicts wage comovement:
- β = 0.471, t = 8.00, p < 10⁻¹⁵

Sparse Autoencoder (SAE) decomposition found semantically coherent features (17/20 interpretable) but added only marginal value beyond binary counting:
- SAE β = 0.526 (+11.5%, target was +20%)
- SAE R² = 0.00292 (+75%, target was +30%)

**Bottom line:** Occupations that share activities have correlated wages. But the *semantic similarity* between activities adds little beyond binary overlap. The signal is in "do they share activities?" not "how similar are their activities?"

---

## Version History

| Version | Approach | Result |
|---------|----------|--------|
| v0.5.0 | Binary Jaccard + SAE comparison | **Binary validated**, SAE marginal |
| v0.4.2.1 | Robustness audit of v0.4.2 | FAIL — random distances outperformed |
| v0.4.2 | DWA + Recipe Y (text embeddings) | Appeared to pass, spurious |
| v0.4.1 | GWA + Recipe X (PCA) | FAIL — wrong sign |

### Key Findings

| Finding | Evidence |
|---------|----------|
| Dense embeddings add noise | Random > MPNet in v0.4.2.1 |
| Binary overlap is predictive | t = 8.00 in v0.5.0 |
| SAE finds real structure | 17/20 coherent features |
| Structure is fine-grained | ~2,066 effective features, no power law |
| Marginal SAE improvement | +11.5% β, +75% R² |

---

## Theoretical Implications

The paper's theory posits occupations as distributions over a *continuous* activity manifold. Today's evidence suggests the manifold may be better modeled as a **discrete graph**:

| Theoretical Claim | Empirical Reality |
|-------------------|-------------------|
| Activities live on smooth manifold | Activities are discrete tokens |
| Kernels K(a,b) capture spillover | Binary indicators capture most signal |
| σ bandwidth matters | σ is inert (r = 0.999 across values) |
| Geometry enables prediction | Counting suffices |

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
    baseline.py           # Binary Jaccard overlap (v0.5.0)
    sae.py                # Sparse Autoencoder (v0.5.0)
    diagnostics.py        # Phase I coherence checks
    validation.py         # Phase I external validation
    crosswalk.py          # O*NET-SOC to OES crosswalk
tests/
    run_phase_a.py        # Binary baseline validation
    run_phase_b.py        # SAE training
    run_phase_c.py        # Feature inspection
    run_phase_d.py        # SAE vs Binary comparison
data/
    onet/                 # O*NET database files (not in git)
    external/oes/         # OES wage data (not in git)
models/
    sae_v1.pt             # Trained SAE checkpoint
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

# Run binary baseline validation (Phase A)
PYTHONPATH=src python tests/run_phase_a.py

# Train SAE (Phase B) - ~20 minutes on CPU
PYTHONPATH=src python tests/run_phase_b.py

# Inspect features (Phase C)
PYTHONPATH=src python tests/run_phase_c.py

# Compare Binary vs SAE (Phase D)
PYTHONPATH=src python tests/run_phase_d.py
```

---

## Usage

### Binary Overlap (Recommended)

```python
from task_space import (
    build_dwa_occupation_measures,
    compute_binary_overlap,
)

# Build occupation measures
measures = build_dwa_occupation_measures()

# Compute binary Jaccard overlap
result = compute_binary_overlap(measures, threshold=0.0)

# result.overlap_matrix contains pairwise Jaccard similarities
# result.sparsity_stats contains activity coverage statistics
```

### SAE Pipeline (For Research)

```python
from task_space import (
    build_dwa_activity_domain,
    train_sae,
    extract_sparse_features,
    SAEConfig,
)

# Get embeddings
from sentence_transformers import SentenceTransformer
domain = build_dwa_activity_domain()
model = SentenceTransformer('all-mpnet-base-v2')
titles = [domain.activity_names[aid] for aid in domain.activity_ids]
embeddings = model.encode(titles)

# Train SAE
config = SAEConfig(hidden_dim=16384, lambda_l1=0.005)
sae, log = train_sae(embeddings, config)

# Extract sparse features
features = extract_sparse_features(sae, embeddings)
```

---

## The Framework in Brief

1. **Activity Domain**: 2,087 DWAs form a discrete set of work activities.

2. **Occupation Measures**: Each occupation is a probability distribution over activities, constructed from O*NET task importance ratings.

3. **Binary Overlap**: Jaccard similarity = |A ∩ B| / |A ∪ B| where A, B are activity sets.

4. **Validation**: Regress wage comovement on overlap with clustered SEs.

See `paper/main.tex` Section 3 for formal definitions and Section 4 for empirical strategy.

---

## Validation Results (v0.5.0)

### Phase A: Binary Baseline

| Metric | Value |
|--------|-------|
| β | 0.471 |
| SE | 0.059 |
| t | 8.00 |
| p | < 10⁻¹⁵ |
| R² | 0.00167 |
| n_pairs | 246,051 |

### Phase D: Binary vs SAE Comparison

| Metric | Binary | SAE | Δ |
|--------|--------|-----|---|
| β | 0.471 | 0.526 | +11.5% |
| R² | 0.00167 | 0.00292 | +75% |
| Correlation | — | r = 0.824 | — |

SAE passed 2/3 criteria but missed the 20% β improvement threshold.

---

## Next Steps

### Recommended Path
Use **Binary Jaccard** for Phase II exposure calculations. The simplicity of "count shared activities" is a feature—interpretable, reproducible, and validated.

### Alternative Directions
1. **Worker mobility validation** — test if overlap predicts CPS transitions
2. **Supervised probing** — extract economic dimensions (routine/cognitive) directly
3. **Hierarchical structure** — use GWA→IWA→DWA taxonomy

---

## License

Research use only.
