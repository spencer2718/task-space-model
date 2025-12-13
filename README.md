# Task Space Model

A geometric framework for measuring labor market exposure to technological shocks.

**Version 0.4.2.1**

---

## What This Is

This project develops a measurement framework for studying how technological change affects labor markets. The key idea: occupations are probability distributions over an underlying space of work activities. When automation affects certain activities, the impact propagates through the geometry of that space, affecting occupations proportionally to their overlap with the shock.

This approach allows for:
- Continuous gradients of exposure across occupations (vs binary "exposed/unexposed")
- Spillover effects between related activities
- Empirical validation against external economic data

---

## Current Status: Validation FAILED (Robustness Checks)

**v0.4.2.1** reveals that the initial v0.4.2 validation result was spurious. While the regression showed significant positive coefficients, robustness checks demonstrate the effect is driven by occupation-activity matrix structure, not the geometric properties of the activity space.

### What Happened

**v0.4.2** appeared to pass validation:
- DWA domain (2,087 activities) + Recipe Y (text embeddings)
- Regression of wage comovement on overlap: β > 0, t ≈ 5.17, p < 0.0001
- 5/5 σ values passed

**v0.4.2.1 audit revealed problems:**
- Identical t-stats across all σ values (suspicious)
- Overlap correlation r = 0.999 across σ values (σ is inert)
- Permutation test: p = 0.31 (effect not different from shuffled measures)
- **Placebo test: Random distances produce 2.5x stronger effect**

### Robustness Check Results

| Check | Status | Finding |
|-------|--------|---------|
| Distance Distribution | ✓ PASS | CV = 0.18, reasonable spread |
| Diagonal Dominance | ✓ PASS | t = 4.85 without diagonal |
| σ-Collinearity | ✗ FAIL | r = 0.999 across bandwidths |
| Permutation Test | ✗ FAIL | p = 0.31, effect not significant |
| Placebo Test | ✗ FAIL | Random distances work better (ratio = 0.4x) |
| Jackknife Stability | ✓ PASS | CV = 0.09, all positive |

**Bottom line:** The correlation between overlap and wage comovement exists but is NOT due to the semantic geometry of activities. Occupations that share activities have correlated wages regardless of how those activities are geometrically related. The kernel structure is irrelevant.

### Interpretation

The validation tested whether "occupations that share activities have correlated wages" — which is trivially true but not what the theory claims. The theory claims that **spillover through activity geometry** matters (occupations connected through similar activities, even if not identical). This was not validated.

**What this means for Phase II:** Do not proceed with shock propagation experiments until the validation methodology is reconsidered.

### Possible Paths Forward

1. **Different validation target**: Test spillover structure more directly (e.g., occupation transition probabilities)
2. **Activity-level outcomes**: Test on activity-level variation rather than occupation pairs
3. **Instrumental variation**: Use natural experiments that shock specific parts of the activity space
4. **Different measure construction**: Build occupation measures with less activity sharing

---

## Data Requirements

### O*NET Database (Required)
Download `db_30_0_excel.zip` from https://www.onetcenter.org/database.html and extract to `data/onet/db_30_0_excel/`.

### OES Wage Data (For Validation)
Download national OES files from https://www.bls.gov/oes/tables.htm for years 2019-2023. Extract to `data/external/oes/`. See `data/external/oes/README.md` for detailed instructions.

**Note:** BLS blocks automated downloads. You must download manually via browser.

---

## Repository Structure

```
paper/
    main.tex              # Theoretical framework and empirical strategy
    references.bib        # Bibliography
src/task_space/
    data.py               # O*NET file loading
    domain.py             # Activity domain and occupation measures (GWA + DWA)
    distances.py          # Recipe X (PCA) and Recipe Y (embeddings) distances
    kernel.py             # Kernel matrix and exposure computation
    diagnostics.py        # Phase I coherence checks and geometry comparison
    validation.py         # Phase I external validation
    crosswalk.py          # O*NET-SOC to OES crosswalk
data/
    onet/                 # O*NET database files (not in git)
    external/oes/         # OES wage data (not in git)
tests/                    # Test scripts
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

# Download O*NET database (required)
# Get db_30_0_excel.zip from https://www.onetcenter.org/database.html
# Extract to data/onet/db_30_0_excel/

# Run DWA pipeline test
PYTHONPATH=src python tests/test_dwa.py

# Run full validation (requires OES data)
PYTHONPATH=src python tests/run_dwa_validation.py
```

---

## Usage

### DWA + Recipe Y Pipeline (Recommended)

```python
from task_space import (
    build_dwa_activity_domain,
    build_dwa_occupation_measures,
    compute_text_embedding_distances,
    build_kernel_matrix,
    compute_occupation_exposure,
    create_shock_profile,
    distance_percentiles,
)

# Build the DWA manifold
domain = build_dwa_activity_domain()       # 2,087 DWAs
measures = build_dwa_occupation_measures() # 894 occupations × 2,087 activities

# Compute text embedding distances (Recipe Y)
titles = list(domain.activity_names.values())
distances = compute_text_embedding_distances(titles, domain.activity_ids)

# Set up kernel with median distance as bandwidth
sigma = distance_percentiles(distances)['p50']
kernel = build_kernel_matrix(distances, sigma=sigma)

# Create a shock profile and compute exposures
shock = create_shock_profile(
    activity_ids=distances.activity_ids,
    target_activities={'4.A.1.a.1.I01.D01': 1.0},  # Target a specific DWA
)
result = compute_occupation_exposure(measures, kernel, shock)

# result.exposures contains E_j for each occupation
```

### GWA + Recipe X Pipeline (Legacy, v0.4.1)

```python
from task_space import (
    build_activity_domain,
    build_occupation_measures,
    compute_activity_distances,
    build_kernel_matrix,
    compute_occupation_exposure,
    create_shock_profile,
    distance_percentiles,
)

# Build the GWA manifold (41 activities)
domain = build_activity_domain()
measures = build_occupation_measures()
distances = compute_activity_distances(measures)  # Recipe X

# Rest of pipeline same as above
```

### Diagnostics

```python
from task_space import (
    diagnose_dwa_sparsity,
    diagnose_measure_coherence,
    compare_geometries,
)

# Check DWA sparsity (recommended for DWA domain)
measures = build_dwa_occupation_measures()
sparsity = diagnose_dwa_sparsity(measures)
print(f"Median effective support: {sparsity.effective_support_percentiles['p50']}")
print(f"DWA coverage: {sparsity.dwa_coverage:.1%}")

# Compare Recipe X vs Recipe Y geometries
dist_x = compute_activity_distances(measures)
dist_y = compute_text_embedding_distances(titles, domain.activity_ids)
comparison = compare_geometries(dist_x, dist_y)
print(f"Spearman correlation: {comparison.spearman_r:.3f}")
print(f"Interpretation: {comparison.interpretation}")
```

---

## The Framework in Brief

1. **Activity Domain**: 2,087 DWAs form a metric space where distance encodes semantic similarity between activities.

2. **Occupation Measures**: Each occupation is a probability distribution over activities, constructed from O*NET task importance ratings aggregated to DWA level.

3. **Shock Propagation**: Technology shocks spread via exponential kernel: k(d) = exp(-d/σ).

4. **Exposure Measurement**: E_j = Σ_a ρ_j(a) × A(a), where A is the propagated shock field.

5. **Overlap**: Occupation-pair overlap O_{i,j} = ρ_i^T K ρ_j measures shared activity exposure.

See `paper/main.tex` Section 3 for formal definitions and Section 4 for empirical strategy.

---

## Validation Outputs

When validation runs, it produces:

- `outputs/phase_i_dwa/validation_results.json` — Full validation results for DWA + Recipe Y

---

## Version History

- **v0.4.2.1** (Current): Robustness audit reveals v0.4.2 result spurious. Validation: **FAIL** (placebo test).
- **v0.4.2**: DWA domain + Recipe Y (text embeddings). Initial validation appeared to pass (5/5 σ), but robustness checks failed.
- **v0.4.1**: GWA domain + Recipe X (rating-cooccurrence PCA). Validation: FAIL (0/5 σ values).
- **v0.4.0**: Core empirical pipeline with Phase I coherence diagnostics.
- **v0.3.7**: Theoretical framework complete.

---

## License

Research use only.
