# Task Space Model

A geometric framework for measuring labor market exposure to technological shocks.

**Version 0.4.2**

---

## What This Is

This project develops a measurement framework for studying how technological change affects labor markets. The key idea: occupations are probability distributions over an underlying space of work activities. When automation affects certain activities, the impact propagates through the geometry of that space, affecting occupations proportionally to their overlap with the shock.

This approach allows for:
- Continuous gradients of exposure across occupations (vs binary "exposed/unexposed")
- Spillover effects between related activities
- Empirical validation against external economic data

---

## Current Status: External Validation PASSED

**v0.4.2** implements the pre-committed pivot from Section 4.4 of the paper after v0.4.1 validation failed. The new geometry uses:

- **DWA domain**: 2,087 Detailed Work Activities (instead of 41 GWAs)
- **Recipe Y distances**: Text embeddings via sentence-transformers (instead of rating-cooccurrence PCA)

### Validation Design

We test whether occupation pairs with higher **overlap** in our activity space also have higher **wage comovement** (correlation of year-over-year log wage changes). The regression specification:

```
WageComovement_{i,j} = α + β × Overlap_{i,j}(σ) + ε_{i,j}
```

Standard errors are clustered by origin occupation. We run this for 5 different kernel bandwidths (σ at the 10th, 25th, 50th, 75th, and 90th percentiles of pairwise activity distances).

**Pass criterion:** β > 0 with p < 0.10

### Results: PASS (5/5)

| σ Percentile | σ Value | β | SE | t | p-value | Status |
|--------------|---------|---|-----|---|---------|--------|
| p10 | 0.559 | 441.25 | 85.33 | 5.17 | <0.0001 | PASS |
| p25 | 0.651 | 523.22 | 101.26 | 5.17 | <0.0001 | PASS |
| p50 | 0.744 | 606.40 | 117.37 | 5.17 | <0.0001 | PASS |
| p75 | 0.828 | 681.52 | 131.89 | 5.17 | <0.0001 | PASS |
| p90 | 0.896 | 742.13 | 143.59 | 5.17 | <0.0001 | PASS |

**Key findings:**
- **All coefficients are positive** — higher overlap is associated with higher wage comovement
- **Highly significant** — t ≈ 5.17, p < 0.0001 across all specifications
- **R² ≈ 0.15%** — small but typical for cross-sectional occupation pair data
- **5 of 5 σ values pass** the validation criterion

### Interpretation

The DWA + Recipe Y geometry captures economically meaningful structure. Occupation pairs that share more activity exposure (as measured by kernel-weighted overlap) exhibit higher wage comovement over business cycles.

**What changed from v0.4.1:**
1. **Domain granularity**: 41 GWAs → 2,087 DWAs provides finer resolution
2. **Distance metric**: Rating-cooccurrence PCA → text embeddings captures semantic similarity
3. **Result**: Coefficients flipped from negative (wrong sign) to strongly positive

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

- **v0.4.2** (Current): DWA domain + Recipe Y (text embeddings). Validation: **PASS** (5/5 σ values).
- **v0.4.1**: GWA domain + Recipe X (rating-cooccurrence PCA). Validation: FAIL (0/5 σ values).
- **v0.4.0**: Core empirical pipeline with Phase I coherence diagnostics.
- **v0.3.7**: Theoretical framework complete.

---

## License

Research use only.
