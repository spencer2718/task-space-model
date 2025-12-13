# Task Space Model

A geometric framework for measuring labor market exposure to technological shocks.

**Version 0.4.1**

---

## What This Is

This project develops a measurement framework for studying how technological change affects labor markets. The key idea: occupations are probability distributions over an underlying space of work activities. When automation affects certain activities, the impact propagates through the geometry of that space, affecting occupations proportionally to their overlap with the shock.

This approach allows for:
- Continuous gradients of exposure across occupations (vs binary "exposed/unexposed")
- Spillover effects between related activities
- Empirical validation against external economic data

---

## Current Status: External Validation Failed

**v0.4.1** implements Phase I external validation (Diagnostic B from Section 4.4 of the paper). The validation tests whether our constructed task geometry predicts real economic relationships between occupations.

### Validation Design

We test whether occupation pairs with higher **overlap** in our activity space also have higher **wage comovement** (correlation of year-over-year log wage changes). The regression specification:

```
WageComovement_{i,j} = α + β × Overlap_{i,j}(σ) + ε_{i,j}
```

Standard errors are clustered by origin occupation. We run this for 5 different kernel bandwidths (σ at the 10th, 25th, 50th, 75th, and 90th percentiles of pairwise activity distances).

**Pass criterion:** β > 0 with p < 0.10

### Results: FAIL

| σ Percentile | σ Value | β | SE | p-value | Status |
|--------------|---------|---|-----|---------|--------|
| p10 | 19.24 | -2.30 | 7.05 | 0.74 | FAIL |
| p25 | 25.59 | -2.83 | 8.88 | 0.75 | FAIL |
| p50 | 37.50 | -3.27 | 12.20 | 0.79 | FAIL |
| p75 | 48.41 | -3.54 | 15.22 | 0.82 | FAIL |
| p90 | 57.34 | -3.76 | 17.72 | 0.83 | FAIL |

**Key findings:**
- **All coefficients are negative** (wrong sign) — higher overlap is associated with *lower* wage comovement
- **R² ≈ 0** across all specifications — overlap explains essentially none of the variance
- **Monotonicity test fails** — Spearman ρ = -0.13 (p = 0.73) across decile bins
- **0 of 5 σ values pass** the validation criterion

### Interpretation

This is a **rejection diagnostic working as intended**. The GWA-based geometry constructed via Recipe X (rating-cooccurrence with PCA) does not capture the economic structure that generates wage comovement across occupations.

**What this means:**
1. The current 41-dimensional GWA space may be too coarse to capture meaningful activity relationships
2. The Recipe X distance construction (transpose → PCA → Euclidean) may not encode economically relevant similarity
3. Wage comovement may not be the right validation target (confounded by industry exposure, business cycles, etc.)

**What this does NOT mean:**
- The framework is wrong — only this specific geometry failed
- Geometric approaches cannot work — alternative constructions may succeed
- External validation is impossible — different targets may reveal structure

### Recommended Next Steps

Per Section 4.4 of the paper, when the baseline geometry fails validation:

1. **Try Recipe Y (text-embedding geometry)**: Embed GWA titles/descriptions using a sentence encoder, compute cosine distances. This captures semantic rather than rating-cooccurrence similarity.

2. **Try the DWA domain**: Use 2,087 Detailed Work Activities instead of 41 GWAs. Higher dimensional but may capture finer-grained activity relationships.

3. **Try alternative validation targets**:
   - **Worker mobility** (CPS/SIPP): Transition rates between occupations may better capture the "reallocation friction" that distances are meant to encode
   - **Skill transferability scores**: Direct measures of how easily workers move between occupations

4. **Investigate wage comovement structure**: The negative (though insignificant) coefficients suggest something systematic. Are high-overlap occupation pairs in different industries? Different business cycle sensitivities?

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
    domain.py             # Activity domain and occupation measures
    distances.py          # Recipe X activity distances
    kernel.py             # Kernel matrix and exposure computation
    diagnostics.py        # Phase I coherence checks
    validation.py         # Phase I external validation (NEW in v0.4.1)
    crosswalk.py          # O*NET-SOC to OES crosswalk (NEW in v0.4.1)
data/
    onet/                 # O*NET database files (not in git)
    external/oes/         # OES wage data (not in git)
tests/                    # Test scripts
outputs/phase_i/          # Validation outputs (not in git)
```

---

## Quick Start

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy pandas scipy openpyxl matplotlib

# Download O*NET database (required)
# Get db_30_0_excel.zip from https://www.onetcenter.org/database.html
# Extract to data/onet/db_30_0_excel/

# Run pipeline test
PYTHONPATH=src python tests/test_pipeline.py

# Run validation (requires OES data)
PYTHONPATH=src python -c "
from pathlib import Path
from task_space import (
    build_occupation_measures,
    load_overlap_grid,
    run_full_validation,
)
from task_space.crosswalk import (
    build_onet_oes_crosswalk,
    load_oes_panel,
    compute_wage_comovement,
)

measures = build_occupation_measures()
overlap_grid = load_overlap_grid(Path('outputs/phase_i'))
panel = load_oes_panel([2019, 2020, 2021, 2022, 2023], Path('data/external/oes'))
comovement = compute_wage_comovement(panel, min_years=3)
crosswalk = build_onet_oes_crosswalk(measures.occupation_codes, panel['OCC_CODE'].unique().tolist())

results = run_full_validation(overlap_grid, comovement, crosswalk, measures)
print(f'Overall: {results.overall_decision}')
print(f'Headline beta: {results.headline.beta:.4f} (p={results.headline.pvalue:.4f})')
"
```

---

## Usage

### Basic Pipeline

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

# Build the manifold
domain = build_activity_domain()
measures = build_occupation_measures()
distances = compute_activity_distances(measures)

# Set up kernel with median distance as bandwidth
sigma = distance_percentiles(distances)['p50']
kernel = build_kernel_matrix(distances, sigma=sigma)

# Create a shock profile and compute exposures
shock = create_shock_profile(
    activity_ids=distances.activity_ids,
    target_activities={'4.A.1.a.1': 1.0},  # Target "Getting Information"
)
result = compute_occupation_exposure(measures, kernel, shock)

# result.exposures contains E_j for each occupation
```

### Validation Pipeline

```python
from pathlib import Path
from task_space import (
    build_occupation_measures,
    compute_activity_distances,
    compute_overlap_grid,
    save_overlap_grid,
    build_validation_dataset,
    run_full_validation,
    check_monotonicity,
    plot_monotonicity,
    save_validation_results,
)
from task_space.crosswalk import (
    build_onet_oes_crosswalk,
    load_oes_panel,
    compute_wage_comovement,
)

# Build manifold
measures = build_occupation_measures()
distances = compute_activity_distances(measures)

# Compute overlap grid for all 5 sigma values
overlap_grid = compute_overlap_grid(measures, distances)
save_overlap_grid(overlap_grid, Path('outputs/phase_i'))

# Load external validation data
panel = load_oes_panel([2019, 2020, 2021, 2022, 2023], Path('data/external/oes'))
comovement = compute_wage_comovement(panel, min_years=3)

# Build crosswalk
crosswalk = build_onet_oes_crosswalk(
    measures.occupation_codes,
    panel['OCC_CODE'].unique().tolist()
)

# Run validation
results = run_full_validation(overlap_grid, comovement, crosswalk, measures)

# Check monotonicity at p50
dataset = build_validation_dataset(
    overlap_grid.results['p50'], comovement, crosswalk, measures
)
mono = check_monotonicity(dataset)
plot_monotonicity(mono, Path('outputs/phase_i/monotonicity_plot.png'))

# Save results
save_validation_results(results, mono, Path('outputs/phase_i'))
```

---

## The Framework in Brief

1. **Activity Domain**: 41 GWAs form a metric space where distance encodes reallocation friction.

2. **Occupation Measures**: Each occupation is a probability distribution over activities, constructed from O*NET Importance ratings.

3. **Shock Propagation**: Technology shocks spread via exponential kernel: k(d) = exp(-d/σ).

4. **Exposure Measurement**: E_j = Σ_a ρ_j(a) × A(a), where A is the propagated shock field.

5. **Overlap**: Occupation-pair overlap O_{i,j} = ρ_i^T K ρ_j measures shared activity exposure.

See `paper/main.tex` Section 3 for formal definitions and Section 4 for empirical strategy.

---

## Validation Outputs

When validation runs, it produces:

- `outputs/phase_i/overlap_p{10,25,50,75,90}.{npz,json}` — Overlap matrices for each σ
- `outputs/phase_i/overlap_stats.json` — Overlap distribution statistics
- `outputs/phase_i/regression_results.json` — Full validation regression results
- `outputs/phase_i/monotonicity_plot.png` — Binned overlap vs wage comovement

---

## Version History

- **v0.4.1** (Current): Phase I external validation. Result: FAIL. GWA-based geometry does not predict wage comovement.
- **v0.4.0**: Core empirical pipeline with Recipe X distances and Phase I coherence diagnostics.
- **v0.3.7**: Theoretical framework complete.

---

## License

Research use only.
