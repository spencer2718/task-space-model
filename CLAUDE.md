# CLAUDE.md - Developer Guide for Future Claudes

This document contains working conventions, version control rules, and quality-of-life information for AI assistants working on this project.

---

## Version Control

**Paper and codebase versions must always match.**

- Versions are bumped when entering a new phase of the research program
- Current: v0.4.1 (Phase I external validation complete — FAIL result)
- Previous: v0.4.0 (core empirical pipeline)

When updating either paper or code, ensure the other stays in sync or is updated together.

---

## Current Status: Validation Failed

**v0.4.1 Result:** The GWA-based geometry (Recipe X) failed external validation against wage comovement.

- All 5 σ values show negative (wrong sign) coefficients
- R² ≈ 0, no predictive power
- Monotonicity test fails (Spearman ρ = -0.13)

**Next steps per Section 4.4:**
1. Try Recipe Y (text-embedding geometry)
2. Try DWA domain (2,087 activities instead of 41)
3. Try alternative validation targets (worker mobility from CPS/SIPP)

See README.md for full results and interpretation.

---

## Referencing the Paper

**Reference definitions by name, not number.** Definition numbers may change as sections reorganize.

Good:
```python
# Implements the normalized spillover operator (Definition: Spillover Operator)
```

Bad:
```python
# Implements Definition 3.5
```

Key definitions to reference by name:
- Task domain
- Occupation measures
- Technological state / Shock profile
- Normalized spillover operator
- Baseline exposure construction
- Occupation-level exposure functionals
- Reduced-form outcome equations

---

## Before Big Implementations: Use the Architect Agent

**Important:** Spencer has a software architect research agent that does deep research on both economics and system design.

Before implementing major components, request that the architect agent be consulted to:
- Plan the implementation strategy
- Verify economic assumptions against literature
- Identify potential issues with data structure
- Design the module architecture

Do not proceed with large implementations without this planning step.

---

## O*NET Data Reference (v0.4)

This section documents O*NET database structure based on architect research (December 2025).

### Data Source

**Use downloaded database files, not API.** Download `db_30_0_excel.zip` from:
- https://www.onetcenter.org/database.html
- License: Creative Commons Attribution 4.0
- Version: 30.0 (current as of December 2025)

Place extracted files in `data/onet/` directory.

### Activity Domain Options

| Domain | Dimension | Source | Status |
|--------|-----------|--------|--------|
| GWA (Generalized Work Activities) | 41 | Direct ratings | v0.4.1: FAILED validation |
| DWA (Detailed Work Activities) | 2,087 | Derived via tasks | Recommended next |

### Key Files

| File | Purpose | Rows |
|------|---------|------|
| `Work Activities.xlsx` | GWA ratings by occupation | ~79,376 |
| `Tasks to DWAs.xlsx` | Task-DWA mappings | ~23,000 |
| `Task Ratings.xlsx` | Task importance ratings | varies |
| `DWA Reference.xlsx` | DWA hierarchy (GWA→IWA→DWA) | 2,087 |
| `Content Model Reference.xlsx` | GWA descriptions for embeddings | ~580 |

### Work Activities.xlsx Schema

| Column | Type | Notes |
|--------|------|-------|
| O*NET-SOC Code | Char(10) | Format: `XX-XXXX.XX` |
| Element ID | Varchar(20) | GWA ID (e.g., `4.A.1.a.1`) |
| Scale ID | Varchar(3) | `IM` = Importance, `LV` = Level |
| Data Value | Float | Raw rating |
| N | Integer | Sample size |
| Standard Error | Float | SEM |
| Recommend Suppress | Char(1) | `Y` or `N` |
| Not Relevant | Char(1) | `Y` or `N` (Level only) |

### Rating Scales

**Importance (Scale ID = `IM`):**
- Range: 1-5
- Anchors: 1=Not Important, 2=Somewhat, 3=Important, 4=Very, 5=Extremely
- Normalize: `(value - 1) / 4` → [0, 1]

**Level (Scale ID = `LV`):**
- Range: 0-7
- Level=0 when Importance=1
- Normalize: `value / 7` → [0, 1]

### GWA Matrix Construction

```python
# Filter to Importance scale, non-suppressed
df = work_activities[
    (work_activities['Scale ID'] == 'IM') &
    (work_activities['Recommend Suppress'] == 'N')
]

# Pivot to occupation × GWA matrix
matrix = df.pivot_table(
    index='O*NET-SOC Code',
    columns='Element ID',
    values='Data Value'
)

# Normalize to [0,1]
matrix = (matrix - 1) / 4

# Result: 894 occupations × 41 GWAs
```

### O*NET-SOC Code Format

Format: `XX-XXXX.XX`
- First 7 chars (`XX-XXXX`): Standard SOC code (for OES matching)
- Suffix `.00`, `.01`, `.02`: O*NET subdivisions
- 894 occupations with full GWA data

---

## OES Data Reference (v0.4.1)

### Data Source

**BLS Occupational Employment and Wage Statistics**
- URL: https://www.bls.gov/oes/tables.htm
- Years used: 2019-2023 (5 years for wage comovement)
- **Note:** BLS blocks automated downloads. Must download manually via browser.

### File Structure

```
data/external/oes/
├── oesm19nat/
│   └── national_M2019_dl.xlsx
├── oesm20nat/
│   └── national_M2020_dl.xlsx
...
```

### Crosswalk: O*NET-SOC to OES

```python
def onet_to_soc(onet_code: str) -> str:
    """Strip .XX suffix: '15-1252.00' → '15-1252'"""
    return onet_code[:7]
```

Coverage statistics (v0.4.1):
- 894 O*NET occupations → 774 unique SOC codes
- 747 SOC codes matched in OES (96.4% coverage)
- 702 SOC codes usable for validation (present in both O*NET and comovement matrix)

---

## Module Structure (v0.4.1)

```
src/task_space/
    __init__.py      # Exports all public APIs
    data.py          # O*NET file loading and filtering
    domain.py        # Activity domain + occupation measure construction
    distances.py     # Activity distance computation (Recipe X)
    kernel.py        # Kernel matrix, propagation, and exposure computation
    diagnostics.py   # Phase I coherence checks
    validation.py    # Phase I external validation (NEW)
    crosswalk.py     # O*NET-SOC to OES crosswalk (NEW)
```

### New in v0.4.1

**validation.py:**
- `OverlapResult`, `OverlapGrid` — Overlap computation results
- `ValidationDataset` — Pair-level dataset for regression
- `RegressionResult`, `ValidationResults` — Regression outputs
- `MonotonicityResult` — Binned overlap-outcome relationship
- `compute_overlap_grid()` — Compute overlaps for all 5 σ values
- `build_validation_dataset()` — Merge overlap with wage comovement
- `run_validation_regression()` — OLS with cluster-robust SEs
- `run_full_validation()` — Run for all 5 σ values
- `check_monotonicity()` — Bin overlap and test monotonicity
- `plot_monotonicity()` — Generate binned scatterplot

**crosswalk.py:**
- `OnetOesCrosswalk` — Crosswalk with coverage statistics
- `WageComovement` — Pairwise wage correlation matrix
- `onet_to_soc()` — Code conversion
- `load_oes_year()`, `load_oes_panel()` — OES data loading
- `compute_wage_comovement()` — Log wage change correlations
- `aggregate_occupation_measures()` — Average O*NET measures to SOC level

---

## Validation Results Reference (v0.4.1)

### Headline Numbers

| σ Percentile | β | SE | p-value | Passes |
|--------------|---|-----|---------|--------|
| p10 | -2.30 | 7.05 | 0.74 | No |
| p25 | -2.83 | 8.88 | 0.75 | No |
| p50 | -3.27 | 12.20 | 0.79 | No |
| p75 | -3.54 | 15.22 | 0.82 | No |
| p90 | -3.76 | 17.72 | 0.83 | No |

- Dataset: 246,051 occupation pairs, 702 occupations
- Clusters: 701 (origin occupation)
- Monotonicity: Spearman ρ = -0.13 (p = 0.73)

### Output Files

```
outputs/phase_i/
    overlap_p{10,25,50,75,90}.npz   # Overlap matrices
    overlap_p{10,25,50,75,90}.json  # Overlap metadata
    overlap_stats.json               # Distribution statistics
    regression_results.json          # Full validation results
    monotonicity_plot.png            # Binned scatterplot
```

---

## File Conventions

```
paper/main.tex           # Source of truth for theory and empirical strategy
paper/references.bib     # Bibliography (BibTeX)
src/task_space/          # Implementation modules
data/onet/               # O*NET database files (not in git)
data/external/oes/       # OES wage data (not in git)
tests/                   # Test scripts
outputs/                 # Generated figures and tables (not in git)
```

---

## Lessons Learned (v0.4.1)

1. **BLS blocks automated downloads** — Must download OES data manually via browser. Document this clearly for future users.

2. **Pandas diff().dropna() drops all rows** — When computing year-over-year wage changes, use `.iloc[1:]` instead of `.dropna()` to avoid dropping all rows due to ANY column having NaN.

3. **Numpy booleans aren't JSON serializable** — Wrap in `bool()` when saving to JSON.

4. **KernelMatrix attribute is `.matrix` not `.kernel_matrix`** — Check dataclass attributes when reusing objects.

5. **Negative coefficients are informative** — The validation didn't just fail to find a relationship; it found the wrong sign. This suggests systematic structure worth investigating (e.g., high-overlap pairs may be in different industries with different cycle sensitivities).

---

## Updating Documentation

When making changes:

1. **Code changes** — Update `__init__.py` version comment if version bumps
2. **README.md** — Keep user-facing; update status, usage instructions, results
3. **CLAUDE.md** — Keep developer-facing; update conventions, roadmap, lessons learned
4. **Paper placeholders** — Fill with empirical outputs when available

If you discover something that would have helped you work faster, add it to this file.
