# CLAUDE.md - Developer Guide for Future Claudes

This document contains working conventions, version control rules, and quality-of-life information for AI assistants working on this project.

---

## Version Control

**Paper and codebase versions must always match.**

- Current: **v0.6.3.1** (Classification Infrastructure + Architecture Tests)
- Previous: v0.6.3 (Infrastructure Consolidation — Phase II ready)
- Previous: v0.6.2 (Phase 2 Complete — semantic signal validated)
- Previous: v0.6.1 (Kernel fix validated, continuous structure confirmed)
- Previous: v0.5.0 (Binary overlap validated, SAE marginal — **artifact**)
- Previous: v0.4.2.1 (random > semantic — **artifact, kernel collapse**)
- Previous: v0.4.2 (appeared to pass, spurious)
- Previous: v0.4.1 (GWA + Recipe X validation failed)

When updating either paper or code, ensure the other stays in sync or is updated together.

**Paper sync:** `paper/main.tex` is updated in chunks after empirical milestones. Current paper version: v0.6.2. Code may be ahead of paper. Canonical empirical results are in `outputs/` JSON files.

---

## Current Status: v0.6.3.1 (Classification Infrastructure)

**Phase I complete. Classification infrastructure for Phase II shock profiles.**

| Measure | t-stat (clustered) | R² | Status |
|---------|-------------------|-----|--------|
| Normalized kernel | 7.14 | 0.00485 | Primary |
| Unnormalized kernel | 5.90 | 0.00310 | Robustness |
| Binary Jaccard | 8.00 | 0.00167 | Baseline |

**What v0.6.3.1 added:**
- Activity classification infrastructure (`data/classifications.py`)
- GWA/DWA classification via O*NET element ID parsing (dot-separated, not fixed-width)
- Routine scores from Work Context 4.C.3.b.7
- Projected routine scores with ENDOGENEITY warning
- Updated shock profiles (`capability_v1`, `capability_v2`) using classifications
- Registry testing infrastructure (`_reset_registry()`, `_restore_default_shocks()`)
- Removed `sigma` parameter from `propagate_shock()` (K already encodes σ)
- Architecture tests (46 tests in unit/ + integration/)

**What Phase II will test:**
- Shock profile construction (v1, v2, v3)
- Retrospective evaluation (1990–2007)
- Prospective evaluation (2022–present)
- RBTC vs AI stability

---

## Methodological Note: Activity vs Occupation Level

**C1-C3 (text embeddings) and C4-C7 (O*NET structured) test different hypotheses:**

| Approach | Level | Method | Tests |
|----------|-------|--------|-------|
| C1-C3 | Activity → Occupation | Kernel-weighted overlap | Task-manifold theory |
| C4-C7 | Occupation → Occupation | Direct similarity | Ability/skill requirements |

C4-C7 bypass the activity-level kernel entirely. Higher R² for C4-C7 does not validate or invalidate the task-manifold theory — it tests whether occupation-level ability requirements predict wage comovement (different hypothesis).

---

## The v0.6.1 Fix

### Root Cause of v0.5.0 Bug

Kernel weights collapsed because:
1. σ calibrated to global distance distribution (median = 0.74)
2. With σ = 0.74, discrimination ratio was only 1.6x
3. Row-normalization over 2,087 activities made all weights ≈ 0.0005

### The Fix

```python
sigma = 0.2230  # Median of NEAREST-NEIGHBOR distances, not overall
normalize_kernel = False  # Skip row-normalization

K = np.exp(-dist_matrix / sigma)  # Unnormalized
overlap = occ_measures @ K @ occ_measures.T
```

### Critical Insight

**σ must be calibrated to local neighborhood structure (nearest-neighbor distances), not global distance distribution.**

| σ Source | Value | Discrimination | Status |
|----------|-------|----------------|--------|
| NN median | 0.22 | 4.5x | **Works** |
| Overall median | 0.74 | 1.6x | Collapsed |

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

## O*NET Data Reference

### Data Source

**Use downloaded database files, not API.** Download `db_30_0_excel.zip` from:
- https://www.onetcenter.org/database.html
- License: Creative Commons Attribution 4.0
- Version: 30.0 (current as of December 2025)

Place extracted files in `data/onet/db_30_0_excel/` directory.

### Activity Domain Options

| Domain | Dimension | Source | Status |
|--------|-----------|--------|--------|
| GWA (Generalized Work Activities) | 41 | Direct ratings | v0.4.1: FAILED |
| DWA (Detailed Work Activities) | 2,087 | Derived via tasks | v0.6.1: **Continuous validated** |

### Key Files

| File | Purpose | Notes |
|------|---------|-------|
| `Work Activities.xlsx` | GWA ratings by occupation | Direct Likert ratings |
| `DWA Reference.xlsx` | DWA hierarchy (GWA→IWA→DWA) | 2,087 DWAs |
| `Tasks to DWAs.xlsx` | Task-DWA mappings | ~23,000 linkages |
| `Task Ratings.xlsx` | Task importance ratings | Used to derive DWA importance |

### O*NET-SOC Code Format

Format: `XX-XXXX.XX`
- First 7 chars (`XX-XXXX`): Standard SOC code (for OES matching)
- Suffix `.00`, `.01`, `.02`: O*NET subdivisions
- 894 occupations with full data

---

## OES Data Reference

### Data Source

**BLS Occupational Employment and Wage Statistics**
- URL: https://www.bls.gov/oes/tables.htm
- Years used: 2019-2023 (5 years for wage comovement)
- **Note:** BLS blocks automated downloads. Must download manually via browser.

### Crosswalk: O*NET-SOC to OES

```python
def onet_to_soc(onet_code: str) -> str:
    """Strip .XX suffix: '15-1252.00' → '15-1252'"""
    return onet_code[:7]
```

Coverage statistics:
- 894 O*NET occupations → 774 unique SOC codes
- 702 SOC codes usable for validation
- 246,051 occupation pairs in validation dataset

---

## Architecture (v0.6.3.1)

### Package Structure

```
src/task_space/
    domain.py              # Activity domain, occupation measures
    data/                  # Data loading
        onet.py, oes.py, crosswalk.py, artifacts.py
        classifications.py # GWA/DWA classification, routine scores
    similarity/            # Similarity computation
        kernel.py, overlap.py, embeddings.py, distances.py
    shocks/                # Shock profiles (Phase II)
        registry.py, profiles.py, propagation.py
    validation/            # Validation utilities
        regression.py, diagnostics.py, permutation.py
    experiments/           # Experiment infrastructure
        config.py, runner.py

tests/
    unit/                  # Fast unit tests
    integration/           # Slower integration tests
    archive/               # Legacy scripts (not run by pytest)
```

### Key Concepts

**Overlap vs Exposure:**
- Overlap: O = ρ K ρ^T — pairwise occupation similarity (Phase I)
- Exposure: E = ρ (K I_t) — occupation exposure to shock (Phase II)

**Registry Pattern:**
- Shocks: `@register_shock("name")` decorator
- Add new shocks without modifying runner

**Artifact Store:**
- Cache: `.cache/artifacts/v1/`
- Use `get_embeddings()`, `get_distance_matrix()`
- Never compute embeddings elsewhere

---

## Validation Results Reference (v0.6.1)

### Primary Results (Clustered SEs — Correct Inference)

| Measure | t-stat (clustered) | R² | Notes |
|---------|-------------------|-----|-------|
| Normalized kernel | **7.14** | **0.00485** | Highest R² |
| Unnormalized kernel | 5.90 | 0.00310 | Robustness |
| Binary Jaccard | **8.00** | 0.00167 | Highest t-stat |

**Interpretation:** Binary Jaccard has the highest t-statistic (most precise estimate); normalized kernel has the highest R² (most explanatory power). Neither measure unambiguously dominates.

### Why t-stat and R² Can Diverge

The t-statistic measures **precision**: how tightly estimated is β relative to zero?
R² measures **explanatory power**: how much outcome variance does the regressor explain?

These can diverge when:
- The regressor has more variance → explains more outcome variance (higher R²)
- But the relationship is noisier or more heterogeneous → less precise estimate (lower t)

Binary Jaccard is a **coarser** measure (values cluster at 0 and small positive numbers). Kernel overlap is a **finer** measure (continuous values across the range). The finer measure picks up more signal and more noise.

**Interpretation:** The semantic structure is real and informative, but the mapping from "semantic similarity" to "economic similarity" is imperfect and heterogeneous across occupation pairs. Some semantically-similar occupation pairs comove strongly; others don't. The discrete measure is cruder but more uniformly predictive.

### Semantic vs Random (100 seeds, OLS for comparability)

*Note: This comparison uses OLS (unclustered) standard errors because random baselines were computed with OLS. These t-stats are NOT comparable to the clustered SE results above.*

| Metric | Semantic | Random Mean | Random Max | Percentile |
|--------|----------|-------------|------------|------------|
| t-stat (OLS) | **27.65** | 5.67 | 15.09 | **100%** |
| R² | **0.00310** | 0.00017 | 0.00052 | **100%** |

This confirms semantic structure is **not noise** — it beats random at the 100th percentile.

### Sigma Discrimination

| σ Source | Value | Discrimination | Status |
|----------|-------|----------------|--------|
| NN p10 | 0.127 | 14.2x | OK |
| NN p25 | 0.167 | 7.6x | OK |
| **NN median** | **0.223** | **4.5x** | **Best** |
| Overall median | 0.744 | 1.6x | Collapsed |

### Output Files

```
outputs/phase1/
    phase1_summary.md              # Initial diagnostic results
    phase1_addendum.md             # Fix documentation
    phase1_fix_results.json        # Kernel fix validation
    semantic_vs_random_fixed.json  # Critical confirmation
    distance_distribution.json     # Task 1.1.1
    similarity_orientation.json    # Task 1.1.2
    kernel_weights.json            # Task 1.1.3 (bug identified)
    jaccard_semantic_correlation.json  # Task 1.1.4
    activity_embeddings.npy        # MPNet embeddings
    jaccard_semantic_scatter.png   # Correlation plot

outputs/phase2/                    # Complete
    primary_validation.json        # Main comparison results
    permutation_tests.json         # Permutation p-values (all p < 0.001)
    cross_validation.json          # CV R² and overfit ratios
    phase2_robustness.json         # Normalized overlap + controls
    phase2_summary.md              # Human-readable summary
```

---

## File Conventions

```
paper/main.tex           # Source of truth for theory
paper/references.bib     # Bibliography
src/task_space/          # Implementation modules
data/onet/               # O*NET database files (not in git)
data/external/oes/       # OES wage data (not in git)
tests/                   # Test and validation scripts
outputs/                 # Generated outputs (not in git)
models/                  # Trained models (not in git)
spec_*.md                # Implementation specifications
```

---

## Lessons Learned

### From v0.4.x (Inherited)

1. **BLS blocks automated downloads** — Must download OES data manually via browser.

2. **Pandas diff().dropna() drops all rows** — Use `.iloc[1:]` instead.

3. **Numpy booleans aren't JSON serializable** — Wrap in `bool()`.

4. **Identical t-stats across σ values is a red flag** — Indicates σ is inert (or kernel collapsed).

5. **ALWAYS run permutation/placebo tests** — A significant coefficient means nothing if random data produces the same result.

### From v0.5.0 (Superseded by v0.6.1)

~~6. The labor market signal is discrete~~ — **WRONG.** This was an artifact of kernel collapse.

~~7. Dense embeddings add noise~~ — **WRONG.** Random > semantic was caused by kernel collapse, not semantic noise.

### From v0.6.1 (Current)

6. **σ must be calibrated to nearest-neighbor distances** — Global distance percentiles (median = 0.74) cause kernel collapse. Use NN median (≈ 0.22) instead.

7. **Row-normalization can destroy signal** — With 2,087 activities and near-uniform kernel weights, normalization washes out structure. Skip it.

8. **Discrimination ratio > 4x is necessary** — Check `exp(-d_p10/σ) / exp(-d_p90/σ)`. If < 2x, kernel is collapsed.

9. **Semantic structure IS predictive** — 100th percentile vs random, higher R² than Jaccard. However, Jaccard has higher t-stat under clustered inference. Both structures are informative; neither unambiguously dominates.

10. **Always diagnose before concluding** — The "discrete dominates" finding seemed robust but was an artifact. Run full diagnostics (distance distribution, orientation, kernel weights, correlation) before drawing conclusions.

11. **Unnormalized kernel overlap works best** — For validation, use raw `exp(-d/σ)` weights without row-normalization.

12. **Correlation r = 0.377 between Jaccard and semantic was diagnostic** — It told us the measures captured similar structure. The bug was in the kernel, not the embedding.

### Theoretical Implications (Revised)

13. **Both continuous and discrete structures are informative** — Continuous semantic similarity explains more variance (higher R²); discrete overlap provides more reliable inference (higher t-stat). Neither representation unambiguously dominates. The choice depends on whether you prioritize explanatory power or statistical reliability.

14. **Kernel bandwidth selection is critical** — The key methodological insight: calibrate to local structure (NN distances), not global distribution.

15. **Geometry enables prediction beyond counting** — +191% R² over binary Jaccard (normalized kernel). Semantic similarity adds explanatory power, but with noisier estimation.

### From v0.6.3 (Consolidation)

16. **Single source of truth** — Duplication caused v0.5.0 bug to persist. One implementation, tested.

17. **Explicit rejection of bad methods** — `calibrate_sigma()` throws error for non-NN methods, not silent fallback.

18. **Config-driven experiments** — New experiments = YAML config + registered function, not new script.

19. **Overlap ≠ Exposure** — Phase I validated overlap (O = ρKρ^T). Phase II computes exposure (E = ρKI).

### From v0.6.3.1 (Classification Infrastructure)

20. **Use dot-parsing for O*NET IDs** — Element IDs like `4.A.3.b.4` are hierarchical with variable-length segments. Use `.split('.')`, not fixed-width slicing.

21. **Projected routine scores are ENDOGENOUS** — `get_activity_projected_routine_scores()` computes "Task X is routine" from "Routine occupations do Task X." This is tautological for exposure regressions without controls.

22. **Classification is EXOGENOUS** — GWA categories (cognitive/physical/technical/interpersonal) are derived from O*NET hierarchy structure, independent of occupation characteristics.

23. **Remove redundant parameters** — `propagate_shock()` no longer takes `sigma` since the kernel matrix already encodes it. One source of truth.

24. **Test registry isolation requires module reload** — `importlib.reload(profiles)` is needed to re-register shocks after `_reset_registry()`.

---

## Recipe Comparison (Updated v0.6.3.1)

| Aspect | Recipe X (v0.4.1) | Recipe Y (v0.4.2) | Binary (v0.5.0) | Kernel (v0.6.1+) |
|--------|-------------------|-------------------|-----------------|------------------|
| Input | Importance profiles | Activity titles | Activity weights | Activity embeddings |
| Method | PCA → Euclidean | Sentence → Cosine | Binarize → Jaccard | Embed → Kernel → Overlap |
| σ | Various | Various | N/A | **NN median (0.22)** |
| Normalized | Yes | Yes | N/A | Yes (overlap), No (kernel) |
| t-stat (clustered) | — | — | **8.00** | 7.14 (norm), 5.90 (unnorm) |
| R² | — | — | 0.00167 | **0.00485** (norm) |
| Validation | FAIL (wrong sign) | FAIL (spurious) | PASS | PASS |
| Recommended | No | No | Baseline | Primary (norm) |

**Note:** Binary Jaccard has higher t-stat; normalized kernel has higher R². Both are valid; use based on whether precision or explanatory power is prioritized.

---

## Updating Documentation

When making changes:

1. **Code changes** — Update `__init__.py` version comment if version bumps
2. **README.md** — Keep user-facing; update status, usage, results
3. **CLAUDE.md** — Keep developer-facing; update conventions, lessons learned
4. **Paper placeholders** — Fill with empirical outputs when available

If you discover something that would have helped you work faster, add it to this file.
