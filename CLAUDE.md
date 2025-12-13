# CLAUDE.md - Developer Guide for Future Claudes

This document contains working conventions, version control rules, and quality-of-life information for AI assistants working on this project.

---

## Version Control

**Paper and codebase versions must always match.**

- Versions are bumped when entering a new phase of the research program
- Current: v0.4.2.1 (Phase I validation FAILED after robustness checks)
- Previous: v0.4.2 (appeared to pass, but robustness checks revealed spurious result)
- Previous: v0.4.1 (GWA + Recipe X validation failed)

When updating either paper or code, ensure the other stays in sync or is updated together.

---

## Current Status: Validation FAILED

**v0.4.2.1 Result:** The initial v0.4.2 validation result was spurious. Robustness checks revealed:

| Check | Status | Finding |
|-------|--------|---------|
| Distance Distribution | ✓ PASS | CV = 0.18, reasonable spread |
| Diagonal Dominance | ✓ PASS | t = 4.85 without diagonal |
| σ-Collinearity | ✗ FAIL | r = 0.999 across bandwidths |
| Permutation Test | ✗ FAIL | p = 0.31, not different from null |
| Placebo Test | ✗ FAIL | Random distances work 2.5x better |
| Jackknife Stability | ✓ PASS | CV = 0.09, all positive |

**Key finding:** The correlation between overlap and wage comovement is driven by occupation-activity matrix structure, NOT the semantic geometry of activities. Random distance matrices produce a stronger effect than the actual text embeddings.

**What this means:** The validation tested "occupations that share activities have correlated wages" (trivially true) rather than "spillover through activity geometry matters" (what the theory claims).

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

## O*NET Data Reference (v0.4.2)

This section documents O*NET database structure based on architect research (December 2025).

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
| DWA (Detailed Work Activities) | 2,087 | Derived via tasks | v0.4.2.1: FAILED (spurious) |

### Key Files

| File | Purpose | Notes |
|------|---------|-------|
| `Work Activities.xlsx` | GWA ratings by occupation | Direct Likert ratings |
| `DWA Reference.xlsx` | DWA hierarchy (GWA→IWA→DWA) | 2,087 DWAs |
| `Tasks to DWAs.xlsx` | Task-DWA mappings | ~23,000 linkages |
| `Task Ratings.xlsx` | Task importance ratings | Used to derive DWA importance |
| `Content Model Reference.xlsx` | GWA descriptions | For embeddings |

### DWA Matrix Construction (v0.4.2)

```python
# DWA importance derived via task linkages
# Per O*NET methodology: max task importance per DWA

task_ratings = load_task_ratings()  # Scale ID = 'IM'
tasks_to_dwas = load_tasks_to_dwas()
merged = tasks_to_dwas.merge(task_ratings, on=['O*NET-SOC Code', 'Task ID'])

# Aggregate: max importance per (occupation, DWA)
dwa_importance = merged.groupby(['O*NET-SOC Code', 'DWA ID'])['Data Value'].max()

# Pivot and normalize
# Result: 894 occupations × 2,087 DWAs
```

### O*NET-SOC Code Format

Format: `XX-XXXX.XX`
- First 7 chars (`XX-XXXX`): Standard SOC code (for OES matching)
- Suffix `.00`, `.01`, `.02`: O*NET subdivisions
- 894 occupations with full data

---

## OES Data Reference (v0.4.2)

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

Coverage statistics (v0.4.2):
- 894 O*NET occupations → 774 unique SOC codes
- 702 SOC codes usable for validation (present in both O*NET and comovement matrix)
- 246,051 occupation pairs in validation dataset

---

## Module Structure (v0.4.2)

```
src/task_space/
    __init__.py      # Exports all public APIs
    data.py          # O*NET file loading and filtering
    domain.py        # Activity domain + occupation measures (GWA and DWA)
    distances.py     # Recipe X (PCA) and Recipe Y (embeddings) distances
    kernel.py        # Kernel matrix, propagation, and exposure computation
    diagnostics.py   # Phase I coherence checks + geometry comparison
    validation.py    # Phase I external validation
    crosswalk.py     # O*NET-SOC to OES crosswalk
```

### New in v0.4.2

**domain.py:**
- `build_dwa_activity_domain()` — Build 2,087-activity DWA domain
- `build_dwa_occupation_measures()` — Build occupation × DWA matrix via task linkages

**distances.py:**
- `compute_text_embedding_distances()` — Recipe Y using sentence-transformers

**diagnostics.py:**
- `DWASparsityReport` — Dataclass for DWA sparsity statistics
- `GeometryComparison` — Dataclass for geometry comparison results
- `diagnose_dwa_sparsity()` — Compute DWA sparsity/coverage statistics
- `compare_geometries()` — Compare two distance matrices (Recipe X vs Y)

---

## Validation Results Reference (v0.4.2.1)

### Initial v0.4.2 Results (Appeared to Pass)

| σ Percentile | σ Value | β | SE | t | p-value |
|--------------|---------|------|------|------|---------|
| p10 | 0.559 | 441.25 | 85.33 | 5.17 | <0.0001 |
| p50 | 0.744 | 606.40 | 117.37 | 5.17 | <0.0001 |
| p90 | 0.896 | 742.13 | 143.59 | 5.17 | <0.0001 |

### v0.4.2.1 Robustness Checks (FAILED)

| Check | Result | Implication |
|-------|--------|-------------|
| σ-Collinearity | r = 0.999 | σ parameter is inert, only 1 effective specification |
| Permutation Test | p = 0.31 | Effect not different from shuffled measures |
| Placebo Test | ratio = 0.4x | Random distances work 2.5x better than semantic |

**Bottom Line:** The overlap-comovement correlation exists but is NOT due to activity geometry. It's driven by the occupation-activity matrix structure (occupations that share activities have correlated wages).

### Output Files

```
outputs/phase_i_dwa/
    validation_results.json   # Initial v0.4.2 results
    robustness_results.json   # v0.4.2.1 robustness checks
    audit_results.json        # Combined audit findings
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
spec_*.md                # Implementation specifications (may be in git)
```

---

## Lessons Learned (v0.4.2.1)

### From v0.4.1 (Inherited)

1. **BLS blocks automated downloads** — Must download OES data manually via browser.

2. **Pandas diff().dropna() drops all rows** — Use `.iloc[1:]` instead.

3. **Numpy booleans aren't JSON serializable** — Wrap in `bool()`.

4. **KernelMatrix attribute is `.matrix` not `.kernel_matrix`** — Check dataclass attributes.

### From v0.4.2

5. **DWA domain has incomplete coverage** — 7 DWAs have no task linkages. The `build_dwa_occupation_measures()` function adds all DWAs from reference to ensure alignment with `build_dwa_activity_domain()`.

6. **Sentence-transformers downloads models on first use** — First run downloads ~420 MB model weights to `~/.cache/torch/sentence_transformers/`. Document this for users.

7. **Cosine distance range is [0, 2]** — Not [0, 1]. Cosine distance = 1 - cosine_similarity, where similarity ∈ [-1, 1].

8. **Text embeddings produce very small overlaps** — With 2,087 activities, occupation measure weights are small (~1/2087 per activity). Overlap values are ~0.0005. This is expected and doesn't affect validation (coefficients adjust accordingly).

9. **Domain and measures must be built from same reference** — Ensure activity_ids align between domain, measures, and distances.

### From v0.4.2.1 (Critical)

10. **Identical t-stats across σ values is a red flag** — When t ≈ 5.17 for ALL 5 σ percentiles, this indicates σ is inert. Overlap correlation r > 0.99 confirms the kernel bandwidth has no effect.

11. **ALWAYS run permutation/placebo tests** — A significant coefficient means nothing if it survives when you (a) shuffle the occupation-activity structure, or (b) replace real distances with random ones.

12. **Overlap captures measure structure, not geometry** — The overlap O_ij = ρ_i^T K ρ_j is dominated by whether occupations share activities, not how those activities are geometrically related. This is the fundamental issue with the current validation approach.

13. **Random distances can produce stronger effects** — In the placebo test, random distance matrices gave β ≈ 1842 vs true β ≈ 728. The semantic structure is counterproductive. This suggests the validation target (wage comovement) may not be appropriate for testing geometric spillover.

14. **"Occupations that share activities have correlated wages" ≠ "Geometry matters"** — The validation inadvertently tested the former (trivially true) rather than the latter (what the theory claims). A better validation would test spillover effects between non-overlapping occupations.

---

## Recipe Comparison

| Aspect | Recipe X (v0.4.1) | Recipe Y (v0.4.2.1) |
|--------|-------------------|---------------------|
| Input | Occupation importance profiles | Activity titles/descriptions |
| Method | Transpose → PCA → Euclidean | Sentence encoder → Cosine |
| Dependencies | sklearn | sentence-transformers |
| Captures | Rating cooccurrence patterns | Semantic similarity |
| Initial Validation | FAIL (wrong sign) | Appeared to PASS |
| Robustness Checks | N/A | FAIL (spurious) |

**Conclusion:** Neither recipe validates the geometric spillover hypothesis. The overlap-comovement correlation is driven by occupation measure structure, not activity geometry.

---

## Updating Documentation

When making changes:

1. **Code changes** — Update `__init__.py` version comment if version bumps
2. **README.md** — Keep user-facing; update status, usage instructions, results
3. **CLAUDE.md** — Keep developer-facing; update conventions, roadmap, lessons learned
4. **Paper placeholders** — Fill with empirical outputs when available

If you discover something that would have helped you work faster, add it to this file.
