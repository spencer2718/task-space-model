# CLAUDE.md - Developer Guide for Future Claudes

This document contains working conventions, version control rules, and quality-of-life information for AI assistants working on this project.

---

## Version Control

**Paper and codebase versions must always match.**

- Current: **v0.5.0** (Binary overlap validated, SAE marginal improvement)
- Previous: v0.4.2.1 (robustness audit revealed v0.4.2 was spurious)
- Previous: v0.4.2 (appeared to pass, but random distances outperformed)
- Previous: v0.4.1 (GWA + Recipe X validation failed)

When updating either paper or code, ensure the other stays in sync or is updated together.

---

## Current Status: Binary Overlap Validated

**v0.5.0 Result:** The labor market signal is **discrete, not continuous**.

| Phase | What We Tested | Result |
|-------|----------------|--------|
| A | Binary Jaccard overlap | **PASS** — β = 0.471, t = 8.00 |
| B | SAE training (768 → 16384) | Converged, L0 ≈ 28 |
| C | Feature interpretability | **PASS** — 17/20 coherent |
| D | SAE vs Binary comparison | **SAE marginal** — +11.5% β (needed +20%) |

**Key finding:** Occupations that share activities have correlated wages. Semantic similarity between activities adds little beyond binary counting.

**Recommendation:** Use Binary Jaccard for Phase II. SAE is available for robustness checks.

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
| DWA (Detailed Work Activities) | 2,087 | Derived via tasks | v0.5.0: Binary validated |

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

## Module Structure (v0.5.0)

```
src/task_space/
    __init__.py      # Exports all public APIs
    data.py          # O*NET file loading and filtering
    domain.py        # Activity domain + occupation measures
    distances.py     # Recipe X (PCA) and Recipe Y (embeddings)
    kernel.py        # Kernel matrix, propagation, exposure
    baseline.py      # Binary Jaccard overlap (NEW in v0.5.0)
    sae.py           # Sparse Autoencoder (NEW in v0.5.0)
    diagnostics.py   # Phase I coherence checks
    validation.py    # Phase I external validation
    crosswalk.py     # O*NET-SOC to OES crosswalk
```

### New in v0.5.0

**baseline.py:**
- `compute_binary_overlap()` — Binary Jaccard overlap
- `run_baseline_regression()` — Validation with clustered SEs
- `save_baseline_results()` — Output to JSON

**sae.py:**
- `SparseAutoencoder` — 768 → 16384 → 768 architecture
- `train_sae()` — Training with L1 penalty and adaptive λ
- `extract_sparse_features()` — Feature extraction with thresholding

---

## Validation Results Reference (v0.5.0)

### Phase A: Binary Baseline

| Metric | Value |
|--------|-------|
| β | 0.471 |
| SE | 0.059 |
| t | 8.00 |
| p | < 10⁻¹⁵ |
| R² | 0.00167 |
| n_pairs | 246,051 |

### Phase D: Binary vs SAE

| Metric | Binary | SAE | Δ | Target | Status |
|--------|--------|-----|---|--------|--------|
| β | 0.471 | 0.526 | +11.5% | >20% | ✗ FAIL |
| R² | 0.00167 | 0.00292 | +75% | >30% | ✓ PASS |
| Correlation | — | r = 0.824 | — | <0.9 | ✓ OK |

### SAE Training Metrics

| Metric | Value |
|--------|-------|
| Training time | 20.4 minutes (CPU) |
| Final L0 | 28.1 (target: 10-20) |
| Dead features | 12,603 / 16,384 (77%) |
| Coherent features | 17/20 inspected |

### Output Files

```
outputs/phase_a/
    baseline_results.json     # Binary validation results
    binary_overlap.npy        # Overlap matrix

outputs/phase_b/
    dwa_embeddings.npy        # MPNet embeddings (2087 × 768)
    dwa_sparse_features.npy   # SAE features (2087 × 16384)
    phase_b_summary.json      # Training summary

outputs/phase_c/
    feature_inspection.json   # Top 20 features analysis
    feature_interpretability.txt  # Human-readable audit

outputs/phase_d/
    validation_comparison.json    # Binary vs SAE results
    sae_overlap.npy              # SAE overlap matrix
    occupation_features.npy       # Aggregated occupation features

models/
    sae_v1.pt                    # Trained SAE checkpoint
    sae_v1_training_log.json     # Training metrics
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

4. **Identical t-stats across σ values is a red flag** — Indicates σ is inert.

5. **ALWAYS run permutation/placebo tests** — A significant coefficient means nothing if random data produces the same result.

### From v0.5.0 (Current)

6. **The labor market signal is discrete** — Binary activity overlap captures most of the predictive power. Continuous semantic similarity adds noise (v0.4.2.1) or marginal value (v0.5.0 SAE).

7. **SAE finds coherent structure but it's fine-grained** — 17/20 features were interpretable, but activation was uniform across ~2,066 effective features. No dominant cross-cutting dimensions.

8. **R² can improve even when β improvement is modest** — SAE R² nearly doubled (+75%) while β only improved +11.5%. This suggests SAE captures signal more efficiently but the magnitude is similar.

9. **Correlation between measures is diagnostic** — r(Binary, SAE) = 0.824 indicates moderate overlap. If r > 0.9, the measures are redundant. If r < 0.7, they capture different structure.

10. **Training SAE on small data (2,087 samples) works** — ~20 minutes on CPU, converges well. The bottleneck is conceptual (what structure to find), not computational.

11. **Feature coherence ≠ economic value** — Features can be semantically interpretable ("software", "healthcare", "construction") without adding predictive power beyond binary counting.

12. **Wage comovement may not be the right validation target** — It might be driven by industry/geographic shocks that don't respect activity geometry. Worker mobility could show more geometric structure.

### Theoretical Implications

13. **The manifold abstraction may be wrong** — Activities behave more like discrete tokens than points on a smooth manifold. Kernel-weighted overlap reduces to counting shared nodes.

14. **Dense embeddings add noise to economic signals** — Random distances outperformed MPNet in v0.4.2.1. The semantic structure of language doesn't align with economic substitutability.

15. **Supervised probing may be the path forward** — If we have theoretical priors (routine/cognitive, automatable, etc.), extracting those dimensions directly from a large language model might capture economic structure that unsupervised methods miss.

---

## Recipe Comparison

| Aspect | Recipe X (v0.4.1) | Recipe Y (v0.4.2) | Binary (v0.5.0) | SAE (v0.5.0) |
|--------|-------------------|-------------------|-----------------|--------------|
| Input | Importance profiles | Activity titles | Activity weights | MPNet embeddings |
| Method | PCA → Euclidean | Sentence encoder → Cosine | Binarize → Jaccard | Encode → Binarize → Jaccard |
| Validation | FAIL (wrong sign) | FAIL (spurious) | **PASS** (t=8.00) | Marginal (+11.5% β) |
| Complexity | Medium | Medium | **Low** | High |
| Recommended | No | No | **Yes** | For robustness |

---

## Updating Documentation

When making changes:

1. **Code changes** — Update `__init__.py` version comment if version bumps
2. **README.md** — Keep user-facing; update status, usage, results
3. **CLAUDE.md** — Keep developer-facing; update conventions, lessons learned
4. **Paper placeholders** — Fill with empirical outputs when available

If you discover something that would have helped you work faster, add it to this file.
