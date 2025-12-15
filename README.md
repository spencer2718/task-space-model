# Task Space Model

A geometric framework for measuring labor market exposure to technological shocks.

**Version 0.6.6.0** — Validation Complete

---

## What This Is

This project develops a measurement framework for studying how technological change affects labor markets. The key idea: occupations are probability distributions over work activities. When automation affects certain activities, impact propagates through shared structure.

The core theoretical contribution is the **semantic-institutional decomposition**: effective distance between occupations separates task-content similarity (can the worker do the job?) from credentialing barriers (is the worker allowed to do the job?).

See `paper/main.tex` for formal theory and specifications.

---

## Current Status

### Validation Summary

| Test | Result | Status |
|------|--------|--------|
| CPS Mobility (Symmetric) | α=2.99, β=0.22, both p<0.001 | ✓ Validated |
| CPS Mobility (Asymmetric) | β_up ≈ β_down (ratio 1.04) | ⚠️ Interpretable null |
| Wage Comovement | Kernel R²=0.0052 vs Jaccard R²=0.0017 | ✓ Geometry informative |
| Employment Prediction | Semantic ΔR²=2.2% over RTI, p=0.07 | ⚠️ Marginal |

### CPS Worker Mobility (✓ Validated)

Conditional logit model of occupation destination choice using 89,329 verified CPS transitions.

| Component | Coefficient | t-stat | Interpretation |
|-----------|-------------|--------|----------------|
| α (semantic distance) | 2.994 | 98.5 | Workers prefer task-similar destinations |
| β (institutional distance) | 0.215 | 63.4 | Workers avoid credential barriers |

Both components independently predictive (r = 0.36 between measures). Framework succeeds at measuring task similarity for mobility analysis.

### Asymmetric Barriers Test (⚠️ Interpretable Null)

Tested whether credentials act as "one-way gates" (β_up > β_down). Result: barriers appear symmetric (ratio 1.04, p=0.0375). Credential-gate hypothesis not supported; theoretically valuable null finding. See paper Section 4.3.6 for interpretation.

### Employment Prediction (⚠️ Marginal)

Incremental validity test: Does semantic exposure predict 2019-2024 employment changes beyond canonical automation indices?

| Model | R² | Interpretation |
|-------|-----|----------------|
| RTI only (Acemoglu-Autor) | 9.8% | Routine occupations lost employment |
| RTI + Semantic | 12.0% | Semantic adds ΔR²=2.2% (p=0.07) |
| Full (RTI + AIOE + Semantic + controls) | 12.9% | Only RTI significant |

**Interpretation:** Framework succeeds at task similarity measurement but shows only marginal improvement for predicting automation-driven employment changes beyond canonical RTI measures.

### Wage Comovement

| Measure | t-stat | R² | Notes |
|---------|--------|-----|-------|
| Normalized kernel | 7.14 | 0.00485 | +191% R² vs binary |
| Binary Jaccard | 8.00 | 0.00167 | Higher precision |

Small absolute R² values—useful for discriminating between representations, not explaining wage dynamics.

---

## Quick Start

```bash
# Install
pip install -e ".[dev,notebooks]"

# Run tests
pytest tests/unit tests/integration -v

# Mobility module specifically
pytest tests/unit/mobility -v
```

---

## Data Requirements

### O*NET Database (Required)
Download `db_30_0_excel.zip` from https://www.onetcenter.org/database.html.
Extract to `data/onet/db_30_0_excel/`.

### OES Wage Data (For Wage Comovement)
Download national OES files from https://www.bls.gov/oes/tables.htm for 2019–2023.
Extract to `data/external/oes/`.

**Note:** BLS blocks automated downloads. Download manually via browser.

---

## Usage

```python
from task_space.mobility import (
    build_institutional_distance_matrix,
    load_canonical_results,
)

# Build institutional distance matrix
d_inst = build_institutional_distance_matrix()
print(f"Occupations: {d_inst.n_occupations}")
print(f"Cert coverage: {d_inst.cert_coverage:.1%}")
print(f"Assumptions: {d_inst.assumptions[0]}")

# Load canonical CPS mobility results
results = load_canonical_results()
print(f"α (semantic) = {results.alpha:.3f} (t = {results.alpha_t:.1f})")
print(f"β (institutional) = {results.beta:.3f} (t = {results.beta_t:.1f})")
```

---

## Repository Structure

```
src/task_space/          # Core implementation
    data/                # Data loading, classifications
    similarity/          # Kernel, overlap, embeddings
    shocks/              # Shock profiles
    validation/          # Regression, diagnostics
    mobility/            # CPS mobility validation (symmetric + asymmetric)
    _legacy/             # Deprecated modules (v0.6.6)

tests/
    unit/                # Fast unit tests
    unit/mobility/       # Mobility module tests
    integration/         # Slower integration tests
    archive/             # Historical research scripts

paper/
    main.tex             # Theory + specifications
    references.bib       # Bibliography

outputs/
    canonical/           # Paper-ready results (immutable)
    experiments/         # Versioned experiment outputs
```

See `CLAUDE.md` for developer details.

---

## Version History

| Version | What Changed |
|---------|--------------|
| **v0.6.6.0** | Asymmetric barriers test (interpretable null). Codebase reorganized: canonical/ directory, _legacy/ modules |
| v0.6.5.3 | Full Acemoglu-Autor RTI implemented. Marginal semantic improvement (ΔR²=2.2%, p=0.07) |
| v0.6.5.1 | CPS mobility validation integrated; semantic-institutional decomposition confirmed |
| v0.6.3.2 | Retrospective battery redesign (1980–2005 canonical settings) |
| v0.6.3.1 | Classification infrastructure, architecture tests |
| v0.6.2 | Both structures informative; neither dominates |
| v0.6.1 | Kernel fix — σ calibrated to NN distances |

---

## License

Research use only.
