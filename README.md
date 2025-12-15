# Task Space Model

A geometric framework for measuring labor market exposure to technological shocks.

**Version 0.6.5.1** — CPS Mobility Validation

---

## What This Is

This project develops a measurement framework for studying how technological change affects labor markets. The key idea: occupations are probability distributions over work activities. When automation affects certain activities, impact propagates through shared structure.

The core theoretical contribution is the **semantic-institutional decomposition**: effective distance between occupations separates task-content similarity (can the worker do the job?) from credentialing barriers (is the worker allowed to do the job?).

See `paper/main.tex` for formal theory and specifications.

---

## Current Status

### Primary Validation: CPS Worker Mobility (Complete)

Conditional logit model of occupation destination choice using 89,329 verified CPS transitions.

| Component | Coefficient | t-stat | Interpretation |
|-----------|-------------|--------|----------------|
| α (semantic distance) | 2.994 | 98.5 | Workers prefer task-similar destinations |
| β (institutional distance) | 0.215 | 63.4 | Workers avoid credential barriers |

Both components independently predictive (r = 0.36 between measures). On a standardized basis, effects are comparable in magnitude. The key finding is that both semantic and institutional structure matter, not that one dominates.

### Complementary Validation: Wage Comovement

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
    mobility/            # CPS mobility validation

tests/
    unit/                # Fast unit tests
    unit/mobility/       # Mobility module tests
    integration/         # Slower integration tests

paper/
    main.tex             # Theory + specifications
    references.bib       # Bibliography

data/processed/mobility/ # Canonical results (parquet, JSON)
outputs/                 # Other empirical results
```

See `CLAUDE.md` for developer details.

---

## Version History

| Version | What Changed |
|---------|--------------|
| **v0.6.5.1** | CPS mobility validation integrated; semantic-institutional decomposition confirmed |
| v0.6.3.2 | Retrospective battery redesign (1980–2005 canonical settings) |
| v0.6.3.1 | Classification infrastructure, architecture tests |
| v0.6.2 | Both structures informative; neither dominates |
| v0.6.1 | Kernel fix — σ calibrated to NN distances |

---

## License

Research use only.
