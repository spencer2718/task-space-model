# Task Space Model

A geometric framework for measuring labor market exposure to technological shocks.

**Version 0.6.3.2** — Retrospective Battery Redesign

---

## What This Is

This project develops a measurement framework for studying how technological change affects labor markets. The key idea: occupations are probability distributions over work activities. When automation affects certain activities, impact propagates through shared structure.

See `paper/main.tex` for formal theory and specifications.

---

## Current Status

### Phase I: Complete

Both continuous and discrete representations predict wage comovement; neither dominates.

| Measure | t-stat (clustered) | R² | Notes |
|---------|-------------------|-----|-------|
| Normalized kernel | 7.14 | **0.00485** | +191% R² vs binary |
| Binary Jaccard | **8.00** | 0.00167 | Higher precision |

### Phase II: In Progress

**Retrospective diagnostic battery** (1980–2005) tests when continuous structure adds value:

| Test | Setting | Period | Primary? |
|------|---------|--------|----------|
| A | ALM task composition shifts | 1980–2000 | |
| B | Autor-Dorn employment reallocation | 1980–2005 | **Yes** |
| C | Acemoglu-Restrepo robot displacement | 1990–2007 | |

**Prospective AI evaluation** (2022–present) applies framework to generative AI.

---

## Quick Start

```bash
# Install
pip install -e ".[dev,notebooks]"

# Run tests
pytest tests/unit tests/integration -v
```

---

## Data Requirements

### O*NET Database (Required)
Download `db_30_0_excel.zip` from https://www.onetcenter.org/database.html.
Extract to `data/onet/db_30_0_excel/`.

### OES Wage Data (For Validation)
Download national OES files from https://www.bls.gov/oes/tables.htm for 2019–2023.
Extract to `data/external/oes/`.

**Note:** BLS blocks automated downloads. Download manually via browser.

---

## Usage

```python
from task_space import build_dwa_occupation_measures, compute_recipe_y_distances
from task_space.diagnostics_v061 import compute_kernel_overlap

# Load occupation measures
measures = build_dwa_occupation_measures()

# Compute activity distances (MPNet embeddings)
dist_matrix = compute_recipe_y_distances(measures.activity_titles)

# Compute kernel overlap with proper σ calibration
overlap, sigma = compute_kernel_overlap(
    measures.raw_matrix,
    dist_matrix,
    sigma=None,  # Auto-calibrate to NN median
    normalize=False
)
```

---

## Repository Structure

```
src/task_space/          # Core implementation
    data/                # Data loading, classifications
    similarity/          # Kernel, overlap, embeddings
    shocks/              # Shock profiles (Phase II)
    validation/          # Regression, diagnostics

tests/
    unit/                # Fast unit tests
    integration/         # Slower integration tests

paper/
    main.tex             # Theory + specifications
    references.bib       # Bibliography

outputs/                 # Empirical results (JSON)
```

See `CLAUDE.md` for developer details.

---

## Version History

| Version | What Changed |
|---------|--------------|
| **v0.6.3.2** | Retrospective battery redesign (1980–2005 canonical settings) |
| v0.6.3.1 | Classification infrastructure, architecture tests |
| v0.6.3 | Infrastructure consolidation |
| v0.6.2 | Both structures informative; neither dominates |
| v0.6.1 | Kernel fix — σ calibrated to NN distances |

---

## License

Research use only.
