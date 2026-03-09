# Task-Space Oracle

Infrastructure for analyzing how task-level technological change propagates to labor market outcomes.

**Version 0.7.8** — Paper Alignment Complete

---

## Quick Navigation

| If you want to... | Start here |
|-------------------|------------|
| Understand the research | `paper/main.tex` (full theory) |
| Reproduce key tables | `scripts/reproduce_tables.py` |
| Understand distance metrics | `src/task_space/similarity/DISTANCE_GUIDE.md` |
| See current scientific state | `LEDGER.md` |
| Contribute as engineer | `CLAUDE.md` |

---

## What This Is

The **task-space oracle** is a modular architecture targeting:

```
(T, I, S, M) → (Δρ, ΔL, ΔW)
```

From task representation, institutional structure, shock profiles, and adjustment mechanisms to occupation-specific changes in task distributions, employment, and wages.

**Core insight:** Technology acts on tasks. Occupations are probability distributions over tasks. When technology changes task productivity, occupation task distributions shift; employment and wages adjust as downstream aggregations.

This is a research program, not a finished model. The paper establishes that the architecture is viable and validates initial module specifications.

See `paper/main.tex` for formal theory and specifications.

---

## Module Status

| Module | Status | Evidence |
|--------|--------|----------|
| **T** (Embedding-informed distance) | ✓ Validated | Centroid pseudo-R² = 14.1%; 74.9% improvement over O*NET baselines |
| **I** (Institutional distance) | ✓ Validated | β = 0.139, t = 33.7 conditional on T |
| **S** (Shock integration) | ✓ Validated | ΔLL = +23,119 on out-of-period comparison |
| **M** (Switching costs) | ⚠️ Calibrated | External anchor (Dix-Carneiro 2014) |
| Pathway ranking | ✓ Validated | MPR = 0.74; per-origin ρ ≈ 0.13 |
| Demand decomposition | ✓ Quantified | Demand ρ = 0.80; geometry ρ = 0.04 |
| Structural stability | ✓ Validated | Aggregate Δα < 1% (p = 0.72); telework heterogeneity δ₄ = −0.086 (p = 0.01) |

**Scope:** The framework measures structural feasibility (where workers CAN go), not realized reallocation (where they DO go). Empirically: demand dominates aggregate inflows (ρ = 0.80); geometry ranks destinations correctly (MPR = 0.74). Feasibility is the supply-side input to equilibrium analysis.

---

## Key Results

### T Module: Distance Metric Comparison

The 2×2 factorial design isolates the embedding representation as the mechanism. Centroid is the primary specification.

| Representation | Aggregation | α | Pseudo-R² |
|---------------|-------------|---|-----------|
| Embedding (MPNet) | Cosine centroid | 7.40 | 14.08% |
| Embedding (MPNet) | Wasserstein | 8.39 | 13.76% |
| O*NET importance | Cosine | 4.55 | 8.05% |
| O*NET importance | Euclidean | 9.76 | 6.06% |

The embedding ground metric improves explanatory power by +83% over an identity ground metric (7.52% → 13.76%). The core mechanism is **semantic task substitutability**: embeddings capture that "operating forklift" ≈ "driving delivery vehicle."

### I Module: Institutional Distance

| Parameter | Coefficient | SE | t-stat |
|-----------|-------------|-----|--------|
| α (semantic) | 7.404 | 0.036 | 204.0 |
| β (institutional) | 0.139 | 0.004 | 33.7 |

Sample: 89,329 verified CPS transitions (2015–2019, 2022–2024). Estimated with sampled alternatives (J = 11). Separability holds.

### S Module: Shock Integration

| Test | Result |
|------|--------|
| AIOE-distance correlation | r = 0.02 |
| Geometry vs Historical baseline | ΔLL = +23,119 |

AIOE and embedding distance are orthogonal — shock profiles identify exposed occupations, geometry identifies compatible destinations.

### Structural Stability (Pre/Post COVID)

| Period | α | Pseudo-R² |
|--------|---|-----------|
| Pre-COVID (2015–2019) | 7.394 | 14.1% |
| Post-COVID (2022–2024) | 7.358 | 13.9% |

Structural break test: χ²(2) = 0.67, p = 0.72. Task-distance penalties are stable across the largest labor market disruption in decades.

---

## Quick Start

```bash
# Install
pip install -e ".[dev,notebooks]"

# Reproduce key tables
python scripts/reproduce_tables.py

# Run tests
pytest tests/unit tests/integration -v
```

---

## Data Requirements

See [`data/README.md`](data/README.md) for complete data documentation including download instructions for O*NET, CPS/IPUMS, OES, and AIOE.

---

## Repository Structure

```
src/task_space/          # Core implementation
    data/                # O*NET loading, crosswalks, AIOE
    similarity/          # Kernel, overlap, wasserstein, embeddings
    shocks/              # Shock profiles, propagation
    validation/          # Regression, diagnostics, reallocation
    mobility/            # CPS mobility validation

tests/
    unit/                # Fast unit tests
    integration/         # Regression tests against canonical values

scripts/
    reproduce_tables.py  # Reproduce paper Tables 2, 3, 5
    run_*_v07*.py        # Versioned experiment scripts

paper/
    main.tex             # Working paper (full theory + results)
    references.bib       # Bibliography

figures/
    fig1_ai_exposure.py  # AI task exposure by occupation group
    fig2_pseudo_r2.py    # Main result: embedding vs O*NET comparison
    fig3_task_scatter.py # Tasks in semantic space visualization
```

See `CLAUDE.md` for developer context, `LEDGER.md` for scientific state.

---

## Research Roadmap

| Phase | Objective | Requirements |
|-------|-----------|--------------|
| 0.8 | Demand-side integration | Lightcast/JOLTS vacancy data |
| 0.9 | Institutional barrier enhancement | CPS licensing supplement |
| 1.0 | Modality-specific shock profiles | Taxonomy design, benchmarks |
| 1.1 | Full GE with propagation | Individual wage data (LEHD) |

---

## License

Research use only.
