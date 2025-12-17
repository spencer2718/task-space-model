# Task-Space Oracle

Infrastructure for analyzing how task-level technological change propagates to labor market outcomes.

**Version 0.7.1.0** — Demand Decomposition Validated

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

| Module | Component | Status | Key Result |
|--------|-----------|--------|------------|
| **T** | Wasserstein geometry | ✓ Validated | ΔLL = +9,576 over kernel (n=89,329) |
| **I** | Institutional distance | ✓ Validated | t = 34.6 conditional on T (n=89,329) |
| **S** | AIOE integration | ✓ Integrated | r = 0.02 (orthogonal to T) |
| **M** | Switching costs | ⚠️ Calibrated | 3.84 wage-years/unit (external) |
| **M** | Demand decomposition | ✓ Quantified | Demand ρ = 0.80; geometry ρ ≈ 0.13 |
| — | Scope | ✓ Delineated | Supply-side feasibility, not realized flows |

**Scope:** The framework measures structural feasibility (where workers CAN go), not realized reallocation (where they DO go). Empirically: demand dominates aggregate inflows (ρ = 0.80); geometry captures supply-side constraints (per-origin ρ ≈ 0.13). Feasibility is the supply-side input to equilibrium analysis.

---

## Validation Details

### T Module: Geometry Comparison

Wasserstein substantially outperforms kernel overlap for predicting worker mobility:

| Metric | Kernel | Wasserstein | Δ |
|--------|--------|-------------|---|
| α (semantic) | 5.688 | 8.936 | +57% |
| Log-likelihood | -192,627 | -183,051 | +9,576 |

Workers minimize skill transformation cost. The "earth mover" interpretation is economically validated.

### I Module: Institutional Distance

| Component | Coefficient | t-stat |
|-----------|-------------|--------|
| d_sem (Wasserstein) | 8.936 | 206.5 |
| d_inst (Job Zone + Cert) | 0.142 | 34.6 |

Sample: 89,329 verified CPS transitions (2015–2019, 2022–2024). Separability holds.

### S Module: Shock Integration

| Test | Result |
|------|--------|
| AIOE-Wasserstein correlation | r = 0.02 |
| Geometry vs Historical baseline | ΔLL = +23,119 |

AIOE and Wasserstein are orthogonal—shock profiles identify exposed occupations, geometry identifies compatible destinations.

### M Module: Switching Costs

| Parameter | Value |
|-----------|-------|
| SC per unit Wasserstein | 3.84 wage-years |
| Source | External calibration (Dix-Carneiro 2014) |

Endogenous identification failed (β_wage < 0). Individual-level wage data required.

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

### OES Wage Data (For Wage Comovement)
Download national OES files from https://www.bls.gov/oes/tables.htm.
Extract to `data/external/oes/`.

**Note:** BLS blocks automated downloads. Download manually via browser.

---

## Repository Structure

```
src/task_space/          # Core implementation
    data/                # O*NET loading, crosswalks, AIOE
    similarity/          # Kernel, overlap, wasserstein
    shocks/              # Shock profiles, propagation
    validation/          # Regression, diagnostics, reallocation
    mobility/            # CPS mobility validation

tests/
    unit/                # Fast unit tests
    integration/         # Slower integration tests

paper/
    main.tex             # Theory + specifications
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

## Version History

| Version | What Changed |
|---------|--------------|
| **0.7.1.0** | Demand decomposition validated. Metric correction (ρ = 0.43 → 0.13). Paper updated. |
| 0.7.0.1 | Oracle architecture framing. Documentation hierarchy. |
| 0.6.9.0 | LEDGER.md created. Asymmetric barriers → heterogeneous. |
| 0.6.8.0 | Wasserstein validated. |
| 0.6.7.x | Wasserstein module, geometry comparison. |

---

## License

Research use only.
