# Task-Space Oracle

Infrastructure for analyzing how task-level technological change propagates to labor market outcomes.

**Version 0.7.2.0** — Multiverse Robustness + Performance Battery

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
| **T** (Wasserstein geometry) | ✓ Robust | 100% win rate (81/81 specs); median ΔLL = +13,052 |
| **I** (Institutional distance) | ✓ Validated | t = 34.6 conditional on T |
| **S** (Shock integration) | ✓ Validated | ΔLL = +23,119 on holdout |
| **M** (Switching costs) | ⚠️ Calibrated | External anchor (Dix-Carneiro) |
| Pathway ranking | ✓ Validated | MPR = 0.74; per-origin ρ ≈ 0.13 |
| Demand decomposition | ✓ Quantified | Demand ρ = 0.80; geometry ρ = 0.04 |

**Scope:** The framework measures structural feasibility (where workers CAN go), not realized reallocation (where they DO go). Empirically: demand dominates aggregate inflows (ρ = 0.80); geometry ranks destinations correctly (MPR = 0.74). Feasibility is the supply-side input to equilibrium analysis.

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

See [`data/README.md`](data/README.md) for complete data documentation including download instructions for O*NET, CPS/IPUMS, OES, and AIOE.

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
| **0.7.2.0** | Multiverse robustness (81/81). Performance battery (MPR=0.74). MS7-MS9 methodology regime. |
| 0.7.1.0 | Demand decomposition validated. Metric correction (ρ = 0.43 → 0.13). |
| 0.7.0.1 | Oracle architecture framing. Documentation hierarchy. |
| 0.6.9.0 | LEDGER.md created. Asymmetric barriers → heterogeneous. |
| 0.6.8.0 | Wasserstein validated. |
| 0.6.7.x | Wasserstein module, geometry comparison. |

---

## License

Research use only.
