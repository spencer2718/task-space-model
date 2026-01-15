# Task-Space Oracle

Infrastructure for analyzing how task-level technological change propagates to labor market outcomes.

**Version 0.7.5.1** — COVID Structural Stability

---

## Quick Navigation

| If you want to... | Start here |
|-------------------|------------|
| Understand the research | `paper/main.tex` (full theory) |
| Run the main validation | `scripts/run_distance_comparison_v0732.py` |
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
| **T** (Embedding-informed distance) | ✓ Robust | 100% win rate (81/81 specs); median ΔLL = +13,052 |
| **I** (Institutional distance) | ✓ Validated | t = 34.6 conditional on T |
| **S** (Shock integration) | ✓ Validated | ΔLL = +23,119 on holdout |
| **M** (Switching costs) | ⚠️ Calibrated | External anchor (Dix-Carneiro) |
| Pathway ranking | ✓ Validated | MPR = 0.74; per-origin ρ ≈ 0.13 |
| Demand decomposition | ✓ Quantified | Demand ρ = 0.80; geometry ρ = 0.04 |
| Structural stability | ✓ Validated | Aggregate Δα < 1% (p = 0.76); telework heterogeneity δ₄ = -0.086 (p = 0.01) |
| Retrospective battery | ⚠️ Partial | Test B: 1+/5; COVID stability: aggregate stable, telework heterogeneity |

**Scope:** The framework measures structural feasibility (where workers CAN go), not realized reallocation (where they DO go). Empirically: demand dominates aggregate inflows (ρ = 0.80); geometry ranks destinations correctly (MPR = 0.74). Feasibility is the supply-side input to equilibrium analysis.

---

## Validation Details

### T Module: Distance Metric Attribution

Task distance comparison shows embedding ground metric (+96%) dominates distributional treatment (+3%). Wasserstein preserves theoretical grounding; cosine_embed achieves comparable predictive performance (ρ = 0.95).

| Metric | Kernel | Wasserstein | Δ |
|--------|--------|-------------|---|
| α (semantic) | 5.688 | 8.936 | +57% |
| Log-likelihood | -192,627 | -183,051 | +9,576 |

The core mechanism is **semantic task substitutability**: embeddings capture that "operating forklift" ≈ "driving delivery vehicle."

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

See `CLAUDE.md` for developer context, `LEDGER.md` for scientific state. Use [`templates/sprint_summary.md`](templates/sprint_summary.md) when writing sprint summaries (one per sprint when updating LEDGER and the paper).

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
| **0.7.5.0** | COVID structural stability. Aggregate geometry invariant; elevated hiring standards for teleworkable occupations post-COVID. |
| 0.7.4.1 | Pre/post COVID comparison implementation. |
| 0.7.4.0 | Attribution reframe: embedding-informed distance. Ground metric validation (+96% vs identity). Gravity model (3.5% partial R², consistent with C-G). |
| 0.7.3.0 | Retrospective battery (Test B: 1+/5). IPUMS pipeline. Documentation schema (MS10, Decision Authority). |
| 0.7.2.0 | Multiverse robustness (81/81). Performance battery (MPR=0.74). MS7-MS9 methodology regime. |
| 0.7.1.0 | Demand decomposition validated. Metric correction (ρ = 0.43 → 0.13). |
| 0.7.0.1 | Oracle architecture framing. Documentation hierarchy. |
| 0.6.9.0 | LEDGER.md created. Asymmetric barriers → heterogeneous. |
| 0.6.8.0 | Wasserstein validated. |
| 0.6.7.x | Wasserstein module, geometry comparison. |

---

## License

Research use only.
