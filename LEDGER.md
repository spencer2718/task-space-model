# LEDGER.md — Task Space Research State

**Current Version:** 0.6.9.0
**Last Updated:** 2025-12-16
**Paper Draft:** `paper/main.tex`

---

## Hard Constraints

These are inviolable. Agents must not contradict or re-litigate.

| ID | Constraint | Rationale | Locked |
|----|------------|-----------|--------|
| HC1 | Wasserstein is PRIMARY geometry | ΔLL = +9,576 over kernel | v0.6.7 |
| HC2 | RTI requires 16-element AA composite | Single O*NET element yields R² ≈ 0 | v0.6.5 |
| HC3 | Kernel bandwidth σ = 0.223 (occupation) | Calibrated to NN median | v0.6.1 |
| HC4 | Asymmetry is HETEROGENEOUS | Ratio varies 0.06–2.79 by sample | v0.6.8 |
| HC5 | Do not row-normalize kernel matrices | Destroys signal with 2,087 activities | v0.6.1 |

---

## Locked Results

Cite these as established facts. See `paper/main.tex` Section 5 for full exposition.

### Geometry Validation (v0.6.7)

| Metric | Kernel | Wasserstein | Δ |
|--------|--------|-------------|---|
| α (semantic) | 5.688 | 8.936 | +57% |
| β (institutional) | 0.278 | 0.142 | -49% |
| Log-likelihood | -192,627 | -183,051 | +9,576 |

**Interpretation:** Workers minimize skill transformation cost. Wasserstein's "earth mover" interpretation is economically validated.

### Mobility Decomposition (v0.6.5)

| Component | Coefficient | t-stat | Interpretation |
|-----------|-------------|--------|----------------|
| d_sem (Wasserstein) | 8.936 | 206.5 | Skill transformation cost |
| d_inst (Job Zone + Cert) | 0.142 | 34.6 | Non-skill barriers |

**Sample:** 89,329 verified CPS transitions (2015–2019, 2022–2024)

### Asymmetric Barriers (v0.6.8)

| Specification | β_up | β_down | Ratio | 95% CI |
|---------------|------|--------|-------|--------|
| Baseline (Wasserstein) | 0.171 | 0.081 | 2.11 | [1.80, 2.42] |
| Prime-age (25–54) | — | — | 1.12 | [0.96, 1.29] |
| Excluding outliers | — | — | 0.06 | [0.01, 0.11] |
| Kernel (comparison) | 0.282 | 0.270 | 1.04 | — |

**Interpretation:** Directional asymmetry is sample-dependent. Neither pure credential-gate nor pure symmetric-friction theories are universally supported.

### RTI Construct Validity (v0.6.8)

| Correlation | Pearson r | p-value |
|-------------|-----------|---------|
| Semantic exposure vs RTI | -0.052 | 0.377 |
| Semantic exposure vs Routine | -0.058 | 0.318 |
| Semantic exposure vs Abstract | -0.051 | 0.378 |

**Interpretation:** Wasserstein-based exposure is orthogonal to Autor-Dorn RTI. The geometry captures mobility friction, not automation susceptibility.

### Automation Prediction (v0.6.5)

| Model | R² | Key β |
|-------|-----|-------|
| RTI only | 9.82% | RTI: -0.077 |
| RTI + Semantic | 12.03% | Semantic: +0.037 (p=0.075) |

**Interpretation:** Marginal improvement. Framework succeeds at mobility, not automation forecasting.

### Wage Comovement (v0.6.5)

| Measure | R² | vs Jaccard |
|---------|-----|------------|
| Normalized kernel overlap | 0.00523 | 3.1× |

**Interpretation:** Geometry is detectable in wage dynamics but explains small share of variance.

---

## Graveyard

Deprecated approaches. Do not retry.

| Approach | Result | Why Deprecated | Version |
|----------|--------|----------------|---------|
| Single O*NET element for RTI | R² ≈ 0 | No predictive power | v0.6.5 |
| Kernel overlap for mobility | ΔLL = -9,576 | Distance compression | v0.6.7 |
| Row-normalized kernel | Signal destruction | Sparse activity space | v0.6.1 |
| Universal asymmetric barriers | Ratio 0.06–2.79 | Sample-dependent | v0.6.8 |

---

## Current Sprint

**Phase:** 0.6.9 complete

No active tasks. Awaiting next research direction.

**Candidate next phases:**
- Path B (JobBERT comparison) — low priority
- Retrospective diagnostic battery — blocked on data
- Modality-specific shocks — long-term

---

## Frontier

### Path B: Embedding Comparison (Deferred)

- **Objective:** JobBERT-v2 vs MPNet
- **Status:** Not started; low priority
- **Blocking:** Custom encoder for asymmetric architecture
- **Expected signal:** May improve mobility α; unlikely to affect RTI correlation

### Retrospective Diagnostic Battery (Specified, Not Executed)

- **Tests A/B/C:** Task composition, employment reallocation, robot displacement
- **Status:** Specifications in `paper/main.tex` Appendix
- **Blocking:** Historical data acquisition (Autor-Dorn replication files)

### Modality-Specific Shocks (Future)

- **Objective:** Distinguish code generation, reasoning, agentic capabilities
- **Status:** Deferred (6–9 month scope)
- **Blocking:** Taxonomy design, benchmark mapping

---

## Artifact Registry

| Artifact | Location | Description |
|----------|----------|-------------|
| Wasserstein (O*NET) | `.cache/artifacts/v1/wasserstein/d_wasserstein_onet.npz` | 894×894 |
| Wasserstein (Census) | `.cache/artifacts/v1/mobility/d_wasserstein_census.npz` | 447×447 |
| Kernel (Census) | `.cache/artifacts/v1/mobility/d_sem_census.npz` | 447×447 |
| Institutional | `.cache/artifacts/v1/mobility/d_inst_census.npz` | 447×447 |
| CPS Mobility (Wasserstein) | `outputs/experiments/path_a_wasserstein_comparison_v0672.json` | Primary |
| Asymmetric (Wasserstein) | `outputs/experiments/mobility_asymmetric_wasserstein_v0682.json` | — |
| Asymmetric Robustness | `outputs/experiments/path_f_robustness_v0683.json` | — |
| RTI Correlation | `outputs/experiments/path_c_rti_construct_validity_v0681.json` | — |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.6.9.0 | 2025-12-16 | LEDGER.md created; CLAUDE.md purified; v0.6.8 findings integrated |
| 0.6.8.0 | — | Wasserstein primary; Path F/C executed |
| 0.6.7.0 | — | Wasserstein module; geometry comparison |
| 0.6.6.0 | — | Asymmetric barriers test (kernel) |
| 0.6.5.0 | — | CPS mobility validation |
