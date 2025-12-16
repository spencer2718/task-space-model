# LEDGER.md — Task Space Research State

**Current Version:** 0.7.0
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
| HC6 | Institutional barriers are FRICTION not GATES | γ_inst/γ_sem = 0.015; credential-blocking unobserved | v0.7.0 |

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

### Shock Integration (v0.7.0)

| Test | Metric | Result |
|------|--------|--------|
| Geometry vs Historical | ΔLL | +23,119 |
| Geometry vs Uniform | ΔLL | +2,239 |
| AIOE-Wasserstein correlation | r | 0.020 |
| Directional accuracy | Spearman ρ | 0.432 |
| Top-5 destination overlap | — | 0.0 |

**Interpretation:** Geometry predicts structural feasibility and directional skill compatibility, not realized reallocation. AIOE and Wasserstein are orthogonal—shock profiles identify exposed occupations, geometry identifies compatible destinations.

### Switching Cost Calibration (v0.7.0)

| Parameter | Value | Source |
|-----------|-------|--------|
| SC per unit Wasserstein | 3.84 wage-years | External calibration (Dix-Carneiro 2014) |
| SC per unit Wasserstein | $276,175 | Using OES 2023 mean wage |
| Median transition distance | 0.52 | Training sample |
| Typical transition cost | 2.0 wage-years | By construction (calibration target) |

**Limitation:** Endogenous wage identification failed (β_wage < 0 in log specification). Occupation-mean wages do not capture entry wages switchers receive. Individual-level wage data (LEHD, admin records) required for structural identification.

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
| Reallocation forecasting from geometry alone | Top-5 overlap = 0 | Demand side, capacity constraints, credential gates required | v0.7.0 |
| Endogenous switching cost identification (CPS + OES) | β_wage < 0 | Need individual wages at transition | v0.7.0 |

---

## Current Sprint

**Phase:** 0.7.0 complete

**Completed:**
- 0.7a: Shock integration validated (geometry >> historical)
- 0.7b: External cost calibration (3.84 wage-years/unit)
- 0.7c: Pathway identification validated (ρ = 0.43), reallocation forecasting failed (top-5 = 0)

**Next candidate phases:**
- Paper revision: Scope clarification, limitations, 0.7 results
- Demand-side integration (Lightcast/JOLTS)
- Institutional barrier enhancement (exogenous classification)

---

## Frontier

### Institutional Barrier Enhancement (Identified, Not Pursued)

- **Problem:** γ_inst/γ_sem = 0.015 underweights credentials
- **Root cause:** Estimated from completed transitions; blocked attempts unobserved
- **Required:** Data on attempted transitions, not just completed
- **Alternative:** Exogenous credential classification (nursing, teaching, licensed trades)
- **Status:** Deferred; requires different data or strong assumptions

### Demand-Side Integration (Future)

- **Objective:** Add job openings / vacancy data to reallocation model
- **Candidates:** Lightcast (if accessible), JOLTS by occupation
- **Status:** Deferred; 0.7c establishes need

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
| Shock Integration | `outputs/experiments/shock_integration_v070a.json` | 0.7a results |
| Scaled Costs | `outputs/experiments/scaled_costs_v070b.json` | 0.7b results |
| Reallocation | `outputs/experiments/reallocation_v070c.json` | 0.7c results |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.7.0 | 2025-12-16 | Shock integration validated; cost calibration; pathway identification (ρ=0.43); reallocation forecasting failed |
| 0.6.9.0 | 2025-12-16 | LEDGER.md created; CLAUDE.md purified; v0.6.8 findings integrated |
| 0.6.8.0 | — | Wasserstein primary; Path F/C executed |
| 0.6.7.0 | — | Wasserstein module; geometry comparison |
| 0.6.6.0 | — | Asymmetric barriers test (kernel) |
| 0.6.5.0 | — | CPS mobility validation |
