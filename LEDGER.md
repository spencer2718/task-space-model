# LEDGER.md — Task-Space Oracle Research State

**Current Version:** 0.7.2.0
**Last Updated:** 2025-12-17
**Paper Draft:** `paper/main.tex`

---

## Oracle Architecture

The task-space oracle maps **(T, I, S, M) → (Δρ, ΔL, ΔW)**:

| Module | Description | Status |
|--------|-------------|--------|
| **T** (Task representation) | Wasserstein geometry on O*NET embeddings | **Validated** |
| **I** (Institutional structure) | Job zones + certification distance | **Validated** |
| **S** (Shock profiles) | External AIOE integration | **Integrated** |
| **M** (Adjustment mechanisms) | Switching costs, equilibrium | **Preliminary** |

**Outputs:**
- **Δρ**: Occupation-specific task distribution changes
- **ΔL**: Occupation-specific employment changes
- **ΔW**: Occupation-specific wage changes

**Core insight:** Technology acts on tasks. Occupations are probability distributions over tasks. Employment and wage outcomes are aggregations of task-level effects.

**Scope:** The framework measures structural feasibility (where workers CAN go), not realized reallocation (where they DO go). Feasibility is the supply-side input to equilibrium analysis.

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
| HC7 | Technology acts on TASKS, not occupations | Occupations are task distributions; outcomes aggregate task-level effects | v0.7.0.1 |
| HC8 | Oracle outputs are (Δρ, ΔL, ΔW) | Task distribution changes, employment changes, wage changes | v0.7.0.1 |

---

## Methodology Standards

These standards govern experiment design and reporting to prevent forking-path inflation of results.

### MS1: Sample Definition Requirement

Every experiment must document:
- **Population:** What is the full dataset?
- **Sample:** What subset is analyzed?
- **Filter:** What inclusion/exclusion criteria are applied?
- **N:** Final sample size

A result computed on a filtered sample MUST report the filter. Results reported without sample documentation are invalid.

### MS2: Metric Definition Requirement

Every reported metric must specify:
- **Formula:** Exact computation method
- **Aggregation:** Per-unit vs. aggregate (and how aggregated)
- **Baseline:** What comparison or null is implied?

Two metrics with the same name (e.g., "Spearman ρ") computed differently are DIFFERENT METRICS and must be labeled distinctly.

### MS3: Exploratory vs. Confirmatory Distinction

- **Exploratory:** Analysis where methodology was developed while examining data. Results are hypothesis-generating, not hypothesis-testing. Label as "EXPLORATORY."
- **Confirmatory:** Analysis where methodology was fixed BEFORE examining the relevant data. Results can be labeled "VALIDATED" only if confirmatory.

An exploratory result promoted to headline finding without confirmatory replication is a methodology violation.

### MS4: Replication Before Validation

A result may be labeled "VALIDATED" only if:
1. Methodology is documented per MS1-MS2
2. Result is replicated on a held-out sample OR
3. Result is structurally guaranteed (e.g., mathematical identity)

Results that are sample-dependent and unreplicated should be labeled "PRELIMINARY" or "EXPLORATORY."

### MS5: Discrepancy Investigation Requirement

If two experiments report the same metric with substantially different values (>2× difference), consolidation MUST NOT proceed until:
1. Methodological differences are documented
2. Reconciliation is attempted on common sample
3. Correct value is determined and incorrect value is sent to Graveyard

### MS6: Graveyard Discipline

When a result is invalidated:
1. Add to Graveyard with reason
2. Remove from any "VALIDATED" status
3. Update any downstream claims that depended on it
4. Do NOT delete the original experiment file (audit trail)

### MS7: Claim Taxonomy and Language Policy

Claims are classified by evidence type. Allowed verbs depend on evidence class.

| Evidence Class | Description | Allowed Verbs | Forbidden Verbs |
|----------------|-------------|---------------|-----------------|
| **E1** (Score Robustness) | Probabilistic scoring / ΔLL / log score comparisons | "outperforms," "improves scoring," "assigns higher probability," "ranks higher" | "predicts destinations" (unless explicitly meaning scoring) |
| **E2** (Ranking Adequacy) | Rank metrics (MPR, RCM, percentile) | "places realized destination in high-feasibility region," "ranks realized transition above baseline" | "accurately predicts," "identifies correct destination" |
| **E3** (Mechanism) | Behavioral interpretation of coefficients | "consistent with," "suggestive of," "compatible with" | "validates that workers minimize," "estimates transformation cost," "identifies causal cost" |
| **E4** (Structural) | Equilibrium / switching cost identification | Allowed only with explicit identification strategy or external calibration | All causal language without identification |

**Critical rule:** "VALIDATED" status applies only to E1/E2 claims confirmed under MS4. Mechanism claims (E3) receive "CONSISTENT" status, never "VALIDATED."

### MS8: Performance Battery Requirement

Any mobility validation claim must report all four metrics:

1. **Log score / ΔLL** (E1) — probabilistic scoring improvement
2. **RCM or MPR** (E2) — realized destination ranking adequacy
3. **Dispersion diagnostic** — N_eff = exp(entropy) or equivalent
4. **Top-k appropriateness statement** — explicit note on why top-k overlap is or is not reported

Claims citing only a subset of the battery must justify the omission. This prevents metric-shopping and ensures consistent evaluation across experiments.

### MS9: Multiverse Gate for Primary Claims

A claim may be promoted to "Primary" status (featured in README, Abstract, or Introduction) only if:

1. It survives a documented multiverse analysis over plausible specification nodes
2. The multiverse covers at least: embedding choice, bandwidth/threshold, sample definition
3. Win rate ≥ 80% across specifications, OR sensitivity pattern is explicitly documented

Claims meeting criteria receive "ROBUST" designation. Claims from single specifications remain "PRELIMINARY" until multiverse confirmation.

**Current multiverse results:**
- T Module (Wasserstein vs kernel): 100% win rate (81/81 specs). Status: ROBUST.

---

## Methodology Violations Log

| Date | Violation | Result Affected | Resolution |
|------|-----------|-----------------|------------|
| 2025-12-17 | MS1, MS3 | ρ = 0.43 reported without sample filter | Corrected to ρ ≈ 0.13; 0.43 to Graveyard |

### MS1 Compliance Audit (v0.7.0.4)

| Section | Result | Audit Status | Notes |
|---------|--------|--------------|-------|
| T Module (v0.6.7) | ΔLL = +9,576 | ✓ VERIFIED | n=89,329, full sample |
| I Module Asymmetric (v0.6.8) | Ratio = 2.11 | ✓ VERIFIED | n=89,329, baseline full sample; variants explicitly documented |
| S Module (v0.7.0) | ΔLL = +23,119 | ✓ VERIFIED | Train n=97,236, Holdout n=8,880 |
| M Module Demand (v0.7.0.3) | Per-origin ρ ≈ 0.13 | ✓ CORRECTED | Now cites rigorous methodology (v0.7.1) |

**Audit result:** All four flagged results now verified. See `outputs/experiments/ms1_compliance_audit_v0704.json`.

---

## Claim Registry

Canonical phrasing for key claims. All documents (main.tex, README, CLAUDE.md) must use phrasing consistent with evidence class.

| Claim ID | Canonical Text | Evidence Class | Status | Primary Location |
|----------|----------------|----------------|--------|------------------|
| T-E1 | Wasserstein improves probabilistic scoring vs kernel (median ΔLL = +13,052 across 81 specs) | E1 | VALIDATED | main.tex §5.1 |
| T-E1b | Wasserstein advantage robust across embedding, bandwidth, sample, threshold choices (100% win rate) | E1 | VALIDATED | main.tex §5.1 |
| T-E3 | Pattern consistent with feasibility/skill-proximity mechanisms | E3 | CONSISTENT | main.tex §5.1 discussion |
| I-E1 | Institutional distance provides incremental validity over task distance (t = 34.6) | E1 | VALIDATED | main.tex §5.2 |
| I-E3 | Residual institutional effect interpreted as non-skill barriers | E3 | CONSISTENT | main.tex §5.2 discussion |
| S-E1 | AIOE integration improves holdout LL (ΔLL = +23,119) | E1 | VALIDATED | main.tex §5.3 |
| P-E2 | Per-origin pathway ranking: modest signal (ρ ≈ 0.13) | E2 | VALIDATED | main.tex §5.5 |
| P-E3 | Geometry captures supply-side feasibility; demand dominates aggregate flows | E3 | CONSISTENT | main.tex §5.5, §7 |
| D-E1 | Demand-only correlation with aggregate inflows: ρ = 0.80 | E1 | VALIDATED | main.tex §5.5 |

**Maintenance rule:** When adding new claims, assign Claim ID, evidence class, and status before writing prose.

---

## Referee Challenge Table (Living)

Claims most likely to draw referee scrutiny. Updated as vulnerabilities are identified or addressed.

| Claim | Likely Challenge | Current Evidence | Gap / Robustness Needed | Status |
|-------|------------------|------------------|-------------------------|--------|
| T-E1 (Wasserstein wins) | "Sensitive to embedding choice" | 81-spec multiverse, 100% win rate | — | ADDRESSED |
| T-E3 (feasibility mechanism) | "Could be hierarchy, taxonomy artifact, or employer screening" | Reduced-form only | Structural model or IV needed | ACKNOWLEDGED (scope condition) |
| I-E3 (institutional = non-skill barriers) | "Job Zone reflects skill content; absorption claim too strong" | r = 0.25 correlation | Partial R² by subgroup; interaction with licensing | OPEN |
| P-E2 (ρ ≈ 0.13 pathway accuracy) | "Too weak to be useful" | Cortes & Gallipoli ~15% benchmark | Add RCM/MPR metrics | IN PROGRESS |
| P-E3 (geometry = feasibility, demand = realization) | "Post-hoc rationalization of weak prediction" | Demand decomposition (ρ = 0.80) | — | CONSISTENT framing only |
| M-E4 (switching costs) | "External calibration, not identified" | Dix-Carneiro anchor | Sensitivity across 0.75×–6.5× range | ADDRESSED |
| Selection bias | "Only see completed transitions; gates invisible" | Jackson 2023 (24% blocked) cited | — | ACKNOWLEDGED (scope condition) |

**Usage:** Before promoting any claim, check this table. If challenge is OPEN, address or explicitly acknowledge in paper.

---

## Module Validation Checkpoints

Verified validation results for oracle modules. See `paper/main.tex` Section 5 for full exposition.

### T Module: Geometry Validation (v0.6.7)

| Metric | Kernel | Wasserstein | Δ |
|--------|--------|-------------|---|
| α (semantic) | 5.688 | 8.936 | +57% |
| β (institutional) | 0.278 | 0.142 | -49% |
| Log-likelihood | -192,627 | -183,051 | +9,576 |

**Sample:** 89,329 verified CPS transitions (full sample, valid Census codes)

**Status: VALIDATED.** Workers minimize skill transformation cost. Wasserstein's "earth mover" interpretation is economically validated.

### T Module: Performance Battery (v0.7.1.2)

MS8-compliant metrics for the baseline Wasserstein specification:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ΔLL vs uniform | +362,084 | E1: Strong probabilistic improvement |
| MPR mean | 0.7381 | E2: Realized destinations in top 26% on average |
| MPR median | 0.8251 | E2: Half of transitions in top 17.5% |
| RCM mean | 0.5550 | Consideration set includes realized choice |
| N_eff mean | 206.9 | 447 destinations; 46% diffuseness |
| N_eff/J | 0.463 | Distribution too diffuse for top-k |

**Top-k appropriateness:** NOT reported. N_eff/J = 0.46 > 0.30 threshold. Mean effective consideration set = 207 destinations. Top-k overlap is structurally inappropriate for diffuse distributions.

**Interpretation:** The model ranks realized destinations highly (MPR >> 0.5) despite distributing probability mass across ~207 effective destinations. This confirms the model captures feasibility structure without artificially concentrating on a small set.

### I Module: Mobility Decomposition (v0.6.5)

| Component | Coefficient | t-stat | Interpretation |
|-----------|-------------|--------|----------------|
| d_sem (Wasserstein) | 8.936 | 206.5 | Skill transformation cost |
| d_inst (Job Zone + Cert) | 0.142 | 34.6 | Non-skill barriers |

**Sample:** 89,329 verified CPS transitions (2015–2019, 2022–2024)

**Status: VALIDATED.** Institutional distance predicts mobility conditional on semantic distance. Separability holds.

### I Module: Asymmetric Barriers (v0.6.8)

| Specification | β_up | β_down | Ratio | 95% CI |
|---------------|------|--------|-------|--------|
| Baseline (Wasserstein) | 0.171 | 0.081 | 2.11 | [1.80, 2.42] |
| Prime-age (25–54) | — | — | 1.12 | [0.96, 1.29] |
| Excluding outliers | — | — | 0.06 | [0.01, 0.11] |
| Kernel (comparison) | 0.282 | 0.270 | 1.04 | — |

**Sample:** Baseline = 89,329 verified CPS transitions (full sample). Variants apply explicit filters.

**Status: HETEROGENEOUS.** Neither pure credential-gate nor pure symmetric-friction theories are universally supported.

### S Module: Shock Integration (v0.7.0)

| Test | Metric | Result |
|------|--------|--------|
| Geometry vs Historical | ΔLL | +23,119 |
| Geometry vs Uniform | ΔLL | +2,239 |
| AIOE-Wasserstein correlation | r | 0.020 |
| Top-5 destination overlap | — | 0.0 |

**Sample:** Train = 97,236 transitions (2015-2019, 2022-2023); Holdout = 8,880 transitions (2024)

**Status: INTEGRATED.** AIOE and Wasserstein are orthogonal—shock profiles identify exposed occupations, geometry identifies compatible destinations.

### S Module: Pathway Accuracy Audit (v0.7.0.3c)

| Methodology | Aggregate ρ | Per-origin ρ | n origins |
|-------------|-------------|--------------|-----------|
| Model probability, all destinations | 0.188 | 0.128 | 233 |
| Raw 1/distance, common destinations | 0.043 | 0.316 | 177 |

**CORRECTION:** Original v0.7.0c reported ρ = 0.43, but this was computed on a **restricted sample** (exposed origins only, n=60). On the full 2024+ holdout (n=424 origins), the model probability method yields ρ = 0.128 per-origin.

**Methodology note:** The two per-origin metrics differ because:
- Model probability evaluates over ALL 447 destinations (including zeros)
- Raw 1/distance evaluates only over COMMON destinations (survivorship bias favors the latter)

**Status: CORRECTED.** Per-origin pathway accuracy is modest (~0.13) using consistent methodology. The geometry provides weak but positive signal for destination ranking within each origin.

### M Module: Switching Cost Calibration (v0.7.0)

| Parameter | Value | Source |
|-----------|-------|--------|
| SC per unit Wasserstein | 3.84 wage-years | External calibration (Dix-Carneiro 2014) |
| SC per unit Wasserstein | $276,175 | Using OES 2023 mean wage |
| Median transition distance | 0.52 | Training sample |
| Typical transition cost | 2.0 wage-years | By construction (calibration target) |

**Status: PRELIMINARY.** Endogenous identification failed (β_wage < 0). External calibration adopted. Individual-level wage data (LEHD) required for structural identification.

### M Module: Switching Cost Sensitivity (v0.7.0.2)

| Calibration Source | Anchor (wage-years) | SC/unit Wasserstein |
|--------------------|---------------------|---------------------|
| Lee & Wolpin (2006) | 0.75 | 1.44 |
| Dix-Carneiro mid (adopted) | 2.00 | 3.84 |
| Dix-Carneiro upper | 2.70 | 5.18 |
| Artuc et al. (2010) | 6.50 | 12.47 |

**Finding:** Ordinal predictions invariant to calibration choice. Transition rankings (which switches are harder) preserved across entire literature range. Absolute magnitudes span ~9× but relative orderings are scale-invariant.

**Note:** Invariance is structural (linear calibration), not empirical. The method guarantees ordering preservation by construction.

**Status: VALIDATED.** Qualitative findings robust to calibration uncertainty.

### M Module: Demand Decomposition (v0.7.0.3)

| Predictor | Spearman ρ | Interpretation |
|-----------|------------|----------------|
| Demand-only (openings) | 0.798 | **Dominant** — explains aggregate inflows |
| Geometry-only (1/distance) | 0.043 | Negligible for aggregate prediction |
| Full flow (outflow × openings / distance) | 0.791 | Geometry adds nothing to demand |

**Per-origin vs aggregate:**
- Per-origin ρ ≈ **0.13** (rigorous: model probability over all destinations; see v0.7.0.3c)
- Aggregate ρ = 0.043 (total inflow prediction: rank destinations by sum of inflows)

**Destination characteristics:**
- Observed top-5 avg openings: 312.8k
- Predicted (geometry) top-5 avg openings: 185.4k
- Ratio: 1.7× — flows favor high-demand over geometrically-central destinations

**Status: VALIDATED.** Aggregate reallocation is demand-dominated. Geometry measures supply-side feasibility (where workers CAN go), not demand-side outcomes (where they DO go). The oracle's (T, I, S, M) architecture correctly separates these components.

### Complementary Validations

**RTI Construct Validity (v0.6.8):**
- Semantic exposure vs RTI: r = -0.052 (p = 0.377)
- **Interpretation:** Geometry captures mobility friction, not automation susceptibility—orthogonal to RBTC.

**Automation Prediction (v0.6.5):**
- RTI only: R² = 9.82%
- RTI + Semantic: R² = 12.03% (Δ = 2.2%, p = 0.075)
- **Interpretation:** Marginal improvement. Framework succeeds at mobility, not automation forecasting.

**Wage Comovement (v0.6.5):**
- Normalized kernel overlap: R² = 0.52%
- **Interpretation:** Detectable but explains small share of variance.

---

## Graveyard

Deprecated approaches. Do not retry.

| Approach | Result | Why Deprecated | Version |
|----------|--------|----------------|---------|
| Single O*NET element for RTI | R² ≈ 0 | No predictive power | v0.6.5 |
| Kernel overlap for mobility | ΔLL = -9,576 | Distance compression | v0.6.7 |
| Row-normalized kernel | Signal destruction | Sparse activity space | v0.6.1 |
| Universal asymmetric barriers | Ratio 0.06–2.79 | Sample-dependent | v0.6.8 |
| Reallocation forecasting from geometry alone | Top-5 overlap = 0 | Demand side, capacity, credential gates required | v0.7.0 |
| Endogenous switching cost identification | β_wage < 0 | Need individual wages at transition | v0.7.0 |
| ρ = 0.43 pathway accuracy | Computed on filtered sample (n=60 exposed origins); full sample ρ = 0.13 | MS1/MS3 violation; sample filter undocumented | v0.7.0.3c |

---

## Current Sprint

**Phase:** 0.7.1 — Consolidation Complete

**Completed (0.7.0.x → 0.7.1):**
- 0.7.0.2: Switching cost sensitivity (ordinal invariance confirmed)
- 0.7.0.3: Demand probe (demand-only ρ = 0.798)
- 0.7.0.3b: Demand decomposition (geometry-only ρ = 0.043, per-origin ρ = 0.128)
- 0.7.0.3c: Methodology audit (ρ = 0.43 → 0.13 correction)
- 0.7.0.4: MS1 compliance audit (T/I/S modules verified)
- 0.7.1: Paper updated with corrected metrics and demand decomposition

**Paper status:** v0.7.1 — Metrics corrected, demand decomposition integrated

**Next:** Phase 0.8 planning

---

## Frontier

### Demand-Side Integration (Phase 0.8)

- **Objective:** Add vacancy dynamics to equilibrium model
- **Finding (v0.7.0.3):** Static openings alone explain ρ = 0.798 of aggregate inflows
- **Next step:** Time-varying vacancy rates (JOLTS) for dynamic reallocation
- **Candidates:** JOLTS by major occupation group, Lightcast (if accessible)
- **Status:** Validated as critical; demand dominates geometry for aggregate prediction

### Institutional Barrier Enhancement (Phase 0.9)

- **Problem:** γ_inst/γ_sem = 0.015 underweights credentials
- **Root cause:** Estimated from completed transitions; blocked attempts unobserved
- **Alternative:** Exogenous credential classification (nursing, teaching, licensed trades)
- **Data:** CPS licensing supplement (2015+)

### Modality-Specific Shocks (Phase 1.0)

- **Objective:** Distinguish code generation, reasoning, agentic capabilities
- **Blocking:** Taxonomy design, benchmark mapping
- **Timeline:** 6–9 month scope

### Embedding Comparison (Deferred)

- **Objective:** JobBERT-v2 vs MPNet
- **Expected signal:** May improve mobility α; unlikely to affect RTI correlation
- **Priority:** Low (geometry change >> embedding change)

### Retrospective Diagnostic Battery (Deferred)

- **Tests A/B/C:** Task composition, employment reallocation, robot displacement
- **Specifications:** `paper/main.tex` Appendix
- **Blocking:** Historical data acquisition (Autor-Dorn replication files)

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
| Sensitivity Analysis | `outputs/experiments/sensitivity_switching_costs_v0702.json` | v0.7.0.2 |
| Demand Probe | `outputs/experiments/demand_probe_v0703.json` | v0.7.0.3 |
| Demand Decomposition | `outputs/experiments/demand_probe_decomposition_v0703b.json` | v0.7.0.3b |
| Methodology Audit | `outputs/experiments/methodology_audit_v0703c.json` | v0.7.0.3c |
| MS1 Compliance Audit | `outputs/experiments/ms1_compliance_audit_v0704.json` | v0.7.0.4 |
| T Module Multiverse | `outputs/multiverse/t_module_v0712/summary.json` | v0.7.1.2 |
| Performance Battery | `outputs/experiments/performance_battery_baseline_v0712.json` | v0.7.1.2 |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.7.2.0 | 2025-12-17 | Paper v0.7.2 complete. Multiverse + performance battery integrated. MS7-MS9 regime active. |
| 0.7.1.2 | 2025-12-17 | Performance battery implemented (MS8); MPR=0.74, RCM=0.56, N_eff=207 |
| 0.7.1.1 | 2025-12-17 | Added MS7-MS9 (language policy, performance battery, multiverse gate); Claim Registry; Referee Challenge Table |
| 0.7.1 | 2025-12-17 | Paper updated; ρ corrected (0.43→0.13); demand decomposition integrated |
| 0.7.0.4 | 2025-12-17 | MS1 compliance audit; 3/4 flagged results verified, sample sizes inlined |
| 0.7.0.3c | 2025-12-17 | **METHODOLOGY AUDIT**: Original ρ = 0.43 corrected to 0.13; was sample-restricted |
| 0.7.0.3 | 2025-12-17 | Demand probe validated; demand-only ρ = 0.798 dominates; geometry ρ = 0.043 |
| 0.7.0.2 | 2025-12-17 | Switching cost sensitivity; ordinal invariance confirmed |
| 0.7.0.1 | 2025-12-16 | Oracle architecture framing; documentation hierarchy; HC7-HC8 added |
| 0.7.0 | 2025-12-16 | Shock integration validated; cost calibration; pathway identification |
| 0.6.9.0 | 2025-12-16 | LEDGER.md created; CLAUDE.md purified |
| 0.6.8.0 | — | Wasserstein primary; Path F/C executed |
| 0.6.7.0 | — | Wasserstein module; geometry comparison |
| 0.6.6.0 | — | Asymmetric barriers test (kernel) |
| 0.6.5.0 | — | CPS mobility validation |
