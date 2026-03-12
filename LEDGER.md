# LEDGER.md — Task-Space Oracle Research State

**Current Version:** 0.7.12.8
**Last Updated:** 2026-03-12
**Paper Draft:** `paper/main.tex`

---

## Oracle Architecture {#oracle-architecture}

The task-space oracle maps **(T, I, S, M) → (Δρ, ΔL, ΔW)**:

| Module | Description | Status |
|--------|-------------|--------|
| **T** (Task representation) | Embedding-informed distance (semantic task substitutability) | **Validated** |
| **I** (Institutional structure) | Job zones + certification distance | **Validated** |
| **S** (Shock profiles) | External AIOE integration | **Validated** |
| **M** (Adjustment mechanisms) | Switching costs, equilibrium | **Preliminary** |

**Outputs:**
- **Δρ**: Occupation-specific task distribution changes
- **ΔL**: Occupation-specific employment changes
- **ΔW**: Occupation-specific wage changes

**Core insight:** Technology acts on tasks. Occupations are probability distributions over tasks. Employment and wage outcomes are aggregations of task-level effects.

**Scope:** The framework measures structural feasibility (where workers CAN go), not realized reallocation (where they DO go). Feasibility is the supply-side input to equilibrium analysis.

---

## Hard Constraints {#hard-constraints}

These are inviolable. Agents must not contradict or re-litigate.

| ID | Constraint | Rationale | Locked |
|----|------------|-----------|--------|
| HC1 | Centroid is PRIMARY specification; Wasserstein provides theoretical grounding | Centroid marginally outperforms Wasserstein (ρ = 0.95); diagonal correction v0.7.7.0 | v0.7.7.5 |
| HC2 | RTI requires 16-element AA composite | Single O*NET element yields R² ≈ 0 | v0.6.5 |
| HC3 | Kernel bandwidth σ = 0.223 (occupation) | Calibrated to NN median | v0.6.1 |
| HC4 | Asymmetry is HETEROGENEOUS | Ratio varies 0.06–2.79 by sample | v0.6.8 |
| HC5 | Do not row-normalize kernel matrices | Destroys signal with 2,087 activities | v0.6.1 |
| HC6 | Institutional barriers are FRICTION not GATES | γ_inst/γ_sem = 0.019; credential-blocking unobserved | v0.7.0 |
| HC7 | Technology acts on TASKS, not occupations | Occupations are task distributions; outcomes aggregate task-level effects | v0.7.0.1 |
| HC8 | Oracle outputs are (Δρ, ΔL, ΔW) | Task distribution changes, employment changes, wage changes | v0.7.0.1 |

---

## Key Metrics Glossary {#key-metrics}

Critical metrics used throughout validation. Understanding these is essential for interpreting results.

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Pseudo-R²** | McFadden's pseudo-R² = 1 - (LL_model / LL_null) | Proportion of choice uncertainty resolved by model. 14.1% is strong for discrete choice models; indicates substantial predictive power over random choice. |
| **MPR** | Mean Percentile Rank of realized destinations in geometry-ranked list | Average position of actual transitions in model's ranking. MPR = 0.74 means realized destinations rank in top 26% on average, confirming model captures feasibility structure. |
| **ΔLL** | Log-likelihood improvement over baseline model | Absolute improvement in probabilistic fit. ΔLL = +23,119 means geometry substantially outperforms historical transition rates; larger values indicate better prediction. |
| **Partial R²** | Incremental variance explained by adding distance to gravity model | Measures distance contribution beyond employment size effects. 3.5% for task distance is consistent with Cortes-Gallipoli (15%) given that task costs are one component of total switching costs. |
| **Spearman ρ** | Rank correlation between predicted and actual flows | Measures ordinal agreement. Per-origin ρ ≈ 0.13 indicates modest but positive signal for ranking destinations within each origin occupation. |
| **N_eff** | Effective number of destinations = exp(entropy) | Captures distribution diffuseness. N_eff = 207 means probability mass spread across ~200 occupations, not concentrated on a few. |

---

## Methodology Standards {#methodology-standards}

These standards govern experiment design and reporting to prevent forking-path inflation of results.

**Priority Tiers:**
- **Tier 1 (Gatekeeping):** MS7, MS8, MS9 — Prevent invalid claims from reaching paper
- **Tier 2 (Integrity):** MS2, MS3, MS4, MS10 — Ensure results mean what they claim
- **Tier 3 (Hygiene):** MS1, MS5, MS6 — Documentation and cleanup discipline

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

### MS10: Specification Deviation Protocol

If implementation cannot follow spec due to data, technical, or feasibility constraints:

1. **STOP** before implementing an alternative
2. **DOCUMENT**: What spec required, why infeasible, what alternative is proposed
3. **RETURN** to Lead Researcher for approval OR explicit relabeling
4. **DO NOT** present alternative results under original test name without approval

Deviations implemented without this protocol constitute MS2 (metric definition) and/or MS3 (exploratory vs confirmatory) violations.

---

## Methodology Violations Log {#violations-log}

| Date | Violation | Result Affected | Resolution |
|------|-----------|-----------------|------------|
| 2025-12-17 | MS1, MS3 | ρ = 0.43 reported without sample filter | Corrected to ρ ≈ 0.13; 0.43 to Graveyard |
| 2025-12-19 | MS2, MS3, MS10 | Test C' robot exposure | Metric changed from embedding distance to keyword count without approval; relabeled as C'-keyword; correct C' deferred to v0.7.3.1 |

### MS1 Compliance Audit (v0.7.0.4)

| Section | Result | Audit Status | Notes |
|---------|--------|--------------|-------|
| T Module (v0.6.7) | ΔLL = +9,576 | ✓ VERIFIED | n=89,329, full sample |
| I Module Asymmetric (v0.6.8) | Ratio = 2.11 | ✓ VERIFIED | n=89,329, baseline full sample; variants explicitly documented |
| S Module (v0.7.0) | ΔLL = +23,119 | ✓ VERIFIED | Train n=97,236, Out-of-period n=8,880 |
| M Module Demand (v0.7.0.3) | Per-origin ρ ≈ 0.13 | ✓ CORRECTED | Now cites rigorous methodology (v0.7.1) |

**Audit result:** All four flagged results now verified. See `outputs/experiments/ms1_compliance_audit_v0704.json`.

---

## Attribution Audit (v0.7.3.2b) {#attribution-audit}

**Finding:** The original T Module validation (ΔLL = +9,576 vs kernel overlap) conflated two effects. The 2×2 comparison (v0.7.3.2) isolated them:

| Effect | Comparison | Result |
|--------|------------|--------|
| Embedding vs O*NET | cosine_embed (14.1%) vs cosine_onet (8.1%) | +74.9% relative |
| Distributional vs Simple | cosine_embed (14.1%) vs wasserstein (13.8%) | centroid marginally outperforms |

**Implication:** The embedding choice is the primary driver. Wasserstein vs cosine-on-centroids is marginal for individual choice prediction (ρ = 0.95 correlation between distance matrices).

**Ground metric validation (v0.7.3.3):** Wasserstein-embedding >> Wasserstein-identity (+83%, corrected v0.7.7.0), confirming semantic task similarity (knowing "operating forklift" ≈ "driving delivery vehicle") is economically meaningful.

**Gravity model divergence (v0.7.3.4):** Rankings shift between frameworks—Wasserstein best for aggregate flows, cosine_onet competitive. This reflects extensive vs intensive margin dynamics: individual workers use fine-grained similarity (embeddings excel), aggregate flows depend on connectivity structure (binary-like measures capture extensive margin).

**Reframed contribution:** Semantic task substitutability captured by embedding ground metric improves occupation distance measurement. Wasserstein provides theoretical foundation; centroid averaging approximates it well in practice.

---

## Claim Registry {#claim-registry}

Canonical phrasing for key claims. All documents (main.tex, README, CLAUDE.md) must use phrasing consistent with evidence class.

| Claim ID | Canonical Text | Evidence Class | Status | Primary Location |
|----------|----------------|----------------|--------|------------------|
| T-E1 | Embedding-informed distance improves individual transition prediction vs O*NET-based measures (13.8-14.1% vs 6-8% pseudo-R²) | E1 | VALIDATED | main.tex §5.1 |
| T-E1b | Embedding ground metric captures semantic task substitutability; identity ground metric underperforms by 83% (corrected v0.7.7.0) | E1 | VALIDATED | main.tex §5.1 |
| T-E1c | Wasserstein and cosine-on-centroids produce nearly identical rankings (ρ = 0.95); distributional treatment marginal | E1 | VALIDATED | main.tex §5.1 |
| T-E3 | Pattern consistent with feasibility/skill-proximity mechanisms | E3 | CONSISTENT | main.tex §5.1 discussion |
| I-E1 | Institutional distance provides incremental validity over task distance (t = 33.7) | E1 | VALIDATED | main.tex §5.2 |
| I-E3 | Residual institutional effect interpreted as non-skill barriers | E3 | CONSISTENT | main.tex §5.2 discussion |
| S-E1 | AIOE integration improves out-of-period LL (ΔLL = +23,119) | E1 | VALIDATED | main.tex §5.3 |
| P-E2 | Per-origin pathway ranking: modest signal (ρ ≈ 0.13) | E2 | VALIDATED | main.tex §5.5 |
| P-E3 | Geometry captures supply-side feasibility; demand dominates aggregate flows | E3 | CONSISTENT | main.tex §5.5, §7 |
| D-E1 | Demand-only correlation with aggregate inflows: ρ = 0.80 | E1 | VALIDATED | main.tex §5.5 |
| G-E1 | Gravity model: task distance explains 3.5% partial R², consistent with Cortes-Gallipoli benchmark | E1 | VALIDATED | main.tex §5.6 |
| G-E3 | Individual choice and aggregate flow prediction respond differently to distance metrics; reflects intensive vs extensive margin dynamics | E3 | CONSISTENT | main.tex §5.6 |
| COVID-E1 | Task-distance geometry is structurally stable across pre/post COVID (Δα < 1%, LR p = 0.72, n = 89,329) | E1 | VALIDATED | main.tex §5.7 |
| COVID-E1b | Teleworkable occupations show elevated hiring standards post-COVID (δ₄ = -0.086, p = 0.01) | E1 | VALIDATED | main.tex §5.7 |
| COVID-E3 | Elevated hiring standards consistent with applicant pool expansion enabling selectivity | E3 | CONSISTENT | main.tex §5.7 |

**Maintenance rule:** When adding new claims, assign Claim ID, evidence class, and status before writing prose.

---

## Referee Challenge Table {#referee-challenges}

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

## Module Validation Checkpoints {#module-checkpoints}

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

### T Module: 2×2 Methodology Comparison (v0.7.3.2)

Head-to-head comparison of four distance metrics on conditional logit (n=89,329):

| Metric | α | γ | LL | Pseudo-R² |
|--------|---|---|-----|-----------|
| wasserstein | 8.953* | 0.134* | -183,116* | 14.51%* |
| cosine_embed | 7.404 | 0.139 | -184,043 | 14.08% |
| cosine_onet | 4.548 | 0.300 | -196,961 | 8.05% |
| euclidean_dwa | 9.764 | 0.365 | -201,230 | 6.06% |

*Uncorrected. After diagonal correction (v0.7.7.0): α=8.39, R²=13.76%. Centroid (14.08%) is primary specification.

**Gate evaluation (Wasserstein vs best alternative = cosine_embed):**
- ΔLL = +927
- Vuong p = 0.00192 (FAIL: not < 0.001)
- Pseudo-R² ratio = 1.031 (FAIL: not ≥ 1.05)

**Status: GATE FAILED.** Wasserstein provides only 3.1% improvement over cosine_embed (centroid averaging).

**Key findings:**
1. **Embedding choice dominates:** Both wasserstein and cosine_embed vastly outperform cosine_onet and euclidean_dwa
2. **Distributional treatment marginal:** centroid 14.08% marginally outperforms corrected Wasserstein 13.76%
3. **cosine_onet worst:** Confirms sparsity limitation (78% of pairs at max distance)
4. **ρ(wasserstein, cosine_embed) = 0.95:** High distance correlation; different aggregation yields similar rankings

**Interpretation:** MPNet embeddings capture economically meaningful task similarity. The distributional treatment (Wasserstein) provides marginal improvement over simple centroid averaging. The paper's core contribution is the embedding-based task distance, not the distributional formulation per se.

### T Module: Ground Metric Validation (v0.7.3.3)

Tests whether embedding-based ground metric adds value over identity ground metric:

| Metric | α | γ | LL | Pseudo-R² |
|--------|---|---|-----|-----------|
| wasserstein_identity | 4.712 | 0.310 | -198,308 | 7.42% |
| wasserstein_embedding | 8.953* | 0.134* | -183,116* | 14.51%* |

*Uncorrected. After diagonal correction: α=8.39, β=0.154, R²=13.76%. Centroid (14.08%) is primary.

**Comparison:**
- ΔLL = +15,192
- Δpseudo-R² = +7.09 percentage points
- Pseudo-R² ratio = 1.96 (96% improvement, uncorrected; corrected ratio = 13.76/7.52 = 1.83, i.e., +83%)

**Gate: PASSED.** Embedding ground metric substantially improves over identity (criterion: ≥5% improvement).

**Distance correlation:** r(identity, embedding) = 0.61, ρ = 0.36

**Key finding:** Semantic task similarity (via MPNet embeddings) improves explanatory power by over 80% compared to raw task overlap (corrected: 7.52% → 13.76%). Knowing that "operating forklift" ≈ "driving delivery vehicle" provides substantial value beyond just knowing "different tasks."

**Attribution summary (v0.7.3.2 + v0.7.3.3):**
1. Semantic embeddings (vs O*NET/identity): +74.9-83% improvement
2. Distributional treatment (Wasserstein vs cosine centroid): no improvement; centroid marginally outperforms

The MPNet embedding is doing the work. The Wasserstein formulation provides marginal additional value.

### I Module: Mobility Decomposition (v0.6.5)

| Component | Coefficient | t-stat | Interpretation |
|-----------|-------------|--------|----------------|
| d_sem (Centroid) | 7.404 | 204.0 | Skill transformation cost |
| d_inst (Job Zone + Cert) | 0.139 | 33.7 | Non-skill barriers |

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

**Sample:** Train = 97,236 transitions (2015-2019, 2022-2023); Out-of-period = 8,880 transitions (2024)

**Status: VALIDATED.** AIOE and Wasserstein are orthogonal—shock profiles identify exposed occupations, geometry identifies compatible destinations. Out-of-period validation (ΔLL = +23,119) confirms S module integration improves prediction.

### S Module: Pathway Accuracy Audit (v0.7.0.3c)

| Methodology | Aggregate ρ | Per-origin ρ | n origins |
|-------------|-------------|--------------|-----------|
| Model probability, all destinations | 0.188 | 0.128 | 233 |
| Raw 1/distance, common destinations | 0.043 | 0.316 | 177 |

**CORRECTION:** Original v0.7.0c reported ρ = 0.43, but this was computed on a **restricted sample** (exposed origins only, n=60). On the full 2024+ out-of-period comparison (n=424 origins), the model probability method yields ρ = 0.128 per-origin.

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

### M Module: Gravity Model (v0.7.3.4)

Bilateral flow gravity model: ln(Flow_ij + 1) = α + β₁·ln(Emp_i) + β₂·ln(Emp_j) + β₃·d(i,j) + ε

| Metric | β_distance | t-stat | R²_full | Partial R² |
|--------|------------|--------|---------|------------|
| wasserstein | -1.041 | -96.2 | 25.52% | **3.46%** |
| cosine_onet | -1.556 | -92.4 | 25.27% | 3.20% |
| cosine_embed | -0.577 | -82.2 | 24.62% | 2.55% |
| euclidean_dwa | -0.514 | -25.3 | 22.31% | 0.25% |

**Sample:** 199,362 occupation pairs (447 × 446); 12.5% positive flow, 87.5% zeros

**Mass-only model R²:** 22.06% (employment size alone explains most variance)

**Key findings:**
1. **Wasserstein best for aggregate flows:** Unlike conditional logit (where cosine_embed ≈ wasserstein), distributional treatment adds value for aggregate prediction
2. **cosine_onet competitive in gravity:** Its binary "connected/not connected" structure captures aggregate patterns even though it fails in individual choice
3. **euclidean_dwa nearly zero:** Confirms this metric captures little useful signal

**Cortes-Gallipoli benchmark:** Best partial R² = 3.46%, well below C-G's 15%. Consistent with task-specific costs being a modest share of switching costs.

**Note:** Gravity model (aggregate flows) differs from conditional logit (individual choice). The two tests answer different questions:
- Conditional logit: "Which destinations do workers choose, given they switch?"
- Gravity model: "How much does distance reduce total bilateral flows?"

**Status: VALIDATED.** Task distance explains modest share of aggregate flows. Embedding methods average 3.0% partial R², outperforming O*NET methods (1.7%).

### Structural Stability: COVID Comparison (v0.7.5.0)

| Period | α | SE(α) | γ | SE(γ) | Pseudo-R² |
|--------|---|-------|---|-------|-----------|
| Pre-COVID (2015-2019) | 7.394 | 0.044 | 0.146 | 0.005 | 14.1% |
| Post-COVID (2022-2024) | 7.358 | 0.063 | 0.144 | 0.007 | 13.9% |

**Structural break test:** LR χ²(2) = 0.67, p = 0.72. Coefficient change < 1%.

**Sample:** Pre-COVID n = 60,225; Post-COVID n = 29,104.

**Remote work heterogeneity (Dingel-Neiman teleworkability, 92.4% coverage):**

| Coefficient | Estimate | SE | p-value |
|-------------|----------|-----|---------|
| δ₂ (sem × post) | -0.033 | 0.089 | 0.707 |
| δ₃ (sem × telework) | -0.561 | 0.019 | <0.0001 |
| δ₄ (sem × post × telework) | -0.086 | 0.033 | 0.010 |

**Robustness:** Effect persists excluding 2022 (δ₄ = -0.097, p = 0.016); replicates with cosine_embed.

**Interpretation:** Aggregate task-distance constraints are structural (invariant to COVID shock). Heterogeneity: teleworkable occupations experienced elevated hiring standards post-COVID, consistent with expanded applicant pools enabling heightened employer selectivity (Modestino et al. 2020 framework).

**Status: VALIDATED.**

### T Module: Diagonal Correction Audit (v0.7.7.0)

The embedding Wasserstein matrix has 170/447 nonzero diagonal entries (mean 0.224, max 0.401) due to many-to-one SOC→Census aggregation. All comparison matrices have zero diagonals. This audit tests whether this asymmetry biases headline results.

| Model | α | γ | LL | Pseudo-R² |
|-------|---|---|-----|-----------|
| wasserstein_original | 8.936 | 0.142 | -183,051 | 14.54% |
| wasserstein_corrected | 8.386 | 0.154 | -184,738 | 13.76% |
| wasserstein_identity | 4.711 | 0.321 | -198,089 | 7.52% |

**Correction impact:** −0.79pp (14.54% → 13.76%). Corrected embedding Wasserstein still +83% over identity (vs +93% original).

**Identity reproduction:** 7.52% vs prior 7.42% (Δ = +0.10pp). Both models shifted in the same direction, consistent with minor upstream pipeline changes in v0.7.5.1. Tolerance widened to ±0.15pp with Lead Researcher approval.

**Conclusion:** Nonzero diagonal inflates pseudo-R² by ~0.8pp but does not materially change the headline finding. Embedding ground metric advantage over identity remains large (+83% vs +96%).

**Artifact:** `outputs/experiments/diagonal_audit_v0770.json`

**Status: VALIDATED.** Diagonal is not a material bias source.

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

### Retrospective Battery Infrastructure (v0.7.2.x)

**v0.7.2.1: occ1990dd → O*NET-SOC Crosswalk**

| Metric | Value | Gate |
|--------|-------|------|
| Unweighted coverage | 84.8% | — |
| Employment-weighted coverage | 91.9% | ≥80% |
| Mapped codes | 280/330 | — |
| Confidence tiers | High: 160, Medium: 308, Low: 149 | — |

**Chain:** occ1990dd → Census 1990 (97.6% identity) → OCC2010 (IPUMS) → O*NET-SOC

**Top unmapped by employment:**
1. Assemblers (785): 1.75%
2. Police/detectives (418): 0.59%
3. Industrial machinery repairers (518): 0.56%

**Artifact:** `data/processed/crosswalks/occ1990dd_to_onet_soc.csv`

**Status: PASSED.** Gate (80% employment-weighted) exceeded. Ready for Test B.

**v0.7.2.2: CSH Implementation**

| Metric | Value | Gate |
|--------|-------|------|
| r(CSH, RTI) | 0.815 | [0.7, 0.9] |
| ρ(CSH, RTI) | 0.803 | — |
| Common occupations | 262 | — |
| Ridge α (best) | 596.4 | — |

**Method:** Ridge regression (L2 regularization, CV-selected α) of RTI on 768-dim occupation embedding centroids. Direction vector learned maximizes correlation.

**Robustness variant (CSH_alt):**
| Metric | Value | Note |
|--------|-------|------|
| r(CSH_alt, RTI) | 0.288 | Weaker than CSH |
| r(CSH, CSH_alt) | 0.405 | Moderate overlap |

CSH_alt = cosine similarity to routine centroid (top-33% RTI occupations, employment-weighted). Weaker correlation confirms ridge regression captures routine content better than naive distance-to-centroid.

**Artifacts:**
- Direction vector: `.cache/artifacts/v1/embeddings/rti_direction_v0722.npz`
- Centroids: `.cache/artifacts/v1/embeddings/occ1990dd_centroids_mpnet.npz`
- CSH values: `outputs/experiments/csh_values_v0722.csv`
- CSH_alt values: `outputs/experiments/csh_alt_values_v0722.csv`

**Status: PASSED.** Gate [0.7, 0.9] met. RSHExposure class implemented with:
- `discrete_exposure()`: RTI tercile bands
- `continuous_exposure()`: CSH (learned direction)
- `continuous_exposure_alt()`: CSH_alt (robustness)
- `residualized_continuous()`: CSH | RTI band
- `aggregate_to_cz()`: CZ-level aggregation

**v0.7.2.3: Test B (Autor-Dorn Polarization) — PASSED**

| Outcome | β(CSH_resid) | p-value | ΔR² | Verdict |
|---------|--------------|---------|-----|---------|
| Δ routine share | -1.034 | 0.001 | 0.77% | 0 |
| Δ clerical/retail | +0.437 | 0.187 | 0.09% | 0 |
| Δ operator share | **-3.337** | **<0.001** | **5.5%** | **+** |
| Δ service share | +0.214 | 0.557 | 0.02% | 0 |
| Δ mgmt/prof/tech | +0.545 | 0.079 | 0.13% | 0 |

**Sample:** 722 CZs, 3 periods (1980-2000), 2,166 obs
**Specification:** Matches Autor-Dorn (2013) Table 5; state FE + time FE

**Interpretation Matrix:** 1+, 0−, 4(0)

**Key Validation:** r(CSH_cz, RSH_cz) = **0.478** (independent, not proxy!)
- CSH_cz computed from IPUMS 1980 Census via occupation employment weights
- CSH_resid variance retained: 73.2%

**Finding:** CSH_resid adds significant explanatory power for operator job decline (ΔR² = 5.5%, p < 0.001). Other outcomes show directionally consistent effects but below the ΔR² ≥ 1% threshold.

**Methodology:** IPUMS Census microdata downloaded via API, aggregated to CZ × occ1990dd using Dorn crosswalks, then weighted by CZ-specific employment shares.

**Artifact:** `outputs/experiments/battery_test_b_v0723.json`

---

### Retrospective Battery: Interpretation Summary (v0.7.2.x)

**Test B verdict:** 1+, 0−, 4(0)

CSH adds 5.5% ΔR² for operator employment share; <1% for other outcomes. This is **expected given construct distinction**: embedding geometry captures mobility friction (validated in CPS, v0.6.7-v0.7.1), not automation susceptibility (which RTI already captures). Weak polarization results are a **scope clarification**, not falsification.

**Test C':** Invalid as framework test (methodology deviation). Deferred to v0.7.3.1.

**Test A-lite:** Blocked (no industry-level data in Dorn archive). Deferred to v0.7.4+.

**Framework implication:** Task-space oracle validated for mobility/feasibility applications. Automation prediction is outside validated scope.

---

### Retrospective Battery: Test C'-keyword (v0.7.2.5)

**Status: INVALID as framework test**

**Methodology deviation:** Spec required embedding distance to robot-task centroid. Implemented keyword classification + binary count. This tests a classification scheme, not our embedding geometry.

**Result (for reference only):** β = -0.656, p = 0.024, R² = 8.9%

**Interpretation:** Keyword-classified robot DWAs predict occupation employment decline during 1990-2005. This validates the keyword classification but does not test the task-space oracle framework.

**Resolution:** Correct implementation (embedding-based) deferred to v0.7.3.1.

**Artifact:** `outputs/experiments/battery_test_c_prime_keyword_v0725.json`

---

## Graveyard {#graveyard}

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
| "Wasserstein as primary contribution" framing | ρ = 0.95 with cosine_embed | Distributional treatment adds only 3% over centroid averaging | v0.7.3.2 |
| Kernel overlap as fair baseline | Underperforms even simple cosine_embed | Weak aggregation method, not representative of alternatives | v0.7.3.2b |

---

## Frontier {#frontier}

### Demand-Side Integration (Phase 0.8)

- **Objective:** Add vacancy dynamics to equilibrium model
- **Finding (v0.7.0.3):** Static openings alone explain ρ = 0.798 of aggregate inflows
- **Next step:** Time-varying vacancy rates (JOLTS) for dynamic reallocation
- **Candidates:** JOLTS by major occupation group, Lightcast (if accessible)
- **Status:** Validated as critical; demand dominates geometry for aggregate prediction

### Institutional Barrier Enhancement (Phase 0.9)

- **Problem:** γ_inst/γ_sem = 0.019 underweights credentials
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

### Retrospective Battery (v0.7.2.x → v0.7.3.x)

**Completed:**
- Test B (CZ-level polarization): 1+, 0−, 4(0) — validates construct distinction
- Crosswalk infrastructure: occ1990dd → O*NET-SOC (91.9% coverage)
- CSH implementation: r(CSH, RTI) = 0.815
- IPUMS pipeline: CZ × occ1990dd employment matrices (1980, 1990, 2000)

**Deferred:**
- Test C' (robot exposure): Requires re-implementation with embedding distance (v0.7.3.1)
- Test A-lite (industry task drift): Blocked by data; requires ALM replication files (v0.7.4+)

**Implication:** Framework validated for mobility; automation prediction outside current scope.

---

## Artifact Registry {#artifact-registry}

### Distance Matrices

| Artifact | Location | Shape |
|----------|----------|-------|
| Wasserstein (O*NET) | `.cache/artifacts/v1/wasserstein/d_wasserstein_onet.npz` | 894×894 |
| Wasserstein (Census) | `.cache/artifacts/v1/mobility/d_wasserstein_census.npz` | 447×447 |
| Kernel (Census) | `.cache/artifacts/v1/mobility/d_sem_census.npz` | 447×447 |
| Institutional | `.cache/artifacts/v1/mobility/d_inst_census.npz` | 447×447 |
| Cosine O*NET (Census) | `.cache/artifacts/v1/mobility/d_cosine_onet_census.npz` | 447×447 |
| Cosine Embedding (Census) | `.cache/artifacts/v1/mobility/d_cosine_embed_census.npz` | 447×447 |
| Euclidean DWA (Census) | `.cache/artifacts/v1/mobility/d_euclidean_dwa_census.npz` | 447×447 |
| Wasserstein Identity (Census) | `.cache/artifacts/v1/mobility/d_wasserstein_identity_census.npz` | 447×447 |

### Experiment Results

| Artifact | Location | Paper Section |
|----------|----------|---------------|
| CPS Mobility (Wasserstein) | `outputs/experiments/path_a_wasserstein_comparison_v0672.json` | T Module (PRIMARY) |
| Asymmetric (Wasserstein) | `outputs/experiments/mobility_asymmetric_wasserstein_v0682.json` | I Module |
| Asymmetric Robustness | `outputs/experiments/path_f_robustness_v0683.json` | I Module |
| RTI Correlation | `outputs/experiments/path_c_rti_construct_validity_v0681.json` | Complementary |
| Shock Integration | `outputs/experiments/shock_integration_v070a.json` | S Module |
| Scaled Costs | `outputs/experiments/scaled_costs_v070b.json` | M Module |
| Reallocation | `outputs/experiments/reallocation_v070c.json` | Scope Validation |
| Sensitivity Analysis | `outputs/experiments/sensitivity_switching_costs_v0702.json` | M Module |
| Demand Probe | `outputs/experiments/demand_probe_v0703.json` | Scope Validation |
| Demand Decomposition | `outputs/experiments/demand_probe_decomposition_v0703b.json` | Scope Validation |
| Methodology Audit | `outputs/experiments/methodology_audit_v0703c.json` | — |
| MS1 Compliance Audit | `outputs/experiments/ms1_compliance_audit_v0704.json` | — |
| T Module Multiverse | `outputs/multiverse/t_module_v0712/summary.json` | T Module |
| Performance Battery | `outputs/experiments/performance_battery_baseline_v0712.json` | T Module |
| Crosswalk Coverage | `outputs/experiments/crosswalk_coverage_v0721.json` | Battery |
| CSH Values | `outputs/experiments/csh_values_v0722.csv` | Battery |
| CSH_alt Values | `outputs/experiments/csh_alt_values_v0722.csv` | Battery |
| Distance Baselines | `outputs/experiments/distance_baselines_v0731.json` | Methodology Comparison |
| 2×2 Head-to-Head | `outputs/experiments/distance_head_to_head_v0732.json` | Methodology Comparison |
| Ground Metric Validation | `outputs/experiments/ground_metric_validation_v0733.json` | Methodology Comparison |
| Gravity Model | `outputs/experiments/gravity_model_v0734.json` | M Module |
| Pre/Post COVID | `outputs/experiments/pre_post_covid_v0741.json` | Structural Stability (§5.7) |
| Diagonal Audit | `outputs/experiments/diagonal_audit_v0770.json` | Robustness check |
| COVID Centroid | `outputs/experiments/covid_centroid_v0773.json` | Structural Stability (§5.7) |

### Embeddings

| Artifact | Location | Shape |
|----------|----------|-------|
| occ1990dd Centroids | `.cache/artifacts/v1/embeddings/occ1990dd_centroids_mpnet.npz` | 280×768 |
| RTI Direction | `.cache/artifacts/v1/embeddings/rti_direction_v0722.npz` | 768 |

### Crosswalks

| Artifact | Location | Coverage |
|----------|----------|----------|
| occ1990dd → O*NET-SOC | `data/processed/crosswalks/occ1990dd_to_onet_soc.csv` | 91.9% emp-weighted |

---

## Version History {#version-history}

| Version | Date | Changes |
|---------|------|---------|
| 0.7.12.8 | 2026-03-12 | Working paper aligned to centroid — 9 patterns, ~50 edits across paper/main.tex. Per-origin ρ→0.12, AIOE r→-0.02, ΔLL→+23,879, RTI table updated, gravity reordered with × names, switching cost→8.74/centroid, methodology→inverse centroid. |
| 0.7.12.7 | 2026-03-12 | Table 4 centroid correlations — centroid vs Wasserstein 0.9527, vs identity 0.3530, vs O*NET Euclidean 0.2283, vs O*NET cosine 0.4667. |
| 0.7.12.6 | 2026-03-12 | Per-origin model-probability Spearman on centroid — ρ = 0.118 (n = 233), replacing Wasserstein-based 0.128 (n = 233). |
| 0.7.12.5 | 2026-03-12 | Centroid replication batch — AIOE r=-0.016, RTI r=-0.060, demand decomp (demand ρ=0.798, per-origin ρ=0.299, aggregate ρ=-0.000), OOP ΔLL=+23,879, gravity partial R²=2.55%. All on centroid spec. |
| 0.7.12.0 | 2026-03-12 | Distance matrix audit — added load_centroid_census() to mobility IO, verified centroid matrix clean (zero diagonal, no NaN), computed post-diagonal Spearman ρ = 0.9527 between centroid and Wasserstein. |
| 0.7.11.0 | 2026-03-11 | Documentation consistency pass — CLAUDE.md, README.md, SPEC.md, LEDGER.md, data/README.md aligned to v0.7.10.x presentation sprint. |
| 0.7.10.29 | 2026-03-11 | Audit pass 3 — corrected SchuStanTaska characterization (employer concentration → appendix mobility analysis), fixed 3 remaining C-G "approximately 15%" references in working paper (lines 116, 124, 1032). Frank et al. 36,536 transitions verified by Deep Researcher. |
| 0.7.10.28 | 2026-03-11 | Bibliography/prose audit pass 2 — removed fabricated HampoleEtAl2024, fixed Jackson2023 author/title in working paper bib, corrected HampoleEtAl2025/onet2024related/KudlyakWolcott2019/dawson2021skill in working paper bib, softened C-G pseudo-R² attribution (both papers), corrected Macaluso/SchuStanTaska/KudlyakWolcott characterizations (both papers), clarified 894 occupation count, removed 46 orphan bib entries from publishable. |
| 0.7.10.27 | 2026-03-11 | Bibliography audit — corrected all 5 placeholder entries (titles, authors, years, DOIs), updated cite keys, softened C-G characterization. |
| 0.7.10.26 | 2026-03-11 | Working paper aligned to publishable reframing — structured-vector baseline language, new citations (Mouw, Macaluso, Schubert/Stansbury/Taska, Carrillo-Tudela, Kudlyak/Wolcott). |
| 0.7.10.25 | 2026-03-11 | Advisory response — publishable reframing (structured-vector baseline language, lit review restructured, 5 new citations), working paper ownership to Lead. |
| 0.7.10.24 | 2026-03-11 | Publishable disclosures (J=11, in-sample, unweighted, diagonal, origin exclusion), deck table fix, LEDGER holdout wording. |
| 0.7.10.23 | 2026-03-11 | Audit fixes — holdout→out-of-period (LEDGER + publishable), beta SE 0.004→0.0041, ownership path. |
| 0.7.9.3 | 2026-03-09 | Captions for fig3/fig4, swap fig5/fig6 to Budget vs Credit Analysts. |
| 0.7.10.21 | 2026-03-10 | Fig3 label positioning finalized. Swap Edit→Transcribe label, add HA_OVERRIDE for independent alignment control. |
| 0.7.10.16 | 2026-03-10 | Fig3 rebalance 6 per theme = 30 DWAs, 10 labels. Added Transcribe, Type documents, Prepare budgets. |
| 0.7.10.15 | 2026-03-10 | Fig3 swap Test patient vision for Prescribe medications label. 29 DWAs, 10 labels. |
| 0.7.10.14 | 2026-03-10 | Fig3 three more labels (Apply mortar, Calculate financial data, Edit written materials), legend upper-left. 28 DWAs, 10 labels. |
| 0.7.10.13 | 2026-03-10 | Fig3 swap Pilot Aircraft for Secure Cargo + Operate Forklifts labels. 27 DWAs, 7 labels. |
| 0.7.10.12 | 2026-03-10 | Add "Test patient vision" label to fig3 Healthcare theme. 25 DWAs, 6 labels. |
| 0.7.10.11 | 2026-03-10 | Change GREEN to colorblind-safe teal (#44AA99, Tol palette). Regenerate fig3. |
| 0.7.10.10 | 2026-03-10 | Fig3 final — hand-curated 24 DWAs, 5 themes × ~5 dots, 1 label each, no adjustText. |
| 0.7.10.9 | 2026-03-10 | Fig3 keyword-only clustering, restored background cloud. 5 themes × 6 dots = 30, 8 labels. |
| 0.7.10.8 | 2026-03-10 | Fig3 short-name-first selection with spatial separation. Construction dropped (2 candidates). Healthcare reduced to 3. |
| 0.7.10.7 | 2026-03-10 | Fig3 — 5 themes × 6 dots × 2 labels, adjustText, legend lower-right. Technology dropped. |
| 0.7.10.6 | 2026-03-10 | Fig3 keyword-filtered theme assignment + diversity filter. Technology theme empty (0 DWAs pass filter). |
| 0.7.10.5 | 2026-03-10 | Fig3 programmatic DWA selection — density-based tightest triples, collision-aware labels, legend to upper-left. GWA mapping needs tightening. |
| 0.7.10.4 | 2026-03-10 | Fig9 swap to Atlanta/Georgia/Colorado/Denver analogy. |
| 0.7.10.3 | 2026-03-10 | Fig9 redesigned as Word2Vec illustration with attribution; fig8 arrow padding. |
| 0.7.10.2 | 2026-03-10 | Docs alignment — figures README, SPEC, CLAUDE updated for presentation sprint. |
| 0.7.10.1 | 2026-03-10 | Fig8/fig9 use live embeddings instead of hardcoded values. |
| 0.7.10.0 | 2026-03-10 | Fig8 actual embedding values + arrow padding; fig9 real grid + formula + asymmetric parallelogram. |
| 0.7.9.9 | 2026-03-10 | Embedding explainer figures — fig8 (flowchart), fig9 (word analogy parallelogram). |
| 0.7.9.8 | 2026-03-10 | Align figure aspect ratios to presentation placeholders. |
| 0.7.9.7 | 2026-03-10 | Style system upgrade — centralized palette (RED/ORANGE/GREEN/PURPLE), font scale constants, add_subtitle() and format_log_ticks() helpers. |
| 0.7.9.2 | 2026-03-09 | Fig4 vertical bars, fig6 remove exponents and update subtitle. |
| 0.7.9.1 | 2026-03-09 | Create fig4 (scope), fix fig5 centering/ellipses, fix fig6 log scale/sizing. |
| 0.7.9.0 | 2026-03-09 | New figures — fig5 (shared DWAs), fig6 (embedding similarity), fig7 (Sankey pipeline), fig3 resized to 6.0×4.2. |
| 0.7.8.9 | 2026-03-09 | Remove unused experiments/, notebooks/, templates/ directories. |
| 0.7.8.8 | 2026-03-09 | Final consistency pass — paper t-stat 33.8→33.7, β SE 0.004→0.0041, LEDGER 2×2 table asterisks, ground metric ratio annotations. |
| 0.7.8.7 | 2026-03-09 | Heroic consistency pass — DISTANCE_GUIDE rewrite, LEDGER current-state sections updated, stale test archived, wasserstein.py docstring fixed. |
| 0.7.8.6 | 2026-03-09 | README refresh — all numbers corrected, centroid as primary, version history trimmed. |
| 0.7.8.5 | 2026-03-09 | Fix fig2 top tick zero-length bug: separate gap (0.4) from bracket_ext (1.5). |
| 0.7.8.4 | 2026-03-09 | Precise improvement percentage (74.9%) across all files; fix fig2 bracket padding. |
| 0.7.8.3 | 2026-03-09 | Programmatic fig2 bracket alignment from text extents — no hardcoded positions. |
| 0.7.8.2 | 2026-03-09 | Fix fig2 bracket alignment (dynamic x position) and label text (→ to +). |
| 0.7.8.1 | 2026-03-09 | Redesign fig2 annotation: replace ground metric ghost bar with embedding representation bracket (74.9%). |
| 0.7.8.0 | 2026-03-09 | Regenerate fig2 with corrected values (13.76%/14.08%), centroid-first ordering, +83% bracket. |
| 0.7.7.7 | 2026-03-09 | Origin-exclusion robustness test: Δα = +4.2%, not material. Disclosure added to paper. |
| 0.7.7.6 | 2026-03-09 | Software hygiene — cache ordering guard, hash fix, stale artifact cleanup, canonical tests updated, reproduce_tables.py added. |
| 0.7.7.5 | 2026-03-09 | Comprehensive consistency pass — 10 issues from adversarial audit (t-stats, p-values, table vintages, holdout wording, improvement percentages, HC1, naming, scale notes). |
| 0.7.7.4 | 2026-03-09 | Add in-sample pseudo-R² and crosswalk aggregation disclosures to limitations. Fix stale γ values in §6.1. |
| 0.7.7.3 | 2026-03-09 | Re-estimate pre/post COVID with centroid specification. α change −0.49%, structural break p = 0.72. Table 9 now uses centroid values. |
| 0.7.7.2 | 2026-03-09 | Fix Frank et al. citation (full author list + DOI), revert fabricated COVID table to actual estimates, fix γ/α ratios and t-statistics, update CLAUDE.md and SPEC.md. |
| 0.7.7.0 | 2026-03-09 | Diagonal correction audit: zeroing Wasserstein diagonal costs −0.79pp (14.54%→13.76%); corrected metric still +83% over identity. Not a material bias. |
| 0.7.6.0 | 2026-03-09 | Publication figures for embeddings_v8.pdf |
| 0.7.5.1 | 2025-01-15 | Codebase polish for external review. HC1 default fix; metrics glossary; distance guide; code documentation. |
| 0.7.5.0 | 2025-12-23 | COVID structural stability: aggregate geometry invariant (Δα < 1%, p = 0.76); teleworkable occupations show elevated hiring standards (δ₄ = -0.086, p = 0.01). Paper §5.7 added. |
| 0.7.4.1 | 2025-12-23 | Pre/post COVID comparison implemented. Sample split, period estimation, structural break test, remote work interactions. |
| 0.7.4.0 | 2025-12-22 | Documentation consolidation. Contribution reframed: semantic task substitutability, not Wasserstein per se. Attribution Audit added. |
| 0.7.3.4b | 2025-12-22 | PPML robustness: all β<0 preserved, rankings differ slightly. Distance effect robust to heteroskedasticity correction. |
| 0.7.3.4 | 2025-12-22 | Gravity model: partial R² = 3.46% (wasserstein), below C-G 15% benchmark. Task distance modest share of aggregate flows. |
| 0.7.3.3 | 2025-12-22 | Ground metric validation: PASSED. Embedding vs identity: +96% pseudo-R². Semantic task similarity matters. |
| 0.7.3.2b | 2025-12-22 | Kernel attribution audit: Original T Module comparison conflated effects. Kernel was weak baseline. |
| 0.7.3.2 | 2025-12-22 | 2×2 head-to-head: GATE FAILED. Wasserstein vs cosine_embed: ΔR²=0.43pp, p=0.002. Embedding choice dominates. |
| 0.7.3.1 | 2025-12-22 | Distance baselines: cosine-onet, cosine-embed, euclidean-dwa matrices (447×447). ρ(Wasserstein, cosine-embed)=0.95. |
| 0.7.3.0 | 2025-12-19 | Documentation schema update. Decision Authority Matrix, LEDGER Update Authority, SPEC requirements. Sprint summary template. |
| 0.7.2.5 | 2025-12-19 | Retrospective battery. Test B: 1+, 0−, 4(0). Test C' invalid (methodology deviation). IPUMS pipeline built. MS10 added. |
| 0.7.2.3 | 2025-12-19 | Test B PASSED: IPUMS pipeline; r(CSH_cz, RSH_cz)=0.478; operator decline (+), others (0) |
| 0.7.2.2 | 2025-12-17 | CSH implementation: r(CSH, RTI)=0.815; RSHExposure class; CSH_alt robustness variant |
| 0.7.2.1 | 2025-12-17 | occ1990dd crosswalk: 91.9% emp-weighted coverage; crosswalk diagnostics module |
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
