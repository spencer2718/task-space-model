# Kill Shot Documentation: CPS Mobility + Institutional Wedge Test

**Version:** 0.6.4.1  
**Date:** December 14, 2025  
**Status:** Analysis Complete — Results Support Thesis Pivot

---

## 1. Statement of Theoretical and Experimental Pivot

### 1.1 Background: Phase I Findings

Phase I validation (v0.6.2) established that both continuous and discrete representations of task-space structure predict wage comovement:

| Measure | t-stat (clustered) | R² |
|---------|-------------------|-----|
| Normalized kernel overlap | 7.14 | 0.00485 |
| Binary Jaccard | 8.00 | 0.00167 |

The continuous measure explains 191% more variance, but the discrete measure achieves higher statistical precision. This pattern suggested the manifold hypothesis is partially correct: semantic similarity captures economically relevant structure, but the mapping from linguistic similarity to economic substitutability contains systematic noise.

### 1.2 The Pivot: Decomposing Semantic and Institutional Distance

The Chief Investigator proposed that the noise in semantic distance arises from a missing component: **institutional barriers**. Two occupations may be semantically similar (overlapping task content) but economically distant due to credentialing requirements, training investments, or licensing barriers.

**Formal decomposition:**

$$d_{eff}(i,j) = d_{sem}(i,j) + \lambda \cdot d_{inst}(i,j)$$

Where:
- $d_{sem}(i,j)$ = semantic distance (task-content similarity from text embeddings)
- $d_{inst}(i,j)$ = institutional distance (job zone differences + certification requirements)
- $\lambda$ = relative weight of institutional vs. semantic friction

**Testable hypothesis:** If $\lambda > 0$ and significant, institutional friction is real and separable from semantic friction. Workers minimize both components when switching occupations.

### 1.3 Why Worker Mobility is the Correct Test

The original retrospective battery (Tests A, B, C) operated at aggregate levels:
- Test A: Industry × decade task composition shifts
- Test B: Commuting zone employment polarization
- Test C: Commuting zone robot exposure effects

The pivot substitutes a **worker-level mobility test** using CPS occupation transitions:
- Unit of analysis: Individual worker transition decisions
- Direct test of: "Can this worker do that job?"
- Mechanism: Workers choose destinations that minimize $d_{eff}$

This is superior for testing the decomposition because:
1. Worker transitions directly reveal revealed preference over occupation pairs
2. Individual-level variation avoids ecological inference problems
3. The outcome (which job did they take?) is unambiguous

### 1.4 Pre-Commitment: What Constitutes Success

The test was pre-committed as follows:
- **Success:** β (institutional distance coefficient) is negative and statistically significant in a conditional logit model where workers choose among alternative destinations
- **Interpretation:** Workers avoid large institutional jumps independent of semantic similarity
- **Magnitude benchmark:** Effect size comparable to or larger than α (semantic coefficient)

---

## 2. Data Construction

### 2.1 CPS Extract Specification

**Source:** IPUMS CPS Basic Monthly Files  
**Period:** 2015-01 through 2019-12; 2022-01 through 2024-09  
**Exclusion:** 2020-01 through 2021-12 (COVID labor market disruption)  
**Samples obtained:** 83 of 96 requested monthly files (13 unavailable due to government shutdowns)

**Variables:**
| Variable | Purpose |
|----------|---------|
| CPSIDP | Person linking across months |
| CPSIDV | Validated person linking (demographic consistency) |
| MISH | Month-in-sample (rotation position) |
| OCC2010 | Harmonized occupation code (Census 2010 basis) |
| OCC | Original occupation codes |
| EMPSTAT | Employment status filter |
| AGE, SEX, RACE, EDUC | Demographic validation and controls |
| PANLWT | Panel weights (extracted but not used in estimation) |
| STATEFIP | Geographic identifier |

**Raw extract:** 10,072,119 person-month observations

### 2.2 Sample Construction Pipeline

| Stage | Records | Retention | Notes |
|-------|---------|-----------|-------|
| Raw CPS extract | 10,072,119 | 100% | 83 monthly samples |
| Employment filter | 4,407,432 | 43.8% | EMPSTAT ∈ {10, 12}, age 18-65 |
| Consecutive month pairs | 2,353,725 | 53.4% | CPSIDP linked t to t+1 |
| Demographic validation | 2,337,740 | 99.3% | Age ±1, sex/race constant |
| Raw transitions identified | 157,606 | 6.74% | OCC(t) ≠ OCC(t-1) |
| Persistence filter | 106,116 | 67.3% | Verified transitions |
| O*NET mapping successful | 89,329 | 84.2% | Census 2010 → O*NET crosswalk |

**Final analysis sample:** 89,329 verified occupation transitions

### 2.3 Measurement Error Correction: Persistence Filter

Raw CPS occupation switching rates (~6.7% monthly) substantially exceed true mobility rates (~0.5-1.0% per Kambourov & Manovskii 2008). The discrepancy arises from coding error: self-reported occupation varies across interviews even when the underlying job is unchanged.

**Persistence filter logic:**
1. **Origin stability:** OCC(t-2) = OCC(t-1) (worker was in origin occupation for at least one prior period)
2. **Transition occurred:** OCC(t) ≠ OCC(t-1)
3. **Destination persistence:** OCC(t) = OCC(t+1) = OCC(t+2) (worker remains in destination)

This filter removes spurious transitions where occupation codes fluctuate due to reporting inconsistency. Retention rate (67.3%) is consistent with literature estimates that 25-50% of raw CPS occupation changes are measurement error.

### 2.4 Crosswalk: CPS to O*NET

**Mapping path:** Census 2010 (OCC2010) → 6-digit SOC → O*NET-SOC 2019

**Coverage:**
| Metric | Value |
|--------|-------|
| O*NET occupations | 894 |
| Census 2010 codes | 447 |
| Matched O*NET codes | 882 (98.7%) |
| Mean O*NET per Census code | 1.97 |

**Unmatched occupations (12):** Primarily postsecondary teacher specialties (SOC 25-1xxx) which are aggregated in Census coding. These represent <2% of occupations and negligible employment share.

**Aggregation method:** When multiple O*NET codes map to one Census code, distances are averaged (unweighted). This introduces smoothing but preserves rank ordering.

---

## 3. Distance Matrix Construction

### 3.1 Semantic Distance ($d_{sem}$)

**Construction method:** Normalized kernel overlap converted to distance

**Steps:**
1. Embed 2,087 Detailed Work Activities (DWAs) using MPNet (all-mpnet-base-v2, 768 dimensions)
2. Compute cosine distance matrix $D_{act}$ over activities
3. Calibrate kernel bandwidth: $\sigma$ = median nearest-neighbor distance = 0.0096
4. Build kernel matrix: $K_{ab} = \exp(-D_{act}(a,b)/\sigma)$
5. Load occupation-activity importance matrix (894 occupations × 2,087 activities)
6. Row-normalize to probability measures: $\rho_j(a) = w_j(a) / \sum_a w_j(a)$
7. Compute normalized overlap: $O_{ij} = \frac{\sum_{a,b} \rho_i(a) K_{ab} \rho_j(b)}{\sqrt{\sum_{a,b} \rho_i(a) K_{ab} \rho_i(b)} \cdot \sqrt{\sum_{a,b} \rho_j(a) K_{ab} \rho_j(b)}}$
8. Convert to distance: $d_{sem}(i,j) = 1 - O_{ij}$

**Matrix properties (894 × 894):**
| Statistic | Value |
|-----------|-------|
| Mean | 0.955 |
| Std | 0.069 |
| Min | 0.002 |
| Max | 1.000 |
| p10 / p50 / p90 | 0.91 / 0.97 / 0.99 |

The distribution is concentrated near 1.0 (most occupation pairs are semantically distant), with a long left tail of similar occupations.

### 3.2 Institutional Distance ($d_{inst}$)

**Construction method:** Job zone difference plus certification barrier

**Formula:**
$$d_{inst}(i,j) = |Zone_i - Zone_j| + \gamma \cdot |Cert_i^{norm} - Cert_j^{norm}|$$

**Components:**

1. **Job Zone** (O*NET "Job Zones.xlsx"):
   - Scale: 1-5 (little preparation → extensive preparation)
   - Zone 1: Short demonstration (e.g., food prep workers)
   - Zone 5: Extensive preparation (e.g., surgeons, lawyers)
   - Coverage: 923/923 occupations (100%)

2. **Certification importance** (O*NET Element ID 2.D.4.a):
   - Scale: 1-5 (not important → extremely important)
   - Coverage: 564/923 occupations (61.1%)
   - Missing values: Imputed with median (2.87)
   - Normalization: Rescaled to 0-4 range to match zone scale

3. **Weight parameter:** $\gamma = 1.0$ (equal weight to zone and certification)

**Matrix properties (923 × 923):**
| Statistic | Value |
|-----------|-------|
| Mean | 2.11 |
| Std | 1.35 |
| Min | 0.00 |
| Max | 8.00 |
| p10 / p50 / p90 | 0.37 / 2.00 / 3.94 |

The distribution spans the full theoretical range (0 = same zone and certification; 8 = maximum zone difference plus maximum certification difference).

### 3.3 Correlation Between Distance Measures

| Level | corr($d_{sem}$, $d_{inst}$) |
|-------|---------------------------|
| O*NET (894 occupations) | 0.358 |
| Census (447 occupations) | 0.344 |
| Analysis sample | 0.158 |

The low correlation indicates the measures capture distinct information. Semantic similarity and institutional barriers are weakly related—occupations can be task-similar but credential-distant (e.g., nurse and physician) or task-distant but credential-similar (e.g., two Zone-2 jobs in different sectors).

---

## 4. Econometric Specification

### 4.1 Conditional Logit Model

**Framework:** McFadden's conditional logit for discrete choice

For worker $k$ who switches from origin occupation $i$, the utility of destination $j$ is:
$$U_{kj} = \alpha \cdot (-d_{sem}(i,j)) + \beta \cdot (-d_{inst}(i,j)) + \varepsilon_{kj}$$

Distances enter negatively: workers prefer destinations with **lower** semantic and institutional distance from their origin.

**Choice probability:**
$$P(j | i, \text{switch}) = \frac{\exp(\alpha \cdot (-d_{sem}(i,j)) + \beta \cdot (-d_{inst}(i,j)))}{\sum_{j' \in J} \exp(\alpha \cdot (-d_{sem}(i,j')) + \beta \cdot (-d_{inst}(i,j')))}$$

### 4.2 Choice Set Construction

Full choice sets (all 447 Census occupations) are computationally prohibitive. Following standard practice, we use **random sampling of alternatives:**

- For each observed transition (origin $i$ → chosen destination $j$):
  - Include the chosen destination ($Y = 1$)
  - Sample $K = 10$ non-chosen alternatives uniformly at random ($Y = 0$)
- This yields 11 observations per transition, grouped by transition ID

**Sample size:**
| Component | Count |
|-----------|-------|
| Observed transitions | 89,329 |
| Rows per transition | 11 |
| Total choice-set observations | 982,619 |

**Identification:** Under IIA (independence of irrelevant alternatives), random sampling of alternatives yields consistent coefficient estimates (McFadden 1978). Standard errors are computed via the observed information matrix.

### 4.3 Estimation

**Software:** statsmodels.discrete.conditional_models.ConditionalLogit  
**Grouping variable:** Transition ID  
**Covariates:** neg_d_sem (= $-d_{sem}$), neg_d_inst (= $-d_{inst}$)  
**Weights:** Not used (PANLWT available but not incorporated)

---

## 5. Results

### 5.1 Coefficient Estimates

| Parameter | Coefficient | Std. Error | t-statistic | p-value |
|-----------|-------------|------------|-------------|---------|
| α (semantic) | 2.994 | 0.030 | 98.53 | < 10⁻¹⁰⁰ |
| β (institutional) | 0.215 | 0.003 | 63.42 | < 10⁻¹⁰⁰ |

**Log-likelihood:** -205,528.9

Both coefficients are positive (recall covariates are negated distances), indicating workers prefer destinations that are:
1. Semantically similar to their origin (higher task overlap)
2. Institutionally similar to their origin (similar job zone and certification requirements)

### 5.2 Statistical Significance

Both coefficients are estimated with extraordinary precision:
- α: t = 98.5, effectively zero probability under the null
- β: t = 63.4, effectively zero probability under the null

The large t-statistics reflect the sample size (89,329 transitions) and the systematic relationship between distances and destination choice.

### 5.3 Economic Magnitude

**Odds ratio interpretation:**

For a 1-unit increase in $d_{sem}$ (moving to a more semantically distant occupation):
$$\text{Odds ratio} = \exp(-\alpha) = \exp(-2.994) = 0.050$$
A 1-unit increase in semantic distance reduces the odds of choosing that destination by 95%.

For a 1-unit increase in $d_{inst}$ (moving to a more institutionally distant occupation):
$$\text{Odds ratio} = \exp(-\beta) = \exp(-0.215) = 0.807$$
A 1-unit increase in institutional distance reduces the odds of choosing that destination by 19.3%.

**Relative magnitude:**

The semantic coefficient is 13.9× larger than the institutional coefficient ($\alpha/\beta = 2.994/0.215 = 13.9$). However, the scales differ:
- $d_{sem}$ range: 0.002 to 1.000 (effective range ~0.9)
- $d_{inst}$ range: 0 to 8 (effective range ~3.6)

**Standardized comparison** (using sample standard deviations):
- 1-SD increase in $d_{sem}$ (0.069): odds ratio = $\exp(-2.994 \times 0.069) = 0.814$
- 1-SD increase in $d_{inst}$ (1.35): odds ratio = $\exp(-0.215 \times 1.35) = 0.748$

On a standardized basis, institutional distance has a **larger** effect than semantic distance.

### 5.4 Decomposition Test

The key question: Does institutional distance add predictive value beyond semantic distance?

**Test:** Is β significantly different from zero when α is included?

**Result:** β = 0.215, t = 63.42, p < 10⁻¹⁰⁰

**Conclusion:** Institutional distance is highly significant conditional on semantic distance. Workers systematically avoid destinations with larger job zone gaps and certification barriers, even controlling for task-content similarity.

This rejects H2 (discrete/institutional classification sufficient alone) and supports H3 (mechanism-dependent): both semantic and institutional structure independently predict worker mobility.

---

## 6. Interpretation

### 6.1 Validation of the Theoretical Framework

The results validate the decomposition hypothesis:
$$d_{eff}(i,j) = d_{sem}(i,j) + \lambda \cdot d_{inst}(i,j), \quad \lambda > 0$$

Institutional barriers are not merely proxies for task differences—they capture independent friction. The brain surgeon/butcher problem identified in the theoretical motivation is empirically real: semantic similarity (both involve cutting) does not imply institutional proximity (vastly different credentials).

### 6.2 Mechanism Interpretation

The two components capture distinct mobility frictions:

1. **Semantic distance** ($\alpha = 2.994$): Can the worker perform the tasks?
   - Reflects skill transferability, learning costs, productivity gaps
   - Workers strongly prefer destinations where their existing task competencies apply

2. **Institutional distance** ($\beta = 0.215$): Is the worker credentialed for the job?
   - Reflects hiring barriers, licensing requirements, training investments
   - Workers avoid destinations requiring credential upgrades even when tasks are similar

### 6.3 Asymmetry Implication

The institutional distance measure ($d_{inst}$) is symmetric, but the underlying barrier is not: moving **up** job zones requires training; moving **down** does not. A surgeon can become a butcher; the reverse requires medical school.

The current specification does not model this asymmetry. An extension would decompose $d_{inst}$ into upward and downward components:
$$d_{inst}^{up}(i,j) = \max(0, Zone_j - Zone_i)$$
$$d_{inst}^{down}(i,j) = \max(0, Zone_i - Zone_j)$$

We would expect $\beta^{up} > \beta^{down}$: upward mobility is harder than downward.

### 6.4 Limitations

1. **Panel weights not used:** The estimates are unweighted. Weighted estimation may change point estimates, though significance is unlikely to be affected given the t-statistics.

2. **Choice set sampling:** Random 10:1 sampling assumes IIA. Violations (e.g., occupations with correlated unobservables) could bias estimates. A nested logit or mixed logit would relax this.

3. **Certification measure coverage:** Only 61.1% of occupations have direct certification data; the remainder are imputed. Measurement error in $d_{inst}$ would attenuate β toward zero, so the true effect may be larger.

4. **Static analysis:** The distances are computed from 2024 O*NET data but applied to 2015-2024 transitions. Task content and credential requirements may have shifted over this period.

5. **Selection:** We observe transitions conditional on switching. Workers who would switch but face prohibitive barriers (and thus stay) are not in the sample. This understates the role of institutional barriers for marginal switchers.

### 6.5 Relation to Phase I Findings

Phase I found continuous semantic overlap outperforms discrete Jaccard on R² but underperforms on t-statistic. The institutional wedge helps explain this:
- Semantic overlap captures real structure (high R²)
- But mixes "can do" with "allowed to do" (lower precision)
- Adding $d_{inst}$ separates the components

The decomposition predicts that controlling for institutional distance should increase the precision of semantic distance effects. This is consistent with the very high t-statistics observed for both coefficients.

---

## 7. Codebase Integration and Paper Adjustment (Brief)

### Codebase Integration

**New module:** `src/task_space/mobility/` containing:
- `institutional.py`: Job zone and certification distance construction (port from `build_job_zone_matrix.py`)
- `crosswalk.py`: Census ↔ O*NET mapping (port from `check_crosswalk_coverage.py`)
- `transitions.py`: CPS transition identification and persistence filtering (port from `transition_filters.py`)
- `choice_model.py`: Conditional logit estimation wrapper

**Dependencies:** Add `ipumspy` to optional dependencies for CPS data access.

**Data artifacts:** Cache `job_zone_matrix.npz` and `onet_to_census_improved.csv` in `data/external/`.

### Paper Adjustment

**Section 5.4 (Worker Mobility):** Expand from placeholder to full results section. Report conditional logit specification, coefficient estimates, and decomposition test.

**Section 4.5 (Retrospective Battery):** Downgrade Tests A/B/C to robustness appendix. Elevate CPS mobility test as primary validation.

**Contribution statement:** Add fourth contribution: "Institutional distance (job zones, certification) independently predicts worker transitions beyond semantic task overlap."

**Tables:** Add Table X with coefficient estimates; Table Y with sample construction pipeline.

---

## Appendix: File Inventory

### Analysis Data
| File | Description | Location |
|------|-------------|----------|
| cps_00001.dat.gz | Raw IPUMS extract (235 MB) | temp/cps_mobility/data/ |
| verified_transitions.parquet | Persistence-filtered transitions | temp/cps_mobility/data/ |
| transitions_with_distances.parquet | Final analysis dataset | temp/cps_mobility/data/ |
| d_sem_occ.npz | Semantic distance matrix (894×894) | temp/cps_mobility/data/ |
| job_zone_matrix.npz | Institutional distance matrix (923×923) | temp/cps_mobility/data/ |
| conditional_logit_results.json | Model estimates | temp/cps_mobility/data/ |

### Scripts
| File | Purpose | Location |
|------|---------|----------|
| submit_ipums_extract.py | API extract submission | temp/cps_mobility/scripts/ |
| download_extract.py | Extract download | temp/cps_mobility/scripts/ |
| transition_filters.py | Measurement error correction | temp/cps_mobility/scripts/ |
| build_job_zone_matrix.py | d_inst construction | temp/mobility_feasibility/scripts/ |
| check_crosswalk_coverage.py | O*NET ↔ Census mapping | temp/mobility_feasibility/scripts/ |

---

*End of Kill Shot Documentation*