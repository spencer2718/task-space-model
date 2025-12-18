**TASK-SPACE ORACLE v0.7.2.x SPECIFICATION**
**Retrospective Battery Implementation**

**Status:** FINAL
**Date:** 2025-12-17
**Authors:** Lead Researcher, Spec Finalizer

---

## 1. Objective

Implement the retrospective diagnostic battery (Appendix A) to test whether continuous task-space geometry adds explanatory power beyond discrete classification for historical technology episodes. This is the falsifiability test for the oracle's representation quality.

**Core question:** When technology shocks hit labor markets, does measuring "distance from shock in continuous embedding space" predict outcomes better than "binary classification of affected vs. unaffected"?

---

## 2. Interpretation Framework

### 2.1 Pre-Registered Thresholds

| Verdict | Criteria |
|---------|----------|
| **+** | p < 0.05 AND incremental R² ≥ 0.01 |
| **−** | p < 0.05 AND β₃ wrong sign |
| **0** | Otherwise |

**Rationale:** 1% incremental R² is economically meaningful given the paper finds geometry explains ~13-15% of transition patterns. A statistically significant but variance-trivial effect is a "0."

### 2.2 Null Result Interpretation

If Test B yields "0" verdicts, this does NOT falsify the framework. Two interpretations:

1. **Construct distinction (expected):** Embedding space captures mobility friction (where workers CAN go), not automation susceptibility (which tasks get displaced). The paper already finds Wasserstein orthogonal to RTI (r = −0.05).

2. **Representation problem:** Embedding space doesn't add information in these settings.

Interpretation (1) is consistent with existing findings. Null Test B results would constrain scope, not invalidate the architecture.

---

## 3. Phase Structure

| Version | Deliverable | Gate | Output |
|---------|-------------|------|--------|
| **0.7.2.1** | Crosswalk: occ1990dd → O*NET-SOC | Coverage ≥ 80% emp-weighted | `occ1990dd_to_onet_soc.csv` |
| **0.7.2.2** | CSH implementation | 0.7 ≤ r(CSH, RTI) ≤ 0.9 | `CSHExposure` class |
| **0.7.2.3** | Test B: Polarization | Signed verdict | `battery_test_b_v0723.json` |
| **0.7.2.4** | Test A-lite: Task drift | Signed verdict | `battery_test_a_lite_v0724.json` |
| **0.7.2.5** | Test C': Webb robot exposure | Implementation | `RobotExposure` class |
| **0.7.3.0** | Battery consolidation | Interpretation matrix | Paper Section 6 update |

---

## 4. Detailed Specifications

### 4.1 Crosswalk Construction (0.7.2.1)

**Objective:** Build occ1990dd → O*NET-SOC mapping via Census 2000 chain.

**Method:**
```
occ1990dd ← [occ2000_occ1990dd.dta] ← Census 2000 OCC → SOC 2000 → O*NET-SOC
```

**Data sources (all present in repo):**
- `data/external/dorn_replication/occ2000_occ1990dd.dta`
- Census 2000 OCC → SOC crosswalk (acquire from census.gov)
- O*NET SOC crosswalk files

**Output schema:**
```
occ1990dd | onet_soc | weight | confidence_tier | emp_1980
```

**Acceptance criteria:**
- Coverage ≥ 80% of 1980 employment-weighted occupations
- Many-to-many mappings use employment-weighted probabilistic assignment
- Loss analysis documented by occupation type

**A-full gate:** Proceed to full Δρ tracking only if:
- Coverage ≥ 90%, OR
- Missing occupations represent < 5% of 1980 total employment

**Stop condition:** If coverage < 70%, return to Lead Researcher before proceeding.

---

### 4.2 CSH Implementation (0.7.2.2)

**Objective:** Operationalize Continuous Semantic Height as learned direction in embedding space.

**Method:**
1. Load ALM task scores from `occ1990dd_task_alm.dta`
2. Compute RTI = ln(Routine) − ln(Manual) − ln(Abstract) for each occ1990dd
3. Map occ1990dd → O*NET-SOC → DWA embedding centroids (via 0.7.2.1 crosswalk)
4. Learn direction vector **v**: maximize correlation between occupation DWA centroid projections and RTI
5. CSH(occ) = projection onto **v**

**Residualization for tests:**
```python
CSH_resid = CSH - (α + β·RSH)  # Regress CSH on discrete RSH
```

This isolates the geometric information that discrete classification misses.

**Robustness variant (Option A):**
- Routine centroid = employment-weighted average embedding of top-third RTI occupations
- CSH_alt = Wasserstein distance to routine centroid

**Acceptance criteria:**
- 0.7 ≤ r(CSH, RTI) ≤ 0.9

**Concern flag:** If r > 0.9, CSH is nearly redundant with RTI. The residualization will remove most signal, making β₃ uninformative. Document and assess variance of CSH_resid before proceeding to Test B.

**Stop condition:** If r(CSH, RTI) < 0.5, the learned direction doesn't capture routine content. Return to Lead Researcher.

**Output:** `CSHExposure` class implementing `ExposureMeasure` interface with:
- `discrete_exposure()` → RSH (binary routine indicator)
- `continuous_exposure()` → CSH_resid
- CZ-level aggregation via employment weights

---

### 4.3 Test B: Autor-Dorn Polarization (0.7.2.3)

**Objective:** Test whether continuous geometry adds explanatory power to polarization patterns.

**Specification:**
```
ΔY_cz = β₁·RSH + β₃·CSH_resid + X'γ + ε
```

**Outcomes (from Dorn workfile):**
1. Δ employment share in service occupations (1980-2005)
2. Δ employment share in routine occupations (1980-2005)
3. Δ log hourly wage, 10th percentile
4. Δ log hourly wage, 50th percentile
5. Δ log hourly wage, 90th percentile

**Controls (standard Autor-Dorn):**
- Census division dummies
- 1980 demographic controls (education shares, age shares, race shares)
- 1980 manufacturing share

**CZ aggregation:**
- CSH_cz = Σ_k (emp_share_k × CSH_k) for occupations k in CZ
- Parallels RSH construction

**Interpretation per outcome:**
- β₃ > 0, p < 0.05, ΔR² ≥ 0.01 → "+"
- β₃ < 0, p < 0.05 → "−"
- Otherwise → "0"

**Output:** JSON with verdicts, coefficients, standard errors, R² values for each outcome.

**Scope note:** Test B validates the embedding space direction (CSH as scalar), not the full Wasserstein optimal transport structure. Complementary to CPS mobility validation which tests the distance metric.

---

### 4.4 Test A-lite: Task Index Drift (0.7.2.4)

**Objective:** Test whether continuous geometry predicts task composition changes.

**Specification:**
```
Δ TaskIndex_it = β₁·ComputerIntensity_it + β₃·(CSH_i - DiscreteRoutine_i) + ε
```

**Unit of analysis:** Industry × decade (1980-1990, 1990-2000)

**Task indices:** Routine, Abstract, Manual (from ALM measures in Dorn files)

**Computer intensity:** PC utilization from Dorn workfile

**Output:** JSON with verdicts per task index per decade.

---

### 4.5 Test C': Webb Robot Exposure (0.7.2.5)

**Decision:** Pivot to Webb (2020) methodology instead of IFR purchase.

**Rationale:**
- Tests oracle's task-based approach (patent→task semantic matching)
- Higher learning value for "research-for-learning" orientation
- Avoids $1,000 IFR cost and data complexity
- If Test B strongly positive, IFR purchase can be revisited

**Method:**
1. Define robot-exposed DWA cluster based on Webb's patent-task findings:
   - Activities involving: moving, handling, welding, assembling objects, material manipulation
   - High loading on: repetitive motions, equipment-paced work, structured physical environments
2. Robot exposure = semantic similarity (embedding distance) to robot-task centroid
3. Aggregate to CZ via occupation employment shares

**Trade-off documented:** Test C' results not directly comparable to Acemoglu-Restrepo. Interpretation matrix distinguishes:
- Test C (IFR): "Does continuous outperform discrete in canonical robot setting?"
- Test C' (Webb): "Does task-based robot exposure outperform industry-based?"

C' is more aligned with oracle's value proposition—testing whether task-level measurement adds value.

**Output:** `RobotExposure` class; evaluation deferred pending B results.

---

### 4.6 Battery Consolidation (0.7.3.0)

**Deliverables:**
1. Complete interpretation matrix across all tests/outcomes
2. Summary statistics and diagnostics
3. Paper Section 6 update (promote Appendix A from "retained for future" to "implemented")

**Interpretation matrix format:**
```
| Test | Outcome | β₃ | SE | p | ΔR² | Verdict |
|------|---------|----|----|---|-----|---------|
| B | Δ Service Share | ... | ... | ... | ... | +/−/0 |
```

**Summary:**
```
| Verdict | Count | Interpretation |
|---------|-------|----------------|
| + | n | Continuous geometry adds information |
| − | n | Discrete classification dominates |
| 0 | n | No difference / inconclusive |
```

---

## 5. Data Requirements

**Already present:**
- `data/external/dorn_replication/occ1990dd_task_alm.dta` — ALM task scores
- `data/external/dorn_replication/occ2000_occ1990dd.dta` — Crosswalk
- `data/external/dorn_replication/dorn_extracted/` — Full archive
- O*NET db_30_0 — DWA embeddings
- Wasserstein distance matrices — Cached

**To acquire:**
- Census 2000 OCC → SOC 2000 crosswalk (census.gov, free)
- Dorn workfile2012.dta if not in archive (ddorn.net, free)
- IPUMS Census extracts for employment weights if needed (ipums.org, free with registration)

---

## 6. Timeline

| Task | Effort | Cumulative |
|------|--------|------------|
| 0.7.2.1 Crosswalk | 3-4 days | Day 4 |
| 0.7.2.2 CSH | 2-3 days | Day 7 |
| 0.7.2.3 Test B | 2-3 days | Day 10 |
| 0.7.2.4 Test A-lite | 2 days | Day 12 |
| 0.7.2.5 Test C' setup | 2 days | Day 14 |
| 0.7.3.0 Consolidation | 2-3 days | Day 17 |

**Total:** ~2.5 weeks to battery completion

---

## 7. Success Criteria for Phase

**Minimum viable:** Interpretation matrix with signed verdicts for Tests B and A-lite.

**Full success:** 
- All tests executed with pre-registered methodology
- Results documented in experiment JSONs
- Paper Section 6 updated
- Scope implications for 0.8.x demand integration identified

**Learning outcomes regardless of verdicts:**
- If "+" dominant: Proceed to 0.8.x with confidence in representation
- If "0" dominant: Confirms construct distinction (geometry ≠ automation); scope constraint documented
- If "−" dominant: Fundamental reassessment required; geometry may not capture economically relevant structure

---

## 8. Forward Link to Phase 0.8

| Phase | Question | Tests |
|-------|----------|-------|
| 0.7.2-3 | Does continuous geometry predict canonical outcomes? | Representation quality |
| 0.8.x | Does geometry combine with demand for equilibrium prediction? | Architecture integration |

**Design requirement for 0.8.x:** Include separability test—does T module contribution remain stable when M incorporates demand?

---

**SPECIFICATION FINAL**

Ready for Engineer instructions on 0.7.2.1 implementation.