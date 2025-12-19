**TASK-SPACE ORACLE v0.7.3.x SPECIFICATION**
**Battery Completion + Mobility Validation**

**Status:** FINAL
**Date:** 2025-12-19
**Version Range:** v0.7.3.1 → v0.7.4.0

---

## 1. Strategic Context

**Key finding from v0.7.2.x:** Embedding geometry captures **mobility friction** (where workers CAN go), not **automation susceptibility** (which jobs get displaced). Test B confirmed this construct distinction: 1+, 0−, 4(0) for polarization outcomes.

**Sprint objective:** Validate this construct distinction through additional tests, then pivot toward mobility applications where the framework shows genuine predictive power.

**Success criteria for v0.7.4.0:**
- Test C' completed with proper methodology
- At least one Tier 1 mobility validation test completed
- Phase 0.8 demand data access assessed
- Framework scope definitively established

---

## 2. Phase Structure

| Version | Deliverable | Gate | Timeline |
|---------|-------------|------|----------|
| **0.7.3.1** | Test C' proper implementation | Robot direction learned, r > 0.5 with Webb/Felten | Week 1-2 |
| **0.7.3.2** | Test C' execution | Signed verdict | Week 2-3 |
| **0.7.3.3** | Decision point: Construct distinction | Confirmed or disconfirmed | Week 3 |
| **0.7.3.4** | Test A-lite data acquisition (conditional) | GO only if C' confirms distinction | Week 3-4 |
| **0.7.3.5** | Tier 1 mobility validation | At least one test executed | Week 5-6 |
| **0.7.4.0** | Sprint consolidation + Phase 0.8 spec | All results documented | Week 7-8 |

---

## 3. Test C' Proper Implementation (v0.7.3.1-0.7.3.2)

### 3.1 Objective

Test whether embedding geometry predicts robot-era employment decline. This completes the battery with correct methodology (embedding distance, not keyword classification).

**Expected outcome:** Geometry does NOT add significant predictive power beyond existing robot exposure measures — confirming construct distinction.

### 3.2 Data Acquisition

**Primary:** Webb (2020) robot exposure scores
- Source: http://eepurl.com/gxo4zr (email signup required)
- Contains occupation-level robot patent exposure
- Most methodologically rigorous option

**Fallback:** Felten et al. AIOE scores
- Source: https://github.com/AIOE-Data/AIOE
- Immediately available
- Validated correlation with Webb

**Decision:** If Webb data not received within 3 business days, proceed with Felten AIOE.

### 3.3 Robot Direction Learning (Option C)

**Method:** Learn direction in embedding space that maximizes correlation with robot exposure, analogous to CSH learning from RTI.

```python
# Pseudocode
1. Load Webb/Felten robot exposure scores at occupation level
2. Map to occ1990dd via crosswalk (reuse v0.7.2.1 infrastructure)
3. Get occupation embedding centroids (reuse v0.7.2.2 infrastructure)
4. Learn robot direction via ridge regression:
   robot_direction = ridge_fit(centroids, robot_exposure)
5. Robot_CSH = centroids @ robot_direction
6. Validate: r(Robot_CSH, robot_exposure) should be > 0.5
```

**Acceptance criteria:**
- r(Robot_CSH, robot_exposure) ≥ 0.5
- Robot_CSH distribution has meaningful variance
- Top/bottom occupations pass smell test (forklift operators high, clergy low)

### 3.4 Test C' Execution

**Design:**
```
Δ ln(emp_share)_occ = β₁·RobotExposure + β₃·Robot_CSH_resid + controls + ε
```

**Sample:** Manufacturing occupations, 1990-2005 (robot adoption period)

**Outcome:** Employment share change

**Robot_CSH_resid:** Robot_CSH after regressing out Webb/Felten score

**Controls:** Baseline wage, baseline employment share, RTI (to isolate robot from general automation)

**Interpretation:**
- β₃ significant, ΔR² ≥ 1% → "+" (geometry adds to robot prediction — **disconfirms** construct distinction)
- β₃ not significant or ΔR² < 1% → "0" (geometry does NOT add — **confirms** construct distinction)

### 3.5 Output

`outputs/experiments/battery_test_c_prime_v0731.json`

---

## 4. Decision Point: Construct Distinction (v0.7.3.3)

**After Test C' results:**

| Test C' Result | Test B Pattern | Interpretation | Action |
|----------------|----------------|----------------|--------|
| "0" | 1+, 0−, 4(0) | **CONFIRMED**: Geometry = mobility, not automation | Proceed to mobility validation |
| "+" | 1+, 0−, 4(0) | **MIXED**: Geometry captures some automation signal | Investigate what robot component geometry captures |
| "+" | (hypothetical strong B) | **DISCONFIRMED**: Geometry = automation | Reconceptualize framework |

**Expected path:** Test C' = "0", confirming construct distinction. Proceed with mobility-focused validation.

**If disconfirmed:** STOP. Return to Lead Researcher for framework reconceptualization before proceeding.

---

## 5. Test A-lite (v0.7.3.4) — Conditional

**Gate:** Execute only if Test C' confirms construct distinction.

### 5.1 Data Acquisition

**Sources:**
- Dorn [B1]: occ1990dd_task_alm.zip (already have)
- Dorn [C4]: ind1990dd balanced industry panel — https://www.ddorn.net/data.htm
- BEA Fixed Assets: Computer capital by industry — https://www.bea.gov/data/investment-fixed-assets/industry

**Estimated effort:** 2-3 days for data acquisition and crosswalk construction

### 5.2 Design

**Unit:** Industry × decade (1980-1990, 1990-2000)

**Specification:**
```
Δ TaskIndex_ind = β₁·ComputerIntensity + β₃·Industry_CSH_resid + controls + ε
```

**Industry_CSH:** Employment-weighted average CSH across occupations within industry

**Outcomes:** Δ Routine, Δ Abstract, Δ Manual task indices

### 5.3 Decision

If data acquisition exceeds 3 days or requires extensive manual crosswalk construction, defer to v0.7.4+ and document as blocked.

---

## 6. Tier 1 Mobility Validation (v0.7.3.5)

**Rationale:** If geometry captures mobility friction, it should predict:
- Unemployment duration (more nearby occupations → faster reemployment)
- Post-displacement wage recovery (higher task similarity → less wage loss)

### 6.1 Option A: Unemployment Duration Test (Preferred)

**Hypothesis:** Workers in occupations with more "nearby" alternatives (lower average Wasserstein distance to other occupations) experience shorter unemployment spells.

**Data:** SIPP unemployment spells merged with O*NET task content

**Measure:** For each occupation, compute:
- `mean_distance` = average Wasserstein distance to all other occupations
- `nearby_mass` = fraction of employment in occupations within distance threshold

**Outcome:** Unemployment duration (weeks)

**Specification:**
```
ln(duration)_i = β₁·nearby_mass_occ + X'γ + ε
```

**Expected:** β₁ < 0 (more nearby options → shorter duration)

### 6.2 Option B: Post-Displacement Wage Recovery

**Hypothesis:** Workers who transition to task-similar occupations retain more of their pre-displacement wage.

**Data:** Displaced Worker Survey (CPS supplement) with occupation codes

**Measure:** Task similarity between pre- and post-displacement occupations (1 - Wasserstein distance)

**Outcome:** Wage ratio (post/pre displacement)

**Specification:**
```
ln(wage_ratio)_i = β₁·task_similarity + X'γ + ε
```

**Expected:** β₁ > 0 (higher similarity → better wage retention)

### 6.3 Selection

**Prefer Option A** (unemployment duration) if SIPP data accessible within 1 week.

**Fall back to Option B** if SIPP preparation exceeds estimates.

**Minimum viable:** At least one test completed with interpretable results.

---

## 7. Parallel Track: Phase 0.8 Preparation

### 7.1 Demand Data Access Assessment

**Objective:** Determine feasibility and cost of occupation-level vacancy data.

**Tasks:**
1. Check institutional WRDS access for Revelio Labs
2. Request Lightcast academic pricing (expect $5,000-12,000/year)
3. Download Indeed Hiring Lab indices as free baseline (github.com/hiring-lab)

**Note:** JOLTS provides industry-level data only — insufficient for occupation-level demand integration.

**Output:** Memo documenting access options, costs, and recommendation for Phase 0.8.

### 7.2 Δρ Operator Scoping

**Defer detailed Δρ specification** until demand data access confirmed. The Δρ concept requires observing task content evolution over time, which only job posting data provides.

**Preliminary design question:** Is Δρ (task distribution change) better modeled as:
- (a) Shift toward/away from shock-exposed tasks within occupation
- (b) Reweighting of DWA importance based on posting frequency
- (c) Emergence/deprecation of specific tasks

---

## 8. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Webb data access delayed | Medium | Low | Use Felten AIOE immediately |
| Test C' disconfirms construct distinction | Low | High | Framework reconceptualization required |
| SIPP data preparation exceeds estimate | Medium | Medium | Fall back to Displaced Worker Survey |
| Lightcast pricing prohibitive | Medium | High | Pursue WRDS-Revelio; use Indeed for validation |
| Test A-lite data too complex | Medium | Low | Defer to v0.7.4+; battery sufficient without it |

---

## 9. Stop-and-Return Conditions

**Return to Lead Researcher if:**

1. Test C' shows β₃ significant with ΔR² ≥ 1% (geometry predicts robots — construct distinction in question)
2. Robot direction learning yields r < 0.3 with Webb/Felten (methodology not working)
3. Both mobility validation tests (A and B) show null results (mobility interpretation unsupported)
4. Any MS10 situation: implementation requires deviation from spec

---

## 10. Sprint Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | Webb data request; Felten download; robot direction learning | Robot_CSH values |
| 2 | Test C' regression; results analysis | battery_test_c_prime_v0731.json |
| 3 | Decision point; A-lite data acquisition begins | Construct distinction verdict |
| 4 | A-lite pipeline (conditional); SIPP/DWS preparation | Data ready for mobility test |
| 5 | Mobility validation test execution | Regression results |
| 6 | Results analysis; demand data assessment | Access memo |
| 7 | Documentation; Phase 0.8 spec draft | Sprint summary |
| 8 | Consolidation; version bump | v0.7.4.0 release |

---

## 11. Success Metrics for v0.7.4.0

**Minimum viable:**
- Test C' completed with proper methodology
- Construct distinction verdict documented
- At least one mobility validation test with interpretable results
- Phase 0.8 scoped

**Full success:**
- All above plus Test A-lite completed (or documented as blocked with clear rationale)
- Demand data access secured or priced
- Framework scope definitively established with multiple supporting tests
- Clear policy application pathway identified

---

## 12. Claim Registry Updates (Anticipated)

If construct distinction confirmed:

| Claim ID | Canonical Text | Evidence Class | Status |
|----------|----------------|----------------|--------|
| CD-E1 | Embedding geometry captures mobility friction, not automation susceptibility | E1 | VALIDATED |
| CD-E2 | Test B (1+/5) and Test C' (0) confirm orthogonality to automation prediction | E1 | VALIDATED |
| M-E2 | Nearby occupation mass predicts unemployment duration (if mobility test succeeds) | E2 | VALIDATED |

---

**SPECIFICATION FINAL**

Ready for Engineer implementation upon Lead Researcher approval.