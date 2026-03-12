## SPEC v0.7.12.x: Centroid Replication Sprint

---

### Strategic Context

**Objective:** Rerun all supporting analyses on the centroid (primary) specification so that every reported number in both papers is internally consistent with HC1. Currently, the main choice-model result uses centroid, but AIOE correlation, RTI correlation, demand decomposition, gravity model, out-of-period comparison, and switching cost calibration were computed on Wasserstein. The publishable paper explicitly discloses this gap (§4.1: "we retain Wasserstein in those tables for continuity"). This sprint closes the gap.

**Precondition:** The centroid distance matrix already exists at `.cache/artifacts/v1/mobility/d_cosine_embed_census.npz` (447×447). No new distance computation is needed.

---

### Versioning

| Phase | Version | Scope | Status |
|-------|---------|-------|--------|
| Distance matrix audit | v0.7.12.0 | Recompute ρ(centroid, Wasserstein) post-diagonal-correction; verify centroid matrix is clean | |
| Supporting experiments | v0.7.12.1–5 | Rerun 6 analyses on centroid | |
| Paper updates | v0.7.12.6 | Swap numbers in publishable + working paper | |
| Figures / deck / docs | v0.7.12.7 | Update any affected figures, build_deck.js, README, CLAUDE, LEDGER | |

---

### Out of Scope

- New experiments or analyses not already in the paper
- Rerunning the main conditional logit (already centroid)
- Rerunning the 2×2 factorial table (already reports both)
- Rerunning the COVID stability test (already centroid per v0.7.7.3)
- Changing the ground metric comparison table (Table 3 — inherently Wasserstein-only by design)
- Any change to the gravity model's Wasserstein result (Wasserstein performs best for aggregate flows; that finding stays, but centroid should also be reported)

---

### Deliverable Specifications

#### D0: Distance Matrix Audit (v0.7.12.0)

- **Objective:** Confirm the centroid matrix is diagonal-clean and recompute inter-matrix correlation post-diagonal-correction.
- **Method:**
  1. Load `d_cosine_embed_census.npz` and `d_wasserstein_census.npz`
  2. Verify centroid diagonal is all zeros (it should be — centroid self-distance is zero by construction, unlike Wasserstein which had the SOC aggregation bug)
  3. Compute Spearman ρ between flattened upper triangles of both matrices (excluding diagonal)
  4. Report updated ρ value
- **Acceptance criteria:** ρ is computed and documented. If ρ < 0.90 (unexpected), STOP and return — the "materially similar" claim may not hold.
- **Contingency:** If centroid matrix doesn't exist or has wrong shape, STOP.

#### D1: AIOE Correlation (v0.7.12.1)

- **Objective:** Recompute AIOE–distance correlation using centroid mean distances instead of Wasserstein.
- **Method:** Adapt `shock_integration.py::compute_aioe_geometry_correlations()` to accept centroid matrix. For each occupation, compute mean centroid distance to all others, correlate with AIOE score (Pearson).
- **Data:** `d_cosine_embed_census.npz`, AIOE scores (existing pipeline)
- **Acceptance criteria:** New r value reported. Expected: close to 0.02.
- **Contingency:** If |r| > 0.20, STOP — would change the orthogonality claim.

#### D2: RTI Correlation (v0.7.12.2)

- **Objective:** Recompute RTI–distance correlation using centroid.
- **Method:** Adapt `path_c_rti_construct_validity.py` to use centroid distances at O*NET level (or Census level with appropriate crosswalk). Compute mean centroid distance per occupation, correlate with Autor-Dorn RTI composite + components (Pearson).
- **Data:** Centroid distances, Dorn RTI file, existing crosswalks
- **Acceptance criteria:** New r values for RTI composite, routine, abstract, manual. Expected: all near zero.
- **Contingency:** If |r(RTI composite)| > 0.20, STOP.

#### D3: Demand Decomposition (v0.7.12.3)

- **Objective:** Recompute three Spearman ρ values (demand, per-origin geometry, aggregate geometry) using centroid.
- **Method:** Adapt `run_demand_decomposition_v0703b.py` to load centroid matrix instead of `load_wasserstein_census()`. All logic stays the same — only the distance matrix input changes.
- **Data:** `d_cosine_embed_census.npz`, BLS projections, holdout transitions
- **Acceptance criteria:** Three ρ values reported. Demand ρ unchanged (doesn't use distance). Per-origin and aggregate geometry ρ expected close to existing values.
- **Contingency:** If per-origin ρ changes sign or changes by more than 0.05, STOP.

#### D4: Out-of-Period Comparison (v0.7.12.4)

- **Objective:** Recompute ΔLL = geometry vs historical baseline using centroid-based model probabilities.
- **Method:** Adapt `shock_integration.py::compute_model_probabilities()` to use centroid matrix. Recompute log-likelihoods on holdout (2024) data.
- **Data:** Centroid matrix, institutional matrix, holdout transitions, fitted α/β from centroid conditional logit
- **Acceptance criteria:** New ΔLL reported. Expected: close to +23,119.
- **Contingency:** If ΔLL < +10,000, STOP — material degradation.

#### D5: Gravity Model (v0.7.12.5)

- **Objective:** Report centroid partial R² alongside existing Wasserstein result.
- **Method:** Rerun gravity specification with centroid distance as the distance variable. The gravity table already has multiple metrics (Wasserstein, cosine O*NET, etc.); add centroid row.
- **Data:** Centroid matrix, employment mass terms, bilateral flows
- **Acceptance criteria:** Centroid partial R² reported. Expected: close to Wasserstein's 3.46%.
- **Contingency:** If centroid partial R² < 1%, flag — may indicate centroid is weaker for aggregate flows (which would actually be informative, not a problem).

#### D6: Switching Cost Recalibration (v0.7.12.5, same pass)

- **Objective:** Recompute the switching cost anchor on the centroid distance scale.
- **Method:** Compute median centroid distance. Apply same calibration: 2.0 wage-years / median distance = SC per unit centroid distance. Report new value alongside 3.84 wage-years/unit Wasserstein.
- **Acceptance criteria:** New calibration number reported.
- **Contingency:** None — this is arithmetic.

#### D7: Paper Updates (v0.7.12.6)

- **Objective:** Swap all affected numbers in both papers.
- **Method:** Lead Researcher provides find/replace pairs for Writer (publishable) and Engineer (working paper alignment). Remove the "we retain Wasserstein for continuity" disclosure from publishable §4.1. Add centroid row to gravity table. Update switching cost section.
- **Acceptance criteria:** All grep checks pass (no remaining Wasserstein-based values in sections that should now be centroid). Publishable bib/citation integrity maintained.
- **Contingency:** If any number change is material (>10% relative), Lead Researcher reviews framing before committing.

#### D8: Figures / Deck / Docs (v0.7.12.7)

- **Objective:** Update any affected figures, the presentation deck, README, CLAUDE.md, and LEDGER.
- **Method:**
  - fig4 (scope bars): Rerun with centroid-based ρ values (per-origin, aggregate). Demand bar unchanged.
  - build_deck.js slide 8: Update bullet point values if changed.
  - README Module Status table: Update if any numbers change.
  - CLAUDE.md: Update switching cost value if changed.
  - LEDGER: Update Claim Registry values, Module Validation Checkpoints, Artifact Registry.
- **Acceptance criteria:** All canonical values in LEDGER match papers match figures match deck.

---

### Stop-and-Return Conditions

- ρ(centroid, Wasserstein) < 0.90 after diagonal correction → STOP (undermines "materially similar" claim)
- Any orthogonality correlation |r| > 0.20 → STOP (changes a headline claim)
- Per-origin Spearman ρ changes sign → STOP
- ΔLL drops below +10,000 → STOP (material out-of-period degradation)
- Any paper number changes by >10% relative → STOP for Lead review before committing

---

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Centroid matrix has issues (shape, NaN) | Low | D0 audit catches this before experiments run |
| ρ(centroid, Wasserstein) materially lower post-diagonal | Low | Diagonal correction hit Wasserstein, not centroid; correlation likely stable or higher |
| RTI crosswalk complexity causes D2 failures | Medium | RTI script uses O*NET-level Wasserstein; need to confirm O*NET-level centroid matrix exists or compute Census-level mean distances instead |
| Gravity model shows centroid much weaker than Wasserstein | Medium | This is actually informative (Wasserstein better for aggregate flows); report both, note in paper |
| Scope creep into re-estimating conditional logit | Low | Already centroid; out of scope per above |