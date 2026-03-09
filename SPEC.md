## SPEC v0.7.7.x: Paper Alignment Sprint

---

### Strategic Context

**Research Question:** Do the published numbers, citations, and originality claims survive scrutiny?

**Why This Matters:** External review of the publishable draft (embeddings_v8.pdf) identified a diagonal bias in the primary distance matrix, missing prior art citations, and undisclosed estimation details. These must be corrected before any further research or submission.

**Contribution:** Corrected, defensible paper with honest numbers and precise originality framing.

---

### Versioning

| Phase | Version | Scope | Status |
|-------|---------|-------|--------|
| Diagonal audit | v0.7.7.0 | Quantify bias, corrected numbers | ✅ Complete |
| Paper numbers + citations | v0.7.7.1 | Working paper updates | ✅ Complete (with issues) |
| Data integrity fixes | v0.7.7.2 | Fix fabricated table, refs, ratios | ✅ Complete |
| Centroid COVID re-estimation | v0.7.7.3 | Re-run pre/post COVID with centroid spec | ✅ Complete |
| Disclosures | v0.7.7.4 | In-sample and crosswalk aggregation caveats | ✅ Complete |
| Consistency pass | v0.7.7.5 | 10 adversarial audit fixes | ✅ Complete |
| Software hygiene | v0.7.7.6 | Cache fixes, stale artifacts, tests, reproducibility | ✅ Complete |
| Origin-exclusion robustness | v0.7.7.7 | Test + disclosure | Pending |
| Overleaf sync | v0.7.8.0 | Carry corrections to publishable draft | Pending |

---

### Completed Findings

- Corrected embedding Wasserstein pseudo-R²: **13.76%** (was 14.51%)
- Centroid pseudo-R²: **14.08%** (unchanged, now best-performing)
- Ground metric improvement: **+83%** over identity (was +96%)
- Correction impact: **−0.79pp** from diagonal zeroing

### Out of Scope

- New experiments or analyses
- Figure updates (deferred from v0.7.6)
- Domain-specific embedding comparisons
- Structural estimation or GE extensions
