# Distillation Plan

**Branch:** `distill/undergrad-paper`
**Target repo:** `spencer2718/task-embeddings` (clean, no git history)
**Target audience:** Academic economist with standard grad coursework but no labor econ specialization and no exposure to sentence embeddings.
**Style reference:** Acemoglu & Restrepo — clean prose, intuition before formalism, policy relevance framing.

---

## Paper Scope

### Keep (core results)

- 2×2 factorial comparison (Table 2): embedding × aggregation, four specifications
- Conditional logit with semantic + institutional decomposition (Table 5)
- Supply-demand decomposition: demand ρ = 0.80, per-origin ρ ≈ 0.12
- COVID structural stability (Δα < 1%, p = 0.72; telework heterogeneity)
- AIOE orthogonality (r = −0.02) and RTI orthogonality (r = −0.06)
- Distance matrix correlations (Table 4, centroid-referenced)
- MPR = 0.74 (pending: all-specs comparison from `exploration/onet_894/mpr_all_specs.py`)

### Drop

- Gravity model (Table 9, §4.6) — aggregate flow prediction is secondary to individual choice model
- Asymmetry appendix (Appendix C) — inconclusive results, adds complexity without payoff
- Switching cost calibration (Appendix B) — external calibration acknowledged as limitation, not a result
- Ground metric diagnostic table (Table 3) — finding absorbed into prose ("identity Wasserstein performs like O*NET baselines")
- Wasserstein formal definition (Appendix A, Definition 2) — centroid is primary, Wasserstein is a comparison spec not the methodology

### Add

- **Embedding preliminaries (2–3 pages):** What is a vector representation, what is cosine similarity, how do sentence transformers work, why this matters for measuring occupational distance. Written for an economist who has never seen an embedding. Implicit, not condescending.
- **Presentation-style figures as exposition:** Similarity heatmap (fig10), theoretical-vs-observed AI exposure (fig1), task scatter with background cloud (fig3). Case-by-case decisions on inclusion.
- **Updated motivation:** Massenkoff & McCrory (2026) observed exposure data in introduction. Reframe from "AI is reshaping work" to "workforce policy needs a map of task space."
- **Intuitive examples throughout:** Budget Analysts vs Credit Analysts, Pipelayers vs Septic Cleaners, postsecondary teacher cluster — use concrete pairs to build intuition before formalism.

---

## Repo Structure (target: task-embeddings)

```
task-embeddings/
    README.md
    writing/
        main.tex
        references.bib
    pipeline/
        python/
            01_onet_ingestion.py
            02_embed_dwas.py          # Only step that requires Python
            03_distance_matrices.py
            04_cps_transitions.py
            05_conditional_logit.py
            06_validation_metrics.py
        r/
            01_onet_ingestion.R
            # 02 skipped — takes embeddings as input
            03_distance_matrices.R
            04_cps_transitions.R
            05_conditional_logit.R     # mlogit or survival::clogit
            06_validation_metrics.R
        shared/
            embeddings/               # Pre-computed MPNet embeddings (cached)
            distances/                # Distance matrices (npz + csv exports)
    figures/
        style.py
        fig*.py
    data/
        README.md
```

### R Pipeline Scope

- **Inputs from Python:** Pre-computed DWA embeddings (768d vectors as CSV), O*NET raw data
- **Replicates in R:** O*NET ingestion, occupation measure construction, centroid distance computation, CPS transition filtering, conditional logit estimation, MPR, pseudo-R², all validation metrics
- **Verification criterion:** Coefficients, pseudo-R², and MPR match Python to 4 decimal places (floating-point tolerance)
- **Key R packages:** `tidyverse` (data), `mlogit` or `survival::clogit` (estimation), `lsa` or manual cosine (distances)
- **Note:** `mlogit` and `survival::clogit` handle sampled alternatives differently from `statsmodels`. Verification pass required.

---

## Sequencing

| Phase | Scope | Status |
|-------|-------|--------|
| 0. Pre-branch | Run MPR all-specs on main, log in LEDGER | TODO |
| 1. Paper distillation | Trim publishable → undergrad paper, add preliminaries | NOT STARTED |
| 2. Figure integration | Add presentation figures to paper where useful | NOT STARTED |
| 3. Python pipeline cleanup | Compress scripts into numbered pipeline, remove redundancy | NOT STARTED |
| 4. R pipeline | Translate pipeline (excluding embedding), verify numerical match | NOT STARTED |
| 5. Repo migration | Copy to spencer2718/task-embeddings, clean README, no history | NOT STARTED |

---

## Style Notes

- **Prose:** Acemoglu & Restrepo register. Clear, direct, policy-aware. No jargon without definition. Intuition paragraph before every formal result.
- **Math:** Introduce notation only when needed. The conditional logit equation is the only essential formula. Distance definitions can be prose with inline math.
- **Figures:** Every figure should be interpretable without reading the caption twice. Labels, colors, and annotations should carry the message.
- **Length:** No hard constraint. Aim for 15–20 pages including references. Quality over completeness.
- **Citations:** Keep only those directly relevant. The 33-entry publishable bib is already lean; may trim further.
