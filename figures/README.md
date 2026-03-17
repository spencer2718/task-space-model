# Presentation Figures

| File | Slide | Status |
|------|-------|--------|
| fig1_ai_exposure.png | 2 — Motivation (theoretical vs observed AI exposure) | ✅ Done |
| fig8_embedding_flowchart.png | 3 — What is a Sentence Embedding? (left) | ✅ Done |
| fig9_word_analogy.png | (archived) | Replaced by fig10 |
| fig10_similarity_heatmap.png | 3 — What is a Sentence Embedding? (right) | ✅ Done |
| fig3_task_scatter.png | 4 — Tasks in Semantic Space | ✅ Done |
| fig5_shared_dwas.png | 5 — Example: Why Embeddings Matter (left) | ✅ Done |
| fig6_embedding_similarity.png | 5 — Example: Why Embeddings Matter (right) | ✅ Done |
| fig_logit_eq.png | 6 — What Drives the Improvement? | ✅ Done |
| fig2_pseudo_r2.png | 7 — Main Result | ✅ Done |
| fig4_scope.png | 8 — What the Measure Captures | ✅ Done |
| fig7_sankey_pipeline.png | B1 — Data Pipeline (backup) | ✅ Done |

Note: fig numbering is by creation order, not slide order.

## fig1 — AI Exposure: Theoretical vs Observed
- **Slide:** 2 — Motivation
- **Data:** Massenkoff & McCrory (2026), Fig. 2; values estimated from published radial chart (occupation-level CSV not yet released)
- **Categories:** 8 SOC major groups, sorted by theoretical exposure
- **Takeaway:** AI capability far exceeds observed deployment — Computer & Math: 94% theoretical, 33% observed

## fig8 — Embedding Flowchart
- **Slide:** 3 — What is a Sentence Embedding? (left)
- **Data:** Live from cached MPNet embeddings of O*NET DWAs
- **Example sentence:** "Analyze business or financial data." (DWA from O*NET)
- **Takeaway:** A sentence goes in, a 768-dimensional vector comes out

## fig9 — Word Analogy (ARCHIVED)
- **Status:** Replaced by fig10 at v0.7.13.4. File retained in repo.

## fig10 — Cosine Similarity Heatmap
- **Slide:** 3 — What is a Sentence Embedding? (right)
- **Data:** 6 hand-picked O*NET DWAs (2 each: Quantitative, Healthcare, Construction), embedded with MPNet
- **Colors:** Domain colors match fig3 scatter (Quantitative=PRIMARY, Healthcare=RED, Construction=SECONDARY)
- **Takeaway:** Similar tasks embed to similar vectors — block diagonal structure shows semantic clustering

## fig3 — Tasks in Semantic Space
- **Slide:** 4 — Tasks in Semantic Space
- **Data:** MPNet embeddings of 30 hand-curated DWAs, semantic anchor projection
- **Source:** `fig3_selected_dwas.csv` (30 DWAs, 5 themes × 6 dots, 10 labels)
- **Themes:** Healthcare (RED), Vehicle & Equipment (ORANGE), Construction (tan), Quantitative (steel blue), Communication (teal)
- **Axes:** x = Non-Routine − Routine, y = Cognitive − Manual (cosine similarity to anchor phrases, z-scored × 2.5)
- **Takeaway:** Related tasks cluster in interpretable quadrants

## fig5 — Shared DWAs (Budget Analysts vs Credit Analysts)
- **Slide:** 5 — Example: Why Embeddings Matter (left)
- **Data:** Hardcoded from O*NET Tasks-to-DWAs mapping
- **Pair:** Budget Analysts (13-2031.00) vs Credit Analysts (13-2041.00)
- **Takeaway:** Only 3 of 19 DWAs overlap — O*NET cosine distance = 0.71

## fig6 — Embedding Similarity
- **Slide:** 5 — Example: Why Embeddings Matter (right)
- **Data:** Census-level distance matrices from `.cache/artifacts/v1/mobility/`
- **Pair:** Same as fig5 — Budget Analysts (Census 820) vs Credit Analysts (Census 830)
- **Takeaway:** Embedding distance = 0.07 (1.3rd percentile) vs O*NET = 0.71 — embeddings detect similarity that O*NET misses

## fig2 — Distance Metric Comparison
- **Slide:** 7 — Main Result (left panel, 60/40 split with parameter table)
- **Data:** Hardcoded from `outputs/experiments/distance_head_to_head_v0732.json`
- **Labels:** Two-line format: "Embedding\n(Centroid)" etc.
- **Takeaway:** Embedding-based measures achieve 13.8–14.1% pseudo-R² vs 6–8% for O*NET — 74.9% improvement

## fig4 — Supply-Demand Decomposition
- **Slide:** 8 — What the Measure Captures
- **Data:** Hardcoded from `outputs/experiments/demand_probe_decomposition_v0703b.json`
- **Takeaway:** Aggregate inflows are demand-dominated (ρ = 0.80); geometry captures per-origin pathway direction (ρ = 0.12)

## fig_logit_eq — Conditional Logit Equation
- **Slide:** 6 — What Drives the Improvement?
- **Data:** None (equation render)
- **Method:** matplotlib mathtext with Computer Modern fontset
- **Takeaway:** P(j|i) ∝ exp(−α·d_sem − β·d_inst) — destination choice weighted by semantic and institutional distance

## fig7 — CPS Data Pipeline (Sankey)
- **Slide:** B1 — Data Pipeline (backup)
- **Data:** Hardcoded from paper Table 1
- **Takeaway:** 10M raw records → 89K verified transitions through 5 filtering stages

## Shared style
- **Module:** `style.py` — colors, fonts, rcParams
- **Font:** DejaVu Sans (Calibri not available on build system)
- **Colors:** Steel blue (#4A7FB5) primary, warm tan (#A0937D) secondary, muted red (#C75C5C), amber (#D4845A), teal (#44AA99, Tol colorblind-safe), purple (#8B6BAE)
