# Presentation Figures

| File | Slide | Size | Status |
|------|-------|------|--------|
| fig1_ai_exposure.png | 2 — Motivation | 9.0″ × 6.5″ | ✅ Done |
| fig8_embedding_flowchart.png | 3 — What is a Sentence Embedding? (left) | 3.8″ × 4.0″ | ✅ Done |
| fig9_word_analogy.png | 3 — What is a Sentence Embedding? (right) | 4.5″ × 4.0″ | ✅ Done |
| fig3_task_scatter.png | 4 — Tasks in Semantic Space | 6.0″ × 4.0″ | ✅ Done |
| fig5_shared_dwas.png | 5 — Example: Why Embeddings Matter (left) | 4.1″ × 4.2″ | ✅ Done |
| fig6_embedding_similarity.png | 5 — Example: Why Embeddings Matter (right) | 4.7″ × 4.0″ | ✅ Done |
| fig2_pseudo_r2.png | 7 — Main Result | 9.2″ × 4.2″ | ✅ Done |
| fig4_scope.png | 8 — What the Measure Captures | 4.2″ × 3.4″ | ✅ Done |
| fig7_sankey_pipeline.png | B1 — Data Pipeline (backup) | 9.0″ × 4.2″ | ✅ Done |

Note: fig numbering is by creation order, not slide order.

## fig1 — AI Task Exposure
- **Slide:** 2 — Motivation
- **Data:** Eloundou et al. (2023) γ = E1 + E2, averaged to 2-digit SOC
- **Source:** `data/external/eloundou/occ_level.csv`
- **Takeaway:** 12–97% of tasks theoretically feasible across occupation groups

## fig8 — Embedding Flowchart
- **Slide:** 3 — What is a Sentence Embedding? (left)
- **Data:** Live from cached MPNet embeddings of O*NET DWAs
- **Example sentence:** "Analyze business or financial data." (DWA from O*NET)
- **Takeaway:** A sentence goes in, a 768-dimensional vector comes out

## fig9 — Word Analogy (Atlanta-Denver)
- **Slide:** 3 — What is a Sentence Embedding? (right)
- **Data:** Illustrative positions (not computed) — Mikolov et al. (2013) analogy
- **Takeaway:** Directions in embedding space encode semantic relationships

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
- **Slide:** 7 — Main Result
- **Data:** Hardcoded from `outputs/experiments/distance_head_to_head_v0732.json`
- **Takeaway:** Embedding-based measures achieve 13.8–14.1% pseudo-R² vs 6–8% for O*NET — 74.9% improvement

## fig4 — Supply-Demand Decomposition
- **Slide:** 8 — What the Measure Captures
- **Data:** Hardcoded from `outputs/experiments/demand_probe_decomposition_v0703b.json`
- **Takeaway:** Aggregate inflows are demand-dominated (ρ = 0.80); geometry captures per-origin pathway direction (ρ = 0.13)

## fig7 — CPS Data Pipeline (Sankey)
- **Slide:** B1 — Data Pipeline (backup)
- **Data:** Hardcoded from paper Table 1
- **Takeaway:** 10M raw records → 89K verified transitions through 5 filtering stages

## Shared style
- **Module:** `style.py` — colors, fonts, rcParams
- **Font:** DejaVu Sans (Calibri not available on build system)
- **Colors:** Steel blue (#4A7FB5) primary, warm tan (#A0937D) secondary, muted red (#C75C5C), amber (#D4845A), teal (#44AA99, Tol colorblind-safe), purple (#8B6BAE)
