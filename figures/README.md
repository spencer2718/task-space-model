# Presentation Figures

| File | Slide | Size | Status |
|------|-------|------|--------|
| fig1_ai_exposure.png | 2 — Motivation | 9.0″ × 6.5″ | Done |
| fig2_pseudo_r2.png | 6 — Main Result | 9.0″ × 5.0″ | Done |
| fig3_task_scatter.png | 3 — Tasks in Semantic Space | 6.0″ × 4.2″ | Done |
| fig5_shared_dwas.png | 4 — Example (left panel) | 4.3″ × 4.2″ | NEW |
| fig6_embedding_similarity.png | 4 — Example (right panel) | 4.5″ × 4.2″ | NEW |
| fig7_sankey_pipeline.png | B1 — Data Pipeline (backup) | 9.0″ × 4.2″ | NEW |

Note: fig4_scope.png (Slide 7) is TBD — not yet specified.

## fig1 — AI Task Exposure
- **Data:** Eloundou et al. (2023) γ = E1 + E2, averaged to 2-digit SOC
- **Source:** `data/external/eloundou/occ_level.csv`
- **Takeaway:** 12–97% of tasks theoretically feasible across occupation groups

## fig2 — Distance Metric Comparison
- **Data:** Hardcoded from `outputs/experiments/distance_head_to_head_v0732.json`
- **Takeaway:** Embedding-based measures achieve 13.8–14.1% pseudo-R² vs 6–8% for O*NET — 74.9% improvement

## fig3 — Tasks in Semantic Space
- **Data:** MPNet embeddings of 30 selected DWAs, semantic anchor projection
- **Source:** `fig3_selected_dwas.csv` (30 DWAs, 6 clusters)
- **Takeaway:** Related tasks cluster in interpretable quadrants

## fig5 — Shared DWAs (Pipelayers vs Cement Masons)
- **Data:** Hardcoded from O*NET Tasks-to-DWAs mapping
- **Pair:** Pipelayers (47-2151.00) vs Cement Masons (47-2051.00)
- **Takeaway:** Only 4 of 40 DWAs overlap — O*NET cosine distance = 0.88

## fig6 — Embedding Similarity
- **Data:** Census-level distance matrices from `.cache/artifacts/v1/mobility/`
- **Pair:** Same as fig5 — Pipelayers (Census 6440) vs Cement Masons (Census 6250)
- **Takeaway:** Embedding distance = 0.15 (7th percentile) vs O*NET = 0.88 — embeddings detect similarity that O*NET misses

## fig7 — CPS Data Pipeline (Sankey)
- **Data:** Hardcoded from paper Table 1
- **Takeaway:** 10M raw records → 89K verified transitions through 5 filtering stages

## Shared style
- **Module:** `style.py` — colors, fonts, rcParams
- **Font:** DejaVu Sans (Calibri not available on build system)
- **Colors:** Steel blue (#4A7FB5) primary, warm gray (#A0937D) secondary
