# Presentation Figures

| File | Target slide | Size | Takeaway | Data source |
|------|-------------|------|----------|-------------|
| fig1_ai_exposure.png | Slide 2 (Motivation) | 16:9 (9.0" x 6.5") | AI can theoretically perform 12–97% of tasks across occupation categories — motivates need for distance measures | Eloundou et al. (2023) γ scores |
| fig2_pseudo_r2.png | Slide 5 (Main Result: Embeddings Dominate) | 16:9 (9.0" x 5.0") | Embedding-based measures achieve 13.8–14.1% pseudo-R², vs 6–8% for O*NET — ~75% relative improvement. Ground metric swap improves R² by over 80%. | Tables 2 & 3 (hardcoded) |
| fig3_task_scatter.png | Slide 3 (Core Idea: Tasks in Semantic Space) | Half-slide (7.5" x 5.0") | Semantically related tasks cluster together in embedding space — vehicle operation near vehicle operation, quantitative analysis near quantitative analysis | MPNet embeddings of 30 selected DWAs via UMAP |

## fig1_ai_exposure.png
- **Target slide:** 2 (Motivation)
- **Size:** 16:9 full-slide (9.0" x 6.5" — tall to accommodate 22 bars)
- **Data:** Eloundou et al. (2023) γ = E1 + E2 scores, averaged to 2-digit SOC major groups (22 categories, unweighted)
- **Source file:** data/external/eloundou/occ_level.csv
- **Takeaway:** AI can theoretically perform tasks across most occupation categories — from 97% of Computer & Math tasks to 12% of Construction. Motivates the need for occupational distance measures.
- **Title:** Omitted (slide title covers it)
- **Font:** DejaVu Sans (Calibri not available on this system)
- **Variant:** horizontal bar chart, sorted by γ descending
- **Radial variants** retained as fig1_radial_ai_exposure_byvalue.png and fig1_radial_ai_exposure_bysoc.png

## fig3_task_scatter.png
- **Target slide:** 3 (Core Idea: Tasks in Semantic Space)
- **Size:** Half-slide (7.5" x 5.0")
- **Data:** MPNet embeddings of 30 hand-selected DWAs, projected via semantic anchor axes
- **Axes:** X = cosine_sim(non_routine) − cosine_sim(routine), rescaled to ±5; Y = cosine_sim(cognitive) − cosine_sim(manual), rescaled to ±5. Interpretable dimensions: Routine↔Non-Routine, Manual↔Cognitive.
- **Takeaway:** Semantically related tasks cluster in interpretable quadrants — quantitative/technology tasks in the upper half (cognitive), vehicle/construction in the lower half (manual), healthcare in the left-center. Embedding geometry captures meaningful task similarity.
- **Title:** Omitted (slide title covers it)
- **Font:** DejaVu Sans (Calibri not available on this system)
- **Selected DWAs:** see fig3_selected_dwas.csv (30 DWAs across 6 themes, 14 labeled)
- **Legend:** Lower-right corner, color-keyed to 6 clusters
- **Clusters:** Healthcare (red), Vehicle & Equipment (orange), Construction (gray), Quantitative (blue), Communication (green), Technology (purple)

## fig2_pseudo_r2.png
- **Target slide:** 5 (Main Result: Embeddings Dominate)
- **Size:** 16:9 full-slide (9.0" x 5.0")
- **Data:** Table 2 (4 specifications), Table 3 (ground metric comparison)
- **Takeaway:** Embedding-based measures achieve 13.8–14.1% pseudo-R², vs 6–8% for O*NET — a ~75% relative improvement. Ground metric swap improves R² by over 80%.
- **Title:** Omitted (slide title covers it)
- **Font:** DejaVu Sans (Calibri not available on this system)
