# Implementation Specification v0.4.2: Recipe Y + DWA Domain Pivot

**Version:** 0.4.2  
**Status:** Post-Validation Pivot  
**Date:** December 2025  
**From:** Research Coordination  
**To:** Software Engineering Agent

---

## Executive Summary

Phase I validation (v0.4.1) failed: the GWA-based Recipe X geometry does not predict wage comovement (β < 0, p > 0.70 across all σ values). Per Section 4.4 of the paper, we now execute the pre-committed pivot:

1. **Domain:** GWA (41 activities) → **DWA (~2,087 activities)**
2. **Distance metric:** Recipe X (rating-cooccurrence PCA) → **Recipe Y (text embeddings)**
3. **Validation target:** Wage comovement → **Worker mobility** (recommended addition)

This document provides complete implementation specifications for all three changes.

---

## Part 1: DWA Domain Implementation

### 1.1 Rationale

The GWA domain (41 activities) approximates a histogram, not a manifold. The DWA domain (~2,087 activities) provides sufficient granularity for meaningful geometric structure:

- GWA: "Processing Information" (one bucket)
- DWA: 50+ distinct information-processing activities (e.g., "Analyze financial data," "Review budget variances," "Interpret statistical reports")

### 1.2 O*NET Files Required

Download from `https://www.onetcenter.org/database.html` (O*NET 30.0):

| File | Purpose | Key Columns |
|------|---------|-------------|
| `DWA Reference.xlsx` | DWA definitions & hierarchy | `DWA ID`, `DWA Title`, `IWA ID`, `Element ID` |
| `Tasks to DWAs.xlsx` | Task→DWA mappings | `O*NET-SOC Code`, `Task ID`, `DWA ID`, `DWA Title` |
| `Task Ratings.xlsx` | Task importance ratings | `O*NET-SOC Code`, `Task ID`, `Scale ID`, `Data Value` |
| `Task Statements.xlsx` | Core vs Supplemental tasks | `O*NET-SOC Code`, `Task ID`, `Task Type` |

### 1.3 DWA Hierarchy Structure

```
GWA (41 activities, Element ID: 4.A.X.X.X)
  └── IWA (332 intermediate activities, ID: 4.A.X.X.X.IXX)
        └── DWA (2,087 detailed activities, ID: 4.A.X.X.X.IXX.DXX)
              └── Tasks (19,000+ occupation-specific statements)
```

Each DWA links to exactly one IWA, which links to exactly one GWA. This hierarchy enables fallback aggregation if DWA proves too sparse.

### 1.4 Constructing DWA Occupation Measures

DWAs are not directly rated—importance is derived through task linkages. Per O*NET methodology ("Ranking Detailed Work Activities Within O*NET Occupational Profiles", April 2015):

```python
def build_dwa_occupation_matrix(
    tasks_to_dwas_df,
    task_ratings_df,
    task_statements_df=None,
    aggregator='max'
):
    """
    Build occupation × DWA importance matrix.
    
    Parameters:
    -----------
    tasks_to_dwas_df : DataFrame from Tasks to DWAs.xlsx
    task_ratings_df : DataFrame from Task Ratings.xlsx  
    task_statements_df : Optional, for core task filtering
    aggregator : 'max' (O*NET default), 'mean', or 'weighted_mean'
    
    Returns:
    --------
    matrix : (n_occupations, n_dwas) normalized to probability measures
    dwa_ids : list of DWA identifiers
    occ_codes : list of O*NET-SOC codes
    """
    
    # Step 1: Get task importance ratings (Scale ID = 'IM')
    task_imp = task_ratings_df[
        task_ratings_df['Scale ID'] == 'IM'
    ][['O*NET-SOC Code', 'Task ID', 'Data Value']].copy()
    
    # Optional: Filter to core tasks only (Relevance ≥ 67%, Importance ≥ 3.0)
    if task_statements_df is not None:
        core_tasks = task_statements_df[
            task_statements_df['Task Type'] == 'Core'
        ]['Task ID'].unique()
        task_imp = task_imp[task_imp['Task ID'].isin(core_tasks)]
    
    # Step 2: Join with task-DWA mappings
    merged = tasks_to_dwas_df.merge(
        task_imp,
        on=['O*NET-SOC Code', 'Task ID'],
        how='inner'
    )
    
    # Step 3: Aggregate to occupation-DWA level
    if aggregator == 'max':
        # O*NET default: max importance across linked tasks
        dwa_importance = merged.groupby(
            ['O*NET-SOC Code', 'DWA ID']
        )['Data Value'].max().reset_index()
    elif aggregator == 'mean':
        dwa_importance = merged.groupby(
            ['O*NET-SOC Code', 'DWA ID']
        )['Data Value'].mean().reset_index()
    
    # Step 4: Pivot to matrix form
    matrix = dwa_importance.pivot_table(
        index='O*NET-SOC Code',
        columns='DWA ID',
        values='Data Value',
        fill_value=1.0  # Minimum importance for unlinked DWAs
    )
    
    # Step 5: Normalize importance to [0,1]
    matrix = (matrix - 1) / 4  # IM scale is [1,5]
    
    # Step 6: Normalize rows to probability measures
    row_sums = matrix.sum(axis=1)
    matrix = matrix.div(row_sums, axis=0)
    
    return matrix, list(matrix.columns), list(matrix.index)
```

### 1.5 Expected Dimensions

| Component | GWA (v0.4.1) | DWA (v0.4.2) |
|-----------|--------------|--------------|
| Activity domain | 41 | ~2,087 |
| Occupation measures | 894 × 41 | ~900 × 2,087 |
| Distance matrix | 41 × 41 | 2,087 × 2,087 |
| Kernel matrix | 41 × 41 | 2,087 × 2,087 |
| Overlap matrix | 894 × 894 | ~900 × 900 |

**Memory considerations:** The DWA distance matrix (2087² × 8 bytes ≈ 35 MB) and kernel matrix (same) are manageable. The occupation overlap matrix remains O(n²) in occupations, unchanged.

### 1.6 Sparsity Handling

Many occupations will have sparse DWA profiles. Document:

1. **Effective support:** |{a : ρⱼ(a) > 0.01}| per occupation
2. **Coverage:** % of DWAs with at least one occupation linkage
3. **Flag criterion:** Occupations with effective support < 30 may need review

---

## Part 2: Recipe Y (Text Embedding) Implementation

### 2.1 Rationale

Recipe X (rating-cooccurrence geometry) failed validation. Recipe Y captures semantic similarity between activity descriptions, independent of Likert-scale rating patterns.

Key insight: "Analyze financial data" and "Review budget variances" are semantically close even if different occupational analysts gave them different importance scores.

### 2.2 Model Selection

**Recommended:** `sentence-transformers/all-mpnet-base-v2`

| Model | Dimensions | Speed | Quality (STS-B) | Recommendation |
|-------|------------|-------|-----------------|----------------|
| `all-MiniLM-L6-v2` | 384 | ~14k sent/sec | 84-85% | Robustness check |
| `all-MiniLM-L12-v2` | 384 | ~7k sent/sec | 85-86% | Alternative |
| **`all-mpnet-base-v2`** | **768** | **~2.5k sent/sec** | **87-88%** | **Primary** |

Per SBERT documentation: "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."

For ~2,087 DWA titles, encoding takes <1 second on GPU, <30 seconds on CPU. Speed is not a constraint.

### 2.3 Installation

```bash
pip install sentence-transformers torch
```

### 2.4 Implementation

```python
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist, squareform
import numpy as np

def compute_text_embedding_distances(
    activities: list[str],
    model_name: str = 'all-mpnet-base-v2',
    metric: str = 'cosine'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise distances between activity descriptions using text embeddings.
    
    Parameters:
    -----------
    activities : list of activity description strings (DWA titles)
    model_name : sentence-transformers model identifier
    metric : 'cosine' (recommended) or 'euclidean'
    
    Returns:
    --------
    distances : (n_activities, n_activities) distance matrix
    embeddings : (n_activities, embedding_dim) raw embeddings for inspection
    """
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Encode all activities
    print(f"Encoding {len(activities)} activities...")
    embeddings = model.encode(
        activities,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Compute pairwise distances
    # cosine distance = 1 - cosine_similarity
    distances = squareform(pdist(embeddings, metric=metric))
    
    return distances, embeddings


def extract_dwa_titles(dwa_reference_df) -> tuple[list[str], list[str]]:
    """
    Extract DWA titles for embedding.
    
    Returns:
    --------
    titles : list of DWA title strings
    dwa_ids : list of corresponding DWA IDs (for alignment)
    """
    # DWA Reference has columns: Element ID, IWA ID, DWA ID, DWA Title
    dwa_df = dwa_reference_df[['DWA ID', 'DWA Title']].drop_duplicates()
    dwa_df = dwa_df.sort_values('DWA ID')
    
    return list(dwa_df['DWA Title']), list(dwa_df['DWA ID'])
```

### 2.5 Distance Metric Choice

**Primary:** Cosine distance (1 - cosine_similarity)
- Standard for sentence embeddings
- Scale-invariant
- Interpretable: 0 = identical, 2 = opposite

**Robustness:** Euclidean distance on L2-normalized embeddings
- Equivalent to cosine for normalized vectors
- Useful for certain downstream analyses

### 2.6 Validation: Recipe X vs Recipe Y Concordance

Before running Phase I validation, check that Recipe X and Recipe Y geometries are related but not redundant:

```python
def compare_geometries(dist_recipe_x, dist_recipe_y):
    """
    Compare two distance matrices.
    
    High correlation (>0.9): Geometries are redundant
    Low correlation (<0.3): Geometries capture different structure
    Moderate correlation (0.3-0.7): Expected for valid alternatives
    """
    # Flatten upper triangles
    triu_idx = np.triu_indices(dist_recipe_x.shape[0], k=1)
    flat_x = dist_recipe_x[triu_idx]
    flat_y = dist_recipe_y[triu_idx]
    
    # Pearson and Spearman correlations
    from scipy.stats import pearsonr, spearmanr
    
    pearson_r, _ = pearsonr(flat_x, flat_y)
    spearman_r, _ = spearmanr(flat_x, flat_y)
    
    return {
        'pearson': pearson_r,
        'spearman': spearman_r,
        'interpretation': 'redundant' if spearman_r > 0.9 else 
                         'distinct' if spearman_r < 0.3 else 'related'
    }
```

Expected: Moderate correlation (0.4-0.7). If >0.9, Recipe Y won't add information; if <0.2, investigate whether one is pathological.

---

## Part 3: Alternative Validation Targets

### 3.1 Rationale for Worker Mobility

Wage comovement is confounded by:
- Industry-level shocks (oil prices affect all oil workers)
- Business cycle sensitivity
- Institutional factors (unionization, minimum wage exposure)

**Worker mobility directly measures the "reallocation friction" that activity distances are meant to encode.** If I can do your job (capability transfer), I can switch to it. The transition rate measures revealed similarity.

### 3.2 Data Source Options

#### Option A: CPS ASEC with OCC10LY (Recommended)

**Source:** IPUMS CPS (https://cps.ipums.org)  
**Registration:** Required (free)  
**Data type:** Cross-sectional with retrospective occupation

| Variable | Description |
|----------|-------------|
| `OCC2010` | Current occupation (2010 Census harmonized) |
| `OCC10LY` | Occupation last year (2010 Census harmonized) |
| `ASECWT` | Person weight for ASEC |
| `YEAR` | Survey year |

**Universe:** Civilians age 15+ who worked last year  
**Availability:** ASEC samples 1968-2025

**Key advantage:** OCC10LY is harmonized to 2010 Census codes, enabling consistent time-series construction.

#### Option B: CPS Basic Monthly with Panel Linking

**Source:** IPUMS CPS basic monthly files  
**Method:** Link individuals across months using CPSIDP

| Variable | Description |
|----------|-------------|
| `OCC2010` | Current occupation |
| `CPSIDP` | Person-level identifier for linking |
| `PANLWT` | Panel weight for linked samples |
| `MISH` | Month-in-sample (rotation group) |

**Panel structure:** 4-8-4 rotation (interview 4 months, out 8, interview 4)

**Caution:** Occupation transitions measured month-to-month are inflated by coding errors and proxy responses. See Kambourov & Manovskii (2013): measurement error in either period registers as a "transition." Use coarser occupation categories or longer time spans to mitigate.

#### Option C: SIPP Longitudinal

**Source:** Census Bureau SIPP (https://www.census.gov/sipp/)  
**Data type:** True panel (same individuals over 4+ years)

| Variable | Description |
|----------|-------------|
| `TJBOCC1`, `TJBOCC2` | Occupation codes for jobs 1 and 2 |
| Wave identifiers | Link records across time |

**Advantage:** Tracks same individuals over time (true transitions)  
**Disadvantage:** Complex data structure, occupation coding changes across panels

#### Option D: CPS Displaced Worker / Job Tenure Supplement

**Source:** IPUMS CPS January supplements (biennial)  
**Variables:** `JTOCC` (occupation 1 year ago), `JTCLASS`, etc.

**Advantage:** Direct question about occupation change  
**Disadvantage:** Less frequent, smaller sample

### 3.3 Recommended Approach: CPS ASEC

**Why ASEC over monthly panel:**
1. OCC10LY provides direct "occupation last year" without panel linking
2. Harmonized codes (OCC10LY) reduce coding error inflation
3. Larger effective sample (entire ASEC vs. linkable panel subset)
4. Simpler implementation

**Sample years:** 2015-2019 (pre-COVID, consistent coding)

### 3.4 Constructing the Mobility Matrix

```python
import pandas as pd
import numpy as np

def construct_mobility_matrix(
    cps_asec_df,
    weight_var='ASECWT',
    min_cell_count=50
):
    """
    Construct occupation-to-occupation transition matrix from CPS ASEC.
    
    Parameters:
    -----------
    cps_asec_df : DataFrame with OCC2010, OCC10LY, ASECWT
    weight_var : weight variable name
    min_cell_count : minimum unweighted count to include cell
    
    Returns:
    --------
    mobility_rate : (n_occ, n_occ) matrix of transition rates
    occ_list : list of occupation codes
    coverage : diagnostic statistics
    """
    
    df = cps_asec_df.copy()
    
    # Filter to valid occupation codes (exclude NIU = 9999)
    df = df[
        (df['OCC2010'] > 0) & (df['OCC2010'] < 9900) &
        (df['OCC10LY'] > 0) & (df['OCC10LY'] < 9900)
    ]
    
    # Compute weighted transition counts
    transitions = df.groupby(['OCC10LY', 'OCC2010'])[weight_var].sum().reset_index()
    transitions.columns = ['origin_occ', 'dest_occ', 'weighted_flow']
    
    # Also get unweighted counts for cell suppression
    cell_counts = df.groupby(['OCC10LY', 'OCC2010']).size().reset_index()
    cell_counts.columns = ['origin_occ', 'dest_occ', 'unweighted_n']
    
    transitions = transitions.merge(cell_counts, on=['origin_occ', 'dest_occ'])
    
    # Suppress small cells
    transitions = transitions[transitions['unweighted_n'] >= min_cell_count]
    
    # Compute origin totals
    origin_totals = transitions.groupby('origin_occ')['weighted_flow'].sum()
    
    # Compute transition rates
    transitions['mobility_rate'] = (
        transitions['weighted_flow'] / 
        transitions['origin_occ'].map(origin_totals)
    )
    
    # Pivot to matrix
    occ_list = sorted(set(transitions['origin_occ']) | set(transitions['dest_occ']))
    
    mobility_matrix = pd.DataFrame(
        0.0, 
        index=occ_list, 
        columns=occ_list
    )
    
    for _, row in transitions.iterrows():
        mobility_matrix.loc[row['origin_occ'], row['dest_occ']] = row['mobility_rate']
    
    # Coverage statistics
    n_occs = len(occ_list)
    n_cells = (mobility_matrix > 0).sum().sum()
    
    coverage = {
        'n_occupations': n_occs,
        'n_nonzero_cells': n_cells,
        'sparsity': 1 - n_cells / (n_occs ** 2),
        'total_flow': transitions['weighted_flow'].sum()
    }
    
    return mobility_matrix.values, occ_list, coverage
```

### 3.5 Crosswalk: O*NET-SOC ↔ Census Occupation Codes

**Key files from BLS:**
- `nem-occcode-cps-crosswalk.xlsx`: SOC → CPS Census codes
- Download: https://www.bls.gov/emp/documentation/crosswalks.htm

**Challenge:** O*NET-SOC (~894 occupations with data) is more granular than Census codes (~540 in 2010 scheme).

**Solution:** Aggregate O*NET occupations to Census level

```python
def aggregate_to_census_level(
    onet_measures,
    onet_soc_codes,
    crosswalk_df
):
    """
    Aggregate O*NET occupation measures to Census occupation level.
    
    Parameters:
    -----------
    onet_measures : (n_onet_occ, n_activities) matrix
    onet_soc_codes : list of O*NET-SOC codes
    crosswalk_df : SOC to Census crosswalk
    
    Returns:
    --------
    census_measures : (n_census_occ, n_activities) aggregated matrix
    census_codes : list of Census occupation codes
    """
    
    # O*NET-SOC format: XX-XXXX.XX
    # SOC format: XX-XXXX
    # Census 2010 format: XXXX (4-digit)
    
    # Build mapping
    mapping = {}  # census_code -> [onet_indices]
    
    for i, onet_code in enumerate(onet_soc_codes):
        soc_6digit = onet_code[:7]  # Strip O*NET suffix
        
        # Look up in crosswalk
        match = crosswalk_df[crosswalk_df['SOC_CODE'] == soc_6digit]
        if len(match) > 0:
            census_code = match.iloc[0]['CENSUS_CODE']
            if census_code not in mapping:
                mapping[census_code] = []
            mapping[census_code].append(i)
    
    # Aggregate by averaging (could weight by employment)
    census_codes = sorted(mapping.keys())
    census_measures = np.zeros((len(census_codes), onet_measures.shape[1]))
    
    for j, census_code in enumerate(census_codes):
        indices = mapping[census_code]
        census_measures[j, :] = onet_measures[indices, :].mean(axis=0)
        # Re-normalize to probability measure
        census_measures[j, :] /= census_measures[j, :].sum()
    
    return census_measures, census_codes
```

### 3.6 Important Caveat: Measurement Error in Transitions

From IPUMS Forum discussion and Kambourov & Manovskii (2013):

> "The number of measured occupational transitions tends to be inflated in the CPS, especially when looking at the most detailed level... This is due to inaccurate proxy responses, miscoding, and imputation. For month-to-month transitions, any change in either period due to error is recorded as an occupation transition."

**Mitigations:**
1. Use coarser occupation categories (major groups) as robustness check
2. Use annual transitions (OCC10LY) rather than monthly
3. Require substantial flow (min_cell_count ≥ 50) to reduce noise
4. Interpret as relative signal, not absolute rates

---

## Part 4: Validation Protocol (Updated)

### 4.1 Diagnostic B (Revised)

**Regression specification:**

```
Mobility_{i,j} = α + β × Overlap_{i,j}(σ) + γ × Controls_{i,j} + ε_{i,j}
```

where:
- `Mobility_{i,j}` = transition rate from occupation i to occupation j
- `Overlap_{i,j}(σ)` = kernel-weighted activity overlap at bandwidth σ
- `Controls` = education distance, wage distance, industry similarity
- SEs clustered by origin occupation

**Pass criterion:** β > 0, p < 0.10

### 4.2 Diagnostic Matrix: What to Run

| Domain | Distance Recipe | Validation Target | Label |
|--------|-----------------|-------------------|-------|
| GWA (41) | Recipe X (PCA) | Wage comovement | v0.4.1 ✗ |
| **DWA (~2,087)** | **Recipe Y (embedding)** | **Wage comovement** | **v0.4.2a** |
| **DWA (~2,087)** | **Recipe Y (embedding)** | **Worker mobility** | **v0.4.2b** |
| DWA (~2,087) | Recipe X (PCA) | Worker mobility | v0.4.2c (robustness) |
| GWA (41) | Recipe Y (embedding) | Worker mobility | v0.4.2d (robustness) |

**Primary tests:** v0.4.2a and v0.4.2b  
**If both fail:** Framework may require fundamental revision

### 4.3 Decision Rules

1. **v0.4.2a passes (DWA + Recipe Y + Wage):**  
   Domain granularity was the issue. Proceed to Phase II with DWA + Recipe Y.

2. **v0.4.2b passes (DWA + Recipe Y + Mobility):**  
   Validation target was the issue. Use mobility as validation; proceed with DWA + Recipe Y.

3. **Both v0.4.2a and v0.4.2b pass:**  
   Strong geometry. Report both validation results.

4. **Both fail:**  
   Check v0.4.2c and v0.4.2d to isolate whether domain or recipe is the problem. If all fail, document null result.

---

## Part 5: Implementation Checklist

### 5.1 New Dependencies

```bash
pip install sentence-transformers torch scipy pandas numpy statsmodels
```

### 5.2 New/Modified Modules

| Module | Changes |
|--------|---------|
| `domain.py` | Add `build_dwa_domain()`, `build_dwa_occupation_measures()` |
| `distances.py` | Add `compute_text_distances()`, keep `compute_rating_distances()` for comparison |
| `validation.py` | Add `construct_mobility_matrix()`, `aggregate_to_census_level()` |
| `diagnostics.py` | Update for larger matrices, add geometry comparison |

### 5.3 Data Acquisition Checklist

- [ ] O*NET 30.0 `DWA Reference.xlsx`
- [ ] O*NET 30.0 `Tasks to DWAs.xlsx`
- [ ] O*NET 30.0 `Task Ratings.xlsx`
- [ ] O*NET 30.0 `Task Statements.xlsx` (optional, for core task filter)
- [ ] BLS crosswalk `nem-occcode-cps-crosswalk.xlsx`
- [ ] IPUMS CPS ASEC extract (OCC2010, OCC10LY, ASECWT, YEAR) 2015-2019

### 5.4 Output Requirements

1. **DWA domain statistics:**
   - Number of DWAs after filtering
   - Occupation coverage (% with linkages)
   - Effective support distribution

2. **Geometry comparison:**
   - Recipe X vs Recipe Y distance correlation
   - Nearest-neighbor stability across recipes

3. **Validation results table:**
   - β, SE, p-value, R² for each (domain × recipe × target) combination
   - Monotonicity plots (overlap decile vs outcome)

4. **Decision documentation:**
   - Which combination(s) passed
   - Recommended configuration for Phase II
   - Limitations and caveats

---

## Part 6: Timeline and Priorities

### Phase 1: DWA + Recipe Y Implementation (Days 1-2)

1. Implement `build_dwa_occupation_measures()`
2. Implement `compute_text_distances()` with all-mpnet-base-v2
3. Verify: 2,087 × 2,087 distance matrix, ~900 × ~2,087 occupation measures
4. Run geometry comparison vs GWA Recipe X

### Phase 2: Validation Data Acquisition (Days 2-3)

1. Register for IPUMS CPS (if not already)
2. Extract ASEC data: OCC2010, OCC10LY, ASECWT, YEAR (2015-2019)
3. Download BLS crosswalk
4. Implement occupation code mapping

### Phase 3: Run Validations (Days 3-4)

1. Compute DWA overlap matrix
2. Run v0.4.2a: DWA + Recipe Y + Wage comovement
3. Construct mobility matrix from CPS
4. Run v0.4.2b: DWA + Recipe Y + Worker mobility
5. Run robustness checks (v0.4.2c, v0.4.2d)

### Phase 4: Documentation (Day 5)

1. Update README with results
2. Document decision: proceed to Phase II or report null
3. Archive all diagnostic outputs

---

## Appendix A: Model Download and Caching

First run will download model weights (~420 MB for all-mpnet-base-v2):

```python
from sentence_transformers import SentenceTransformer

# Downloads to ~/.cache/torch/sentence_transformers/
model = SentenceTransformer('all-mpnet-base-v2')
```

For offline environments, pre-download:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
```

---

## Appendix B: Sample DWA Titles (for verification)

```
4.A.1.a.1.I01.D01: Collect information from people through observation, interviews, or surveys
4.A.1.a.1.I01.D02: Read materials to gather technical, scientific, or other specialized information
4.A.1.a.2.I02.D01: Evaluate characteristics of raw materials or finished goods
4.A.2.a.1.I03.D01: Estimate material requirements for projects
4.A.2.b.1.I04.D01: Analyze business or financial data
4.A.2.b.2.I05.D01: Design control systems for mechanical or other equipment
4.A.4.a.1.I06.D01: Write reports or evaluations
4.A.4.a.2.I07.D01: Give speeches or talks
...
```

Verify that embedding similarities make semantic sense:
- "Analyze financial data" should be close to "Review budget variances"
- "Give speeches or talks" should be close to "Deliver presentations"
- "Repair machinery" should be close to "Maintain industrial equipment"

---

## Appendix C: Key References

1. **O*NET DWA Methodology:**  
   "Ranking Detailed Work Activities Within O*NET Occupational Profiles"  
   https://www.onetcenter.org/dl_files/Related_DWA.pdf

2. **Sentence Transformers:**  
   https://www.sbert.net/docs/sentence_transformer/pretrained_models.html

3. **CPS Occupation Transition Measurement Error:**  
   Kambourov, G. & Manovskii, I. (2013). "A Cautionary Note on Using (March) Current Population Survey and Panel Study of Income Dynamics Data to Study Worker Mobility"

4. **BLS Crosswalks:**  
   https://www.bls.gov/emp/documentation/crosswalks.htm

5. **IPUMS CPS Documentation:**  
   https://cps.ipums.org/cps-action/variables/OCC10LY

---

**End of Specification v0.4.2**