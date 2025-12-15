# IPUMS CPS Extract Specification

## Project: CPS Mobility + Job Zone Wedge Analysis

---

## Variables Required

### Core Linking Variables

| Variable | Description | Purpose |
|----------|-------------|---------|
| **CPSIDP** | Person identifier | Primary longitudinal linking |
| **CPSIDV** | Validated person identifier | Links with demographic consistency checks |
| **MISH** | Month-in-sample (1-8) | Track rotation position |

### Occupation Variables

| Variable | Description | Purpose |
|----------|-------------|---------|
| **OCC2010** | Harmonized 2010 Census occupation | Primary occupation measure (comparable across years) |
| **OCC** | Original occupation codes | Robustness checks, more detail for some years |

### Employment Variables

| Variable | Description | Purpose |
|----------|-------------|---------|
| **EMPSTAT** | Employment status | Filter to employed (codes 10, 12) |
| **LABFORCE** | Labor force status | Additional employment filter |
| **CLASSWKR** | Class of worker | Employer change proxy (wage/salary vs self-employed) |

### Demographic Variables (for validation)

| Variable | Description | Purpose |
|----------|-------------|---------|
| **AGE** | Age in years | Link validation (should increase 0-1 per month) |
| **SEX** | Sex | Link validation (should be constant) |
| **RACE** | Race | Link validation (should be constant) |
| **EDUC** | Educational attainment | Control variable, potential heterogeneity analysis |

### Weights

| Variable | Description | Purpose |
|----------|-------------|---------|
| **PANLWT** | Panel weight | Proper weighting for longitudinal analysis |

### Time and Geography

| Variable | Description | Purpose |
|----------|-------------|---------|
| **YEAR** | Survey year | Time identifier |
| **MONTH** | Survey month | Time identifier |
| **STATEFIP** | State FIPS code | Geographic controls, potential regional analysis |

---

## Sample Selection

### Time Period

| Period | Start | End | Months | Notes |
|--------|-------|-----|--------|-------|
| Pre-COVID | 2015-01 | 2019-12 | 60 | Stable labor market, pre-pandemic |
| Post-COVID | 2022-01 | 2024-09 | 33 | Post-pandemic recovery |
| **EXCLUDE** | 2020-01 | 2021-12 | 24 | COVID disruption period |
| **Total** | - | - | **93** | |

### Universe Restrictions

| Restriction | Specification | Rationale |
|-------------|---------------|-----------|
| Age | 18-65 | Prime working age |
| Employment | EMPSTAT = 10, 12 (employed) | Occupation only meaningful for employed |
| Labor Force | LABFORCE = 2 (in labor force) | Exclude NILFs |
| Civilian | Exclude military households | Standard CPS practice |

---

## Extract Configuration (IPUMS Website)

### Data Format
- **Format:** Fixed-width or CSV
- **Structure:** Rectangular (person-level)
- **Case Selection:** Apply age and labor force filters

### Data Quality Flags
- Include data quality flags if available
- Include allocation flags for imputed values

---

## Expected File Size

| Component | Estimate |
|-----------|----------|
| Person-month observations | ~9.3 million |
| After age/employment filter | ~5.5 million |
| Variables (18) × bytes/var | ~100 bytes/row |
| Raw file size | ~550 MB |
| Compressed | ~100-150 MB |

---

## Post-Download Processing Steps

### Step 1: Load and Validate
```python
import pandas as pd

df = pd.read_csv("cps_extract.csv")
print(f"Observations: {len(df):,}")
print(f"Unique persons (CPSIDP): {df['CPSIDP'].nunique():,}")
print(f"Date range: {df['YEAR'].min()}-{df['MONTH'].min()} to {df['YEAR'].max()}-{df['MONTH'].max()}")
```

### Step 2: Create Time Index
```python
df['YEARMONTH'] = df['YEAR'] * 100 + df['MONTH']
df = df.sort_values(['CPSIDP', 'YEARMONTH'])
```

### Step 3: Apply Filters (use transition_filters.py)
```python
from transition_filters import apply_all_filters

df_filtered, stats = apply_all_filters(
    df,
    use_cpsidv=True,
    persistence_months=2,
    require_continuous_ee=True
)
```

### Step 4: Merge with O*NET Data
```python
# Load crosswalk
crosswalk = pd.read_csv("onet_to_census_improved.csv")
occ_to_census = crosswalk.set_index('census_2010')

# Merge with d_sem and d_inst matrices
# (Implementation depends on analysis structure)
```

---

## Data Quality Checks

### Linking Quality
- [ ] Verify CPSIDP links have consistent demographics
- [ ] Compare CPSIDP vs CPSIDV match rates
- [ ] Check for impossible age sequences (age decrease)

### Occupation Coding
- [ ] Verify OCC2010 distribution matches BLS benchmarks
- [ ] Check for unusual occupation transitions
- [ ] Compare pre/post 2020 occupation distributions

### Sample Representativeness
- [ ] Compare weighted employment totals to BLS reports
- [ ] Check demographic distributions against Census estimates
- [ ] Verify state-level employment patterns

---

## Notes and Caveats

### Known Issues

1. **2020 Code Break**: OCC2010 handles this via modal assignment, but some noise introduced
2. **CPSIDP False Positives**: Use CPSIDV when available; apply demographic validation
3. **Occupation Coding Error**: ~70% of raw transitions may be spurious (K&M 2004)
4. **Panel Attrition**: Only ~50% of sample linked month-to-month due to rotation

### Recommended Robustness Checks

1. Compare results using OCC vs OCC2010
2. Test sensitivity to persistence filter threshold (1, 2, 3 months)
3. Analyze separately for pre- vs post-COVID periods
4. Check heterogeneity by education level, age group, industry

---

## IPUMS Account and Extract Submission

### Account Setup
1. Create account at https://cps.ipums.org/cps/
2. Complete user agreement
3. Typical approval time: <24 hours

### Extract Submission
1. Select samples (2015-2019, 2022-2024 monthly files)
2. Add variables listed above
3. Apply case selection: AGE 18-65, LABFORCE = 2
4. Request fixed-width or CSV format
5. Submit and wait for email notification
6. Download via direct link or FTP

### Estimated Processing Time
- Small extract (<1M records): 10-30 minutes
- This extract (~5M records): 1-3 hours
- Download time: 30-60 minutes (depending on connection)

---

## Contact

For IPUMS support: https://ipums.org/support
For data questions: ipums@umn.edu
