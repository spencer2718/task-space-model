# CPS Data Assessment for Occupation Mobility Analysis

## Data Source Recommendation: IPUMS CPS

**Recommended:** IPUMS CPS (https://cps.ipums.org/)

**Reasons:**
- Provides CPSIDP for individual-level longitudinal linking (1976-2024)
- Offers harmonized OCC2010 variable for consistent occupation coding across years
- Well-documented 4-8-4 rotation pattern enables month-to-month matching
- ~60,000 households/month → approximately 100-120k employed adults

## Variables Needed for Extract

### Core Variables
| Variable | Description | Notes |
|----------|-------------|-------|
| CPSIDP | Person-level longitudinal ID | Enables cross-month linking |
| MISH | Month-in-sample (1-8) | Tracks rotation position |
| OCC2010 | Harmonized 2010 occupation | ~460 detailed codes |
| OCC | Original occupation code | More detail, less comparable |
| EMPSTAT | Employment status | Filter to employed |
| AGE, SEX, RACE | Demographics | For link validation |
| PANLWT | Panel weight | For proper weighting |
| YEAR, MONTH | Time identifiers | |

### Optional for Robustness
- EDUC: Education level
- IND/IND1990: Industry codes
- EARNWT: Earnings weight (if analyzing wages)

## Occupation Code Harmonization Strategy

### The 2020 Break Problem
- CPS switched from Census 2010 codes to Census 2018 codes in January 2020
- Pre-2020: OCC = Census 2010 codes (identical to OCC2010)
- Post-2020: OCC = Census 2018 codes (different)

### Solution: Use OCC2010
IPUMS provides harmonized OCC2010 that maps 2018 codes back to 2010 codes using Census crosswalks with modal assignment. This is the recommended approach.

**Caveat:** Some 2018 codes split or merge 2010 codes, introducing noise. Approximately 5-10% of occupations may have imperfect mapping.

### Alternative: Aggregate to Major Groups
For robustness, can aggregate to 2-digit SOC (22 major groups) which are stable across coding changes.

## Estimated Sample Size

### Monthly Numbers (approximate)
- Employed adults in sample: ~100,000/month
- Monthly occupation switch rate: ~2-5% (varies by definition and occupation)
- Occupation switchers per month: ~2,000-5,000

### Target Period Sample
| Period | Months | Est. Person-Months | Est. Switchers |
|--------|--------|-------------------|----------------|
| 2015-2019 | 60 | 6,000,000 | 120,000-300,000 |
| 2022-2024 | 36 | 3,600,000 | 72,000-180,000 |
| **Total** | 96 | 9,600,000 | 192,000-480,000 |

### Sparsity Concern
With ~460 OCC2010 codes:
- Possible pairs: 460 × 460 = 211,600
- Expected non-zero pairs: Maybe 5,000-20,000 (most pairs never observed)
- Average observations per non-zero pair: ~10-100

This is SPARSE but potentially sufficient for identification if we use distance-based regression rather than pair fixed effects.

## Known Issues/Limitations

### 1. Linkage Imperfections
- CPSIDP links ~75% of sample month-to-month
- Must validate with AGE, SEX, RACE consistency
- Some false positives in 2003-2005 ASEC samples

### 2. Measurement Error in Occupation Codes
- Self-reported occupation subject to error
- Same worker may report different occupation codes across months due to reporting variation (not actual job change)
- Some "occupation switches" are coding noise, not real transitions

### 3. Composition Changes
- 2020 COVID disruption affects 2020-2021 data quality
- Labor force composition changed during pandemic
- Consider dropping 2020-2021 or treating separately

### 4. Weight Complexity
- Panel weights (PANLWT) required for population estimates
- Weights become more complex with multi-month panels

### 5. Occupation vs Job Switching
- CPS measures occupation, not job
- A worker can switch jobs (employers) without switching occupation
- And can be coded in different occupation without switching jobs

## Crosswalk Path: O*NET-SOC → CPS OCC2010

### Available Crosswalks
1. **O*NET-SOC 2019 → SOC 2018**: O*NET Resource Center
2. **SOC 2018 → Census 2018**: BLS crosswalks
3. **Census 2018 → Census 2010**: Census Bureau crosswalks (used in IPUMS OCC2010)

### Expected Coverage Issues
- O*NET: ~923 detailed occupations (8-digit SOC)
- CPS OCC2010: ~460 codes (6-digit level)
- Many-to-one mapping: Multiple O*NET codes → single CPS code
- Aggregation required: Will lose some O*NET granularity

### Recommended Approach
1. Download crosswalk from O*NET Center (O*NET-SOC → SOC 2018)
2. Use Census crosswalk (SOC 2018 → Census 2018)
3. Use IPUMS internal mapping (Census 2018 → OCC2010)
4. For O*NET codes without direct mapping, use 6-digit prefix aggregation

## Alternative Data Sources (If CPS Proves Insufficient)

### SIPP (Survey of Income and Program Participation)
- True panel (same individuals over 2.5 years)
- Better for occupation transitions
- Smaller sample (~30,000 households)
- More complex design

### NLSY (National Longitudinal Survey of Youth)
- True panel following cohorts
- Detailed occupation histories
- Very small sample
- Age-specific (not population representative)

### PSID (Panel Study of Income Dynamics)
- Long panel (50+ years)
- Small sample (~10,000 families)
- Good for long-run mobility

### LEHD/J2J
- Administrative data (employer records)
- Large sample
- Industry-based, NOT occupation-based
- Not suitable for this analysis

## Conclusion

**Feasibility Assessment: CAUTIOUSLY OPTIMISTIC**

- CPS can provide the data needed
- Sparsity is a concern but likely manageable with distance-based regression
- OCC2010 harmonization handles the 2020 code break
- Should test with synthetic data before committing to full extract

**Recommended Next Steps:**
1. Run sparsity simulation (Task 3) to confirm sample size is adequate
2. Test crosswalk coverage (Task 4) before extract
3. If feasible, create IPUMS extract with variables above
4. Start with 2015-2019 (pre-COVID, stable codes) for pilot
