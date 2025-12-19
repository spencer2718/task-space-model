# Data Sources

External data required for the task-space oracle. Specifications match `paper/main.tex` (External Data Integration).

---

## Required Data

### O*NET Database

- **Version:** 30.0
- **Download:** https://www.onetcenter.org/database.html
- **File:** `db_30_0_excel.zip`
- **Extract to:** `data/onet/db_30_0_excel/`
- **Contents:** 2,087 Detailed Work Activities (DWAs) linked to 894 occupations

### CPS Microdata (IPUMS)

- **Source:** IPUMS CPS Basic Monthly Files
- **Period:** 2015-01 through 2024-09 (excluding 2020-2021 for COVID)
- **Variables:** CPSIDP, YEARMONTH, OCC2010, AGE, SEX, RACE, EMPSTAT
- **Sample:** 89,329 verified transitions after filtering
- **Download:** https://cps.ipums.org/cps/ (requires free IPUMS account)

### Census Microdata (IPUMS) — Retrospective Battery

- **Source:** IPUMS USA 5% samples
- **Years:** 1980, 1990, 2000
- **Variables:** STATEFIP, PUMA/CNTYGP98, OCC/OCC1950/OCC1990, PERWT, AGE, EMPSTAT
- **Location:** `data/external/ipums/census/`
- **Purpose:** CZ × occupation employment matrices for Test B
- **Download:** Via `ipumspy` API (see `src/task_space/battery/ipums_extract.py`)

---

## Optional Data

### OES Wage Data

- **Source:** BLS Occupational Employment and Wage Statistics
- **Years:** 2015-2023
- **Download:** See `data/external/oes/README.md` for direct links
- **Note:** BLS blocks automated downloads; use browser

### AIOE (AI Occupational Exposure)

- **Source:** Felten, Raj, Seamans (2021)
- **Download:** https://github.com/AIOE-Data/AIOE
- **Extract to:** `data/external/aioe/`
- **Coverage:** 93.5% of CPS sample occupations

### BLS Employment Projections

- **Source:** BLS Occupational Outlook Handbook
- **Coverage:** 94.3% of sample occupations
- **Used for:** Demand decomposition analysis

---

## Historical/Planned Data

These exist in `data/external/` for retrospective tests (main.tex Appendix A) or future phases:

| Data | Location | Purpose | Status |
|------|----------|---------|--------|
| Dorn replication | `data/external/dorn_replication/` | Appendix A Test B (1980-2005) | Available |
| IPUMS Census (1980-2000) | `data/external/ipums/census/` | CZ employment matrices | Available (v0.7.2.3) |
| Dingel-Neiman telework | `data/external/dingel_neiman/` | Telework shock profiles | Available |
| JOLTS vacancy rates | — | Demand-side integration | Phase 0.8 |
| CPS licensing supplement | — | Institutional barriers | Phase 0.9 |
| LEHD individual wages | — | Switching cost identification | Phase 1.0 |

---

## Directory Structure

```
data/
├── onet/db_30_0_excel/     # O*NET database (required)
├── external/
│   ├── oes/                # OES wage data
│   ├── aioe/               # AI exposure scores
│   ├── dorn_replication/   # Historical (Appendix A)
│   ├── dingel_neiman/      # Telework scores
│   └── ipums/              # CPS extracts
└── processed/              # Generated artifacts
```

---

## Consistency Note

Sample sizes and specifications must match `paper/main.tex` (External Data Integration):

| Specification | Value |
|--------------|-------|
| CPS verified transitions | 89,329 |
| CPS training sample | 97,236 |
| CPS holdout sample | 8,880 |
| OES coverage | 98.2% |
| AIOE coverage | 93.5% |
| BLS openings coverage | 94.3% |
