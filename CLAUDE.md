# CLAUDE.md - Developer Guide for Future Claudes

This document contains working conventions, version control rules, and quality-of-life information for AI assistants working on this project.

---

## Version Control

**Paper and codebase versions must always match.**

- Versions are bumped when entering a new phase of the research program
- Current: v0.4.0 (empirical implementation of Section 4)
- Previous: v0.3.7 (theoretical framework complete)

When updating either paper or code, ensure the other stays in sync or is updated together.

---

## Referencing the Paper

**Reference definitions by name, not number.** Definition numbers may change as sections reorganize.

Good:
```python
# Implements the normalized spillover operator (Definition: Spillover Operator)
```

Bad:
```python
# Implements Definition 3.5
```

Key definitions to reference by name:
- Task domain
- Occupation measures
- Technological state / Shock profile
- Normalized spillover operator
- Baseline exposure construction
- Occupation-level exposure functionals
- Reduced-form outcome equations

---

## Before Big Implementations: Use the Architect Agent

**Important:** Spencer has a software architect research agent that does deep research on both economics and system design.

Before implementing major components, request that the architect agent be consulted to:
- Plan the implementation strategy
- Verify economic assumptions against literature
- Identify potential issues with data structure
- Design the module architecture

Do not proceed with large implementations without this planning step.

---

## O*NET Data Reference (v0.4)

This section documents O*NET database structure based on architect research (December 2025).

### Data Source

**Use downloaded database files, not API.** Download `db_30_0_excel.zip` from:
- https://www.onetcenter.org/database.html
- License: Creative Commons Attribution 4.0
- Version: 30.0 (current as of December 2025)

Place extracted files in `data/onet/` directory.

### Activity Domain Options

| Domain | Dimension | Source | Recommended |
|--------|-----------|--------|-------------|
| GWA (Generalized Work Activities) | 41 | Direct ratings | Yes (v0.4) |
| DWA (Detailed Work Activities) | 2,087 | Derived via tasks | Later |

**v0.4 uses GWA-based domain** (simpler, direct ratings, well-validated).

### Key Files

| File | Purpose | Rows |
|------|---------|------|
| `Work Activities.xlsx` | GWA ratings by occupation | ~79,376 |
| `Tasks to DWAs.xlsx` | Task-DWA mappings | ~23,000 |
| `Task Ratings.xlsx` | Task importance ratings | varies |
| `DWA Reference.xlsx` | DWA hierarchy (GWA→IWA→DWA) | 2,087 |
| `Content Model Reference.xlsx` | GWA descriptions for embeddings | ~580 |

### Work Activities.xlsx Schema

| Column | Type | Notes |
|--------|------|-------|
| O*NET-SOC Code | Char(10) | Format: `XX-XXXX.XX` |
| Element ID | Varchar(20) | GWA ID (e.g., `4.A.1.a.1`) |
| Scale ID | Varchar(3) | `IM` = Importance, `LV` = Level |
| Data Value | Float | Raw rating |
| N | Integer | Sample size |
| Standard Error | Float | SEM |
| Recommend Suppress | Char(1) | `Y` or `N` |
| Not Relevant | Char(1) | `Y` or `N` (Level only) |

### Rating Scales

**Importance (Scale ID = `IM`):**
- Range: 1-5
- Anchors: 1=Not Important, 2=Somewhat, 3=Important, 4=Very, 5=Extremely
- Normalize: `(value - 1) / 4` → [0, 1]

**Level (Scale ID = `LV`):**
- Range: 0-7
- Level=0 when Importance=1
- Normalize: `value / 7` → [0, 1]

**v0.4 uses Importance only** (consistent anchors, available for both GWA and tasks).

### Suppression Rules

**Always filter:** `Recommend Suppress = 'N'`

Suppression criteria (for reference):
- N < 10
- Variance = 0 AND N < 15
- Relative Standard Error > 0.5

**Not Relevant flag** (Level only): When >75% rated Importance=1. Handle by assigning Level=0 or excluding.

### GWA Matrix Construction

```python
# Filter to Importance scale, non-suppressed
df = work_activities[
    (work_activities['Scale ID'] == 'IM') &
    (work_activities['Recommend Suppress'] == 'N')
]

# Pivot to occupation × GWA matrix
matrix = df.pivot_table(
    index='O*NET-SOC Code',
    columns='Element ID',
    values='Data Value'
)

# Normalize to [0,1]
matrix = (matrix - 1) / 4

# Result: ~923 occupations × 41 GWAs
```

### O*NET-SOC Code Format

Format: `XX-XXXX.XX`
- First 6 chars: Standard SOC code
- Suffix `.00`, `.01`, `.02`: O*NET subdivisions
- 1,016 total occupations (923 with full data)

---

## Implementation Roadmap (v0.4.0)

**Scope:** Core pipeline with GWA-based domain, Recipe X distances, Phase I coherence diagnostics.

**Not in scope:** External validation (mobility/wages), Phase II experiments, Recipe Y embeddings.

### Build Order

1. **Data loading** - Load Work Activities.xlsx, apply filters
2. **Activity domain T_n** - 41 GWAs as discrete activity space
3. **Occupation measures ρ_j** - Normalized importance vectors (923 × 41)
4. **Activity distances d(a,b)** - Recipe X: transpose to activity profiles, PCA, Euclidean
5. **Kernel matrix K** - Row-normalized exponential kernel on activities
6. **Propagation** - A = K @ I for shock profile I
7. **Exposure computation** - E_j = ρ_j @ A
8. **Phase I diagnostics** - Entropy, effective support, face validity

### Module Structure

```
src/task_space/
    __init__.py
    data.py          # O*NET file loading and filtering
    domain.py        # Activity domain + occupation measure construction
    distances.py     # Activity distance computation (Recipe X)
    kernel.py        # Kernel matrix, propagation, and exposure computation
    diagnostics.py   # Phase I coherence checks
```

---

## Paper Placeholders

The paper contains placeholders like `[PLACEHOLDER PH0]`, `[PLACEHOLDER PH1]`, etc.

- These are filled with empirical outputs (tables, statistics, diagnostic results)
- Code should generate outputs that can be pasted into these sections
- Automating this transfer is not a priority; manual paste is fine

---

## File Conventions

```
paper/main.tex       # Source of truth for theory and empirical strategy
paper/references.bib # Bibliography (BibTeX)
src/task_space/      # Implementation modules
data/onet/           # O*NET database files (not in git)
tests/               # Test scripts
outputs/             # Generated figures and tables
```

---

## Existing Utilities

- `tests/test_auth.py` - O*NET V2 API connectivity probe (may be useful for spot-checks)
- `tests/probe_level.py` - Investigation of Importance vs Level score availability

Note: v0.4 uses downloaded database files, not API. These probes remain for reference.

---

## Updating Documentation

When making changes:

1. **Code changes** - Update `__init__.py` version comment if version bumps
2. **README.md** - Keep user-facing; update status, usage instructions
3. **CLAUDE.md** - Keep developer-facing; update conventions, roadmap, lessons learned
4. **Paper placeholders** - Fill with empirical outputs when available

If you discover something that would have helped you work faster, add it to this file.
