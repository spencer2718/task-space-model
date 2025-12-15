# CLAUDE.md - Developer Context

Concise context for AI assistants. For theory, see `paper/main.tex`.

---

## Version & Status

| Component | Version |
|-----------|---------|
| Codebase  | v0.6.5.3 |
| Paper     | v0.6.5 |

### Validation Track 1: Worker Mobility (✓ Validated)

CPS conditional logit confirms semantic-institutional decomposition. Both components significant, not redundant.

| Component | Coefficient | t-stat | Interpretation |
|-----------|-------------|--------|----------------|
| α (semantic) | 2.994 | 98.5 | Workers prefer task-similar destinations |
| β (institutional) | 0.215 | 63.4 | Workers avoid credential barriers |

Correlation between measures r = 0.36. On standardized basis, effects are comparable magnitude.

### Validation Track 2: Automation Prediction (⚠️ Marginal)

Incremental validity test: Does semantic exposure predict 2019-2024 employment changes beyond canonical measures?

| Model | R² | Key Predictor | p-value |
|-------|-----|---------------|---------|
| RTI only (Acemoglu-Autor 16-element) | 9.8% | RTI: β = -0.077 | <0.0001 |
| RTI + Semantic | 12.0% | Semantic: β = +0.037 | 0.075 |
| AIOE + Semantic | 5.9% | Semantic: β = +0.033 | 0.092 |
| Full horse race (RTI + AIOE + Semantic) | 12.2% | Only RTI significant | — |

**Interpretation:** RTI (properly constructed) dominates. Semantic exposure adds ΔR² = 2.2% over RTI (p = 0.07, marginal). Framework succeeds at measuring task similarity but shows only marginal improvement for automation prediction.

### Complementary Validation: Wage Comovement

Semantic geometry detectable (R² = 0.00485 vs 0.00167 for binary). Small absolute R² but useful for model comparison.

---

## Quick Commands

```bash
# Install
pip install -e ".[dev,notebooks]"

# Run tests
pytest tests/unit tests/integration -v

# Skip slow tests
pytest tests/unit tests/integration -v -m "not slow"

# Mobility module tests
pytest tests/unit/mobility -v
pytest tests/integration/mobility -v -m slow
```

---

## Architecture

```
src/task_space/
    domain.py              # Activity domain, occupation measures
    data/                  # Data loading and classifications
        onet.py            # O*NET database loading
        oes.py             # BLS employment/wage data
        crosswalk.py       # O*NET ↔ SOC mapping
        classifications.py # AA task scores, job zones, GWA categories
        aioe.py            # Felten-Raj-Seamans AI exposure (v0.6.5.3)
        telework.py        # Dingel-Neiman telework feasibility (v0.6.5.3)
    similarity/            # kernel.py, overlap.py, embeddings.py
    shocks/                # registry.py, profiles.py, propagation.py
    validation/            # regression.py, diagnostics.py, permutation.py
    experiments/           # config.py, runner.py
    mobility/              # CPS mobility validation (v0.6.5)
        institutional.py   # d_inst: job zones + certification
        census_crosswalk.py # Census 2010 ↔ O*NET mapping
        filters.py         # Persistence filter for measurement error
        choice_model.py    # Conditional logit estimation

tests/
    unit/                  # Fast unit tests
    unit/mobility/         # Mobility module unit tests
    integration/           # Slower tests (some @slow)
    integration/mobility/  # Canonical results verification
    archive/               # Legacy scripts
```

**Key concepts:**
- **Overlap** (Phase I): O = ρ K ρᵀ — pairwise occupation similarity
- **Effective distance** (Phase II): d_eff = d_sem + λ·d_inst — semantic + institutional
- **Epistemic metadata:** Result dataclasses include `assumptions` lists documenting modeling choices

---

## Key Files

| Location | Purpose |
|----------|---------|
| `paper/main.tex` | Theory + specifications (source of truth) |
| `outputs/` | Empirical results (JSON files = canonical) |
| `outputs/experiments/incremental_validity_v0653.json` | Automation prediction results (v0.6.5.3) |
| `data/onet/db_30_0_excel/` | O*NET 30.0 database (not in git) |
| `data/external/oes/` | BLS wage data 2019-2024 (not in git) |
| `data/external/aioe/` | Felten-Raj-Seamans AIOE (not in git) |
| `data/external/dingel_neiman/` | Telework feasibility (not in git) |
| `data/processed/mobility/` | CPS analysis outputs (canonical results) |
| `.cache/artifacts/v1/` | Cached embeddings, distance matrices |
| `.cache/artifacts/v1/mobility/` | Census-level distance matrices, crosswalk |

---

## Current Focus (v0.6.5.3)

**Interpretation of partial negative result.** Semantic geometry validates for mobility but shows only marginal improvement over canonical RTI for automation prediction.

**Key finding:** Framework succeeds at task similarity measurement (CPS mobility, wage comovement) but does not outperform properly-specified Acemoglu-Autor RTI for predicting employment changes. This is methodologically valuable—we correctly identified our RTI proxy was broken and got an honest answer.

**Extensions:**
- Asymmetric d_inst: decompose into d_inst_up vs d_inst_down (upward mobility harder)
- Alternative shock profiles: test whether different capability vectors yield better prediction
- Prospective AI evaluation: apply framework to generative AI profiles

---

## Critical Rules

### Do Not Touch
- Kernel bandwidth: σ = 0.223 (NN median for occupations). Activity-level σ = 0.0096.
- Embeddings: Always use `get_embeddings()` from artifacts.py. Never compute elsewhere.
- Row-normalization: Skip for kernels (destroys signal with 2,087 activities)

### Known Gotchas
- BLS blocks automated downloads — must download OES data manually via browser
- O*NET element IDs use dot-parsing (`4.A.3.b.4`), not fixed-width slicing
- Projected routine scores are ENDOGENOUS — use classifications.py for exogenous GWA categories
- Test registry isolation requires `importlib.reload(profiles)` after `_reset_registry()`
- **Conditional logit assumes IIA** — violations possible for occupation choice; documented in ChoiceModelResult.assumptions
- **RTI must use full 16-element AA composite** — single O*NET element (4.C.3.b.7) yields R² ≈ 0; use `get_aa_task_scores_df()` for proper RTI

### Dependencies
- Core: numpy, pandas, scipy, torch, sentence-transformers
- Mobility: statsmodels (ConditionalLogit), pyarrow (parquet I/O)

### Before Big Implementations
Request the architect agent for:
- Implementation strategy planning
- Economic assumption verification
- Module architecture design

---

## For Details

- **Theory:** `paper/main.tex` Sections 3–4
- **Decomposition:** `paper/main.tex` Section 3.4 (Effective Distance Decomposition)
- **CPS mobility results:** `paper/main.tex` Section 4.4
- **Wage comovement:** `paper/main.tex` Section 4.5
- **Prospective AI evaluation:** `paper/main.tex` Section 4.6
- **Data schemas:** Docstrings in `src/task_space/data/`

---

## Referencing the Paper

Reference definitions by **name**, not number (numbers change):

```python
# Good: Implements the normalized spillover operator (Definition: Spillover Operator)
# Bad: Implements Definition 3.5
```

Key definitions: Task domain, Occupation measures, Effective distance decomposition, Spillover operator, Exposure functionals.

---

## Version History (Brief)

- **v0.6.5.3:** Full Acemoglu-Autor RTI implemented. Incremental validity test shows marginal semantic improvement (ΔR²=2.2%, p=0.07) over properly-specified RTI
- **v0.6.5.1:** CPS mobility validation integrated; semantic-institutional decomposition confirmed
- **v0.6.3.2:** Retrospective battery redesign (1980–2005 canonical settings)
- **v0.6.3.1:** Classification infrastructure, architecture tests
- **v0.6.1:** Kernel fix — σ calibrated to NN distances (was global → collapse)
- **v0.5.0:** Binary overlap validated, but "discrete dominates" was artifact
