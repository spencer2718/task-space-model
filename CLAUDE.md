# CLAUDE.md - Developer Context

Concise context for AI assistants. For theory, see `paper/main.tex`.

---

## Version & Status

| Component | Version |
|-----------|---------|
| Codebase  | v0.6.8.0 |
| Paper     | v0.6.8 |

### Primary Result: Wasserstein >> Kernel

CPS conditional logit (89K transitions). Wasserstein is primary semantic measure.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| α (semantic) | 8.936 (t=206.5) | Workers minimize transformation cost |
| β (institutional) | 0.142 (t=34.6) | Residual non-skill barriers |
| Δ LL vs kernel | +9,576 | Metric structure matters |

ρ(d_wass, d_inst) = 0.25. See `paper/main.tex` Section 5.1 for details.

### Scope Conditions

| Finding | Result | Implication |
|---------|--------|-------------|
| Asymmetric barriers | β_up/β_down = 1.04 | Job Zones = distance, not gates |
| Automation prediction | ΔR²=2.2% (p=0.07) | Marginal over RTI |
| Wage comovement | R²=0.005 (3× Jaccard) | Geometry detectable |

See `paper/main.tex` Sections 5.2–5.4 for details.

### Research Program Framing

The **task-space oracle** F: (T, S, I, M) → outcomes is the organizing metaphor—a hypothetical complete mapping from task representations, shocks, institutions, and mechanisms to labor market outcomes. We test candidate components; we don't estimate the oracle. See `paper/main.tex` Section 3.1.

| Component | Status |
|-----------|--------|
| T (task geometry) | Wasserstein validated |
| I (institutional) | Job Zones validated (symmetric) |
| S (shock profiles) | Generic marginal; modality-specific deferred |
| M (mechanisms) | Future work |

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

# Verify cache integrity
ls -la .cache/artifacts/v1/
ls -la .cache/artifacts/v1/mobility/
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
        artifacts.py       # Embedding/distance cache management
    similarity/            # kernel.py, overlap.py, embeddings.py, distances.py, wasserstein.py (v0.6.7.1)
    shocks/                # registry.py, profiles.py, propagation.py
    validation/            # regression.py, diagnostics.py, permutation.py
    experiments/           # config.py, runner.py
    mobility/              # CPS mobility validation (v0.6.5+)
        institutional.py   # d_inst: job zones + certification; asymmetric d_up/d_down (v0.6.6)
        census_crosswalk.py # Census 2010 ↔ O*NET mapping
        filters.py         # Persistence filter for measurement error
        choice_model.py    # Conditional logit; asymmetric model with LR test (v0.6.6)
    _legacy/               # Deprecated modules (v0.6.6) — see _legacy/README.md

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
| `paper/main.tex` | Theory + specifications (v0.6.8) |
| `outputs/canonical/` | **Paper-ready results (immutable)** |
| `outputs/canonical/mobility_cps.json` | CPS kernel baseline (α=2.994) |
| `outputs/experiments/path_a_wasserstein_comparison_v0672.json` | **Wasserstein primary (α=8.936)** |
| `outputs/canonical/mobility_asymmetric.json` | Asymmetric barriers test (v0.6.6.0) |
| `outputs/canonical/automation_v0653.json` | Automation prediction (v0.6.5.3) |
| `outputs/canonical/wage_comovement.json` | Wage comovement validation |
| `outputs/experiments/` | Versioned experiment results |
| `data/onet/db_30_0_excel/` | O*NET 30.0 database (not in git) |
| `data/external/oes/` | BLS wage data 2019-2024 (not in git) |
| `data/external/aioe/` | Felten-Raj-Seamans AIOE (not in git) |
| `data/external/dingel_neiman/` | Telework feasibility (not in git) |
| `data/processed/mobility/` | CPS analysis source outputs |
| `.cache/artifacts/v1/` | Cached embeddings, distance matrices |
| `.cache/artifacts/v1/mobility/` | Census-level distance matrices, crosswalk |

---

## Model Space Status

| Axis | Current | Status |
|------|---------|--------|
| Representation | MPNet | JobBERT pending (Path B) |
| Geometry | **Wasserstein** | **✓ Validated** |
| Shocks | Generic | Modality-specific deferred |

**Pending paths:** (B) JobBERT vs MPNet; (C) RTI construct validity; (F) Asymmetric with Wasserstein. See `paper/main.tex` Section 6.

---

## Critical Rules

### Do Not Touch
- Kernel bandwidth: σ = 0.223 (NN median for occupations). Activity-level σ = 0.0096.
- Embeddings: Always use `get_embeddings()` from artifacts.py. Never compute elsewhere.
- Row-normalization: Skip for kernels (destroys signal with 2,087 activities)

### Known Gotchas
- BLS blocks automated downloads — download OES data via browser
- O*NET element IDs use dot-parsing (`4.A.3.b.4`), not fixed-width slicing
- RTI must use 16-element AA composite — single O*NET element yields R² ≈ 0
- **Wasserstein is primary** — Use d_wasserstein for mobility. Kernel overlap is comparison baseline only.
- **OT ground metric sensitivity** — Wasserstein amplifies embedding errors. See `paper/main.tex` Remark 3.16.
- **d_sem_census.npz is kernel-based** — Despite "semantic" name, contains kernel overlap distances (not Wasserstein). Also mislabeled as "census" but is O*NET-level (894). Wasserstein distances: `.cache/artifacts/v1/wasserstein/d_wasserstein_onet.npz`.

### Dependencies
- Core: numpy, pandas, scipy, torch, sentence-transformers
- Mobility: statsmodels (ConditionalLogit), pyarrow (parquet I/O)
- OT extension: pot (Python Optimal Transport)

---

## Maintenance Rules

These rules prevent codebase drift. Enforce on every PR / major change.

### New Results
- Every canonical result needs a regression test in `tests/integration/test_canonical_values.py`
- Experiment outputs go to `outputs/experiments/` with version suffix (e.g., `_v0680.json`)
- Update `outputs/canonical/PROVENANCE.md` when canonical values change

### New Modules
- Every `src/task_space/X/` module needs corresponding `tests/unit/X/` tests
- Public functions need docstrings with parameter types
- Add module to Architecture section of this file

### Version Bumps
- `pyproject.toml`, `CLAUDE.md`, `README.md` versions must match
- Run `python scripts/run_validation_battery.py` before tagging

### Before Asking for Changes
- Read `paper/main.tex` for theory (Sections 3-4)
- Check `outputs/canonical/` for current results
- Run `pytest tests/unit -v` to verify environment works

---

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

## Version History

- **v0.6.8.0:** Wasserstein validated as primary. Documentation aligned with paper v0.6.8.
- **v0.6.7.x:** Wasserstein module, full matrix, CPS comparison (Δ LL = +9,576).
- **v0.6.6.0:** Asymmetric barriers test (null: β_up ≈ β_down). Codebase reorganized.
- **v0.6.5.x:** CPS mobility validation. RTI implemented. Automation prediction marginal.
- **v0.6.1:** Kernel bandwidth fix (σ = 0.223 from NN distances).
