# CLAUDE.md - Developer Context

Concise context for AI assistants. For theory, see `paper/main.tex`.

---

## Version & Status

| Component | Version |
|-----------|---------|
| Codebase  | v0.6.3.2 |
| Paper     | v0.6.3.2 |

**Phase I:** Complete. Both continuous and discrete structures are informative.

| Measure | t-stat | R² |
|---------|--------|-----|
| Normalized kernel | 7.14 | 0.00485 |
| Binary Jaccard | 8.00 | 0.00167 |

**Phase II:** Retrospective diagnostic battery (1980–2005) + prospective AI evaluation.
- **Primary target:** Test B (Autor-Dorn employment reallocation)
- Three tests: ALM task composition, Autor-Dorn CZ reallocation, Acemoglu-Restrepo robots

---

## Quick Commands

```bash
# Install
pip install -e ".[dev,notebooks]"

# Run tests
pytest tests/unit tests/integration -v

# Skip slow tests
pytest tests/unit tests/integration -v -m "not slow"
```

---

## Architecture

```
src/task_space/
    domain.py              # Activity domain, occupation measures
    data/                  # onet.py, oes.py, crosswalk.py, classifications.py
    similarity/            # kernel.py, overlap.py, embeddings.py
    shocks/                # registry.py, profiles.py, propagation.py
    validation/            # regression.py, diagnostics.py, permutation.py
    experiments/           # config.py, runner.py

tests/
    unit/                  # Fast unit tests
    integration/           # Slower tests (some @slow)
    archive/               # Legacy scripts
```

**Key concepts:**
- **Overlap** (Phase I): O = ρ K ρᵀ — pairwise occupation similarity
- **Exposure** (Phase II): E = ρ K I — occupation exposure to shock profile I
- **Registry pattern:** `@register_shock("name")` to add shocks without modifying runner

---

## Key Files

| Location | Purpose |
|----------|---------|
| `paper/main.tex` | Theory + specifications (source of truth) |
| `outputs/` | Empirical results (JSON files = canonical) |
| `data/onet/db_30_0_excel/` | O*NET 30.0 database (not in git) |
| `data/external/oes/` | BLS wage data 2019-2023 (not in git) |
| `.cache/artifacts/v1/` | Cached embeddings, distance matrices |

---

## Current Focus (v0.6.3.2)

**Retrospective diagnostic battery** — see `paper/main.tex` Section 4.5:

1. **Test A:** Task composition shifts (ALM 1980–2000)
2. **Test B:** Employment reallocation (Autor-Dorn 1980–2005) ← PRIMARY
3. **Test C:** Robot displacement (Acemoglu-Restrepo 1990–2007)

**Goal:** Mechanism discrimination — when does continuous structure improve prediction beyond discrete classification?

**Next steps:**
- Build occ1990dd → O*NET-SOC crosswalk
- Construct routine shock profile I^routine(a)
- Replicate Autor-Dorn Table 5 as baseline
- Run decomposition test (eq. 4.3 in paper)

---

## Critical Rules

### Do Not Touch
- Kernel bandwidth: σ = 0.223 (NN median). Do not use global distance median (0.74 → collapse)
- Embeddings: Always use `get_embeddings()` from artifacts.py. Never compute elsewhere.
- Row-normalization: Skip for kernels (destroys signal with 2,087 activities)

### Known Gotchas
- BLS blocks automated downloads — must download OES data manually via browser
- O*NET element IDs use dot-parsing (`4.A.3.b.4`), not fixed-width slicing
- Projected routine scores are ENDOGENOUS — use classifications.py for exogenous GWA categories
- Test registry isolation requires `importlib.reload(profiles)` after `_reset_registry()`

### Before Big Implementations
Request the architect agent for:
- Implementation strategy planning
- Economic assumption verification
- Module architecture design

---

## For Details

- **Theory:** `paper/main.tex` Sections 3–4
- **Phase I results:** `outputs/phase1/*.json`, `outputs/phase2/*.json`
- **Retrospective battery spec:** `paper/main.tex` Section 4.5
- **Prospective AI evaluation:** `paper/main.tex` Section 4.6
- **Data schemas:** Docstrings in `src/task_space/data/`

---

## Referencing the Paper

Reference definitions by **name**, not number (numbers change):

```python
# Good: Implements the normalized spillover operator (Definition: Spillover Operator)
# Bad: Implements Definition 3.5
```

Key definitions: Task domain, Occupation measures, Shock profile, Spillover operator, Exposure functionals.

---

## Version History (Brief)

- **v0.6.3.2:** Retrospective battery redesign (1980–2005 canonical settings)
- **v0.6.3.1:** Classification infrastructure, architecture tests
- **v0.6.1:** Kernel fix — σ calibrated to NN distances (was global → collapse)
- **v0.5.0:** Binary overlap validated, but "discrete dominates" was artifact
