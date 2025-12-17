# CLAUDE.md — Agent Context

## What This Is

**Task-space oracle:** Modular architecture for analyzing how task-level technological change propagates to occupation-level employment and wages.

**Oracle:** (T, I, S, M) → (Δρ, ΔL, ΔW)

**Current status:** T and I validated. S integrated externally. M preliminary.

**Core insight:** Technology acts on tasks. Occupations are distributions over tasks. Employment and wage outcomes aggregate task-level effects.

**Scope:** Measures structural feasibility (where workers CAN go), not realized reallocation.

---

## Before Any Work

1. **Read LEDGER.md** — Single source of truth for scientific state
2. **Check Hard Constraints** — Inviolable rules (HC1–HC8)
3. **Check Graveyard** — Deprecated approaches that must not be retried

---

## Key Files

| File | Purpose |
|------|---------|
| `LEDGER.md` | Scientific state — **READ FIRST** |
| `paper/main.tex` | Theory, specifications, results |
| `.cache/artifacts/v1/` | Distance matrices, embeddings |
| `outputs/experiments/` | Versioned experiment results |

---

## Critical Implementation Rules

- **Wasserstein is primary** — Not kernel overlap (HC1)
- **RTI requires 16-element composite** — Single element yields R² ≈ 0 (HC2)
- **Do not row-normalize kernels** — Destroys signal (HC5)
- **Institutional barriers are friction, not gates** — γ_inst captures completed transitions only (HC6)
- **Switching costs require external calibration** — 3.84 wage-years/unit Wasserstein
- **Always use `get_embeddings()` from artifacts.py** — Never compute embeddings elsewhere

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

## Architecture (Code)

```
src/task_space/
    data/           # O*NET loading, crosswalks, AIOE
    similarity/     # kernel, overlap, wasserstein, embeddings
    shocks/         # profiles, propagation
    validation/     # regression, diagnostics, reallocation
    mobility/       # CPS validation, institutional distance

tests/
    unit/           # Fast tests
    integration/    # Slow tests (@slow marker)
```

---

## For Details

- **Oracle architecture:** `LEDGER.md` > Oracle Architecture
- **Hard constraints:** `LEDGER.md` > Hard Constraints
- **Module validation:** `paper/main.tex` Section 5
- **Theory:** `paper/main.tex` Sections 3–4
- **Artifact locations:** `LEDGER.md` > Artifact Registry

---

## Version

**0.7.0.1** — Oracle architecture framing, documentation hierarchy
