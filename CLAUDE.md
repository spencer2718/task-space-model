# CLAUDE.md - Developer Guide for Future Claudes

This document contains working conventions, version control rules, and quality-of-life information for AI assistants working on this project.

---

## Version Control

**Paper and codebase versions must always match.**

- Versions are bumped when entering a new phase of the research program
- Current: v0.3.7 (theoretical framework complete)
- Next: v0.4 (empirical implementation of Section 4)

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

Before implementing major components (e.g., the full v0.4 pipeline), request that the architect agent be consulted to:
- Plan the implementation strategy
- Verify economic assumptions against literature
- Identify potential issues with O*NET data structure
- Design the module architecture

Do not proceed with large implementations without this planning step.

---

## Implementation Roadmap (v0.4)

When implementing Section 4 of the paper, follow this dependency order:

1. **Activity domain T_n** - Fetch DWA or GWA from O*NET
2. **Occupation measures rho_j** - Construct from importance/level ratings
3. **Activity distances d(a,b)** - Recipe X (rating-cooccurrence) or Recipe Y (text embedding)
4. **Kernel matrix K** - Row-normalized exponential kernel
5. **Propagation and exposure** - A = K @ I, then E_j = rho_j^T @ A
6. **Phase I diagnostics** - Coherence checks, metric validation
7. **Phase II experiments** - Retrospective and prospective evaluations

Each step should be testable independently before proceeding.

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
src/task_space/      # Implementation modules (v0.4+)
tests/               # Test scripts and API probes
outputs/             # Generated figures and tables
```

---

## Existing Utilities

These files from earlier exploration may be useful:

- `tests/test_auth.py` - O*NET V2 API connectivity probe
- `tests/probe_level.py` - Investigation of Importance vs Level score availability

These were written for the V2 API (`api-v2.onetcenter.org`) and test Work Activities / Abilities endpoints. DWA endpoints may differ.

---

## Updating Documentation

When making changes:

1. **Code changes** - Update `__init__.py` version comment if version bumps
2. **README.md** - Keep user-facing; update status, usage instructions
3. **CLAUDE.md** - Keep developer-facing; update conventions, roadmap, lessons learned
4. **Paper placeholders** - Fill with empirical outputs when available

If you discover something that would have helped you work faster, add it to this file.
