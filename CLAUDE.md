# CLAUDE.md — Agent Context

**You are the Engineer.** You implement experiments, write code, and update `LEDGER.md`. The Lead Researcher (Spencer) designs specs and guides version updates. The Writer updates `paper/main.tex`.

---

## Scientific State

See `LEDGER.md` for:
- Oracle architecture and module status
- Hard constraints (HC1–HC8)
- Methodology standards (MS1–MS10)
- Current experiment results and validations
- Artifact locations

**v0.7.3.x reframing:** The contribution has been reattributed from "Wasserstein geometry" to "embedding-informed distance with semantic task substitutability." See Attribution Audit section in LEDGER.md.

---

## Before Any Work

1. **Read LEDGER.md** — Single source of truth for scientific state
2. **Check Hard Constraints** — Inviolable rules (HC1–HC8)
3. **Check Graveyard** — Deprecated approaches that must not be retried

---

## Documentation Architecture

Each document has ONE purpose. No redundancy.

| Document | Purpose | Owner |
|----------|---------|-------|
| `CLAUDE.md` | Agent entry point | Lead Researcher (at v0.x.y) |
| `LEDGER.md` | Scientific state (validations, constraints) | Engineer (during v0.x.y.z) |
| `SPEC.md` | Current sprint planning | Lead Researcher |
| `README.md` | Public-facing overview | Lead Researcher (at v0.x.y) |
| `data/README.md` | Data acquisition | Engineer (when sources change) |
| `paper/main.tex` | Authoritative theory & claims | Writer |

**Principle:** Progressive disclosure. This file is minimal; detail lives in referenced docs.

---

## Key Files

| File | Purpose |
|------|---------|
| `LEDGER.md` | Scientific state — **READ FIRST** |
| `paper/main.tex` | Theory, specifications, results |
| `data/README.md` | External data sources — **read before data work** |
| `.cache/artifacts/v1/` | Distance matrices, embeddings |
| `outputs/experiments/` | Versioned experiment results |

---

## Critical Implementation Rules

- **Wasserstein is primary** — Not kernel overlap (HC1). Primary for theoretical grounding; cosine_embed is acceptable approximation (ρ = 0.95)
- **Embedding ground metric is the mechanism** — Semantic task similarity, not distributional treatment, drives improvement over O*NET methods
- **RTI requires 16-element composite** — Single element yields R² ≈ 0 (HC2)
- **Do not row-normalize kernels** — Destroys signal (HC5)
- **Institutional barriers are friction, not gates** — γ_inst captures completed transitions only (HC6)
- **Switching costs require external calibration** — 3.84 wage-years/unit Wasserstein
- **Always use `get_embeddings()` from artifacts.py** — Never compute embeddings elsewhere
- **Methodology Standards (MS1-MS10)** — See LEDGER.md for full definitions

---

## Contribution Framing (v0.7.4.0)

The core contribution is **semantic task substitutability** as the basis for occupation distance:

1. **Embedding choice drives improvement** (+75-96% over identity ground metric)
2. **Distributional treatment is secondary** (+3% from Wasserstein vs centroid)
3. **Mechanism**: Embeddings capture that "operating forklift" ≈ "driving delivery vehicle"

This reframes from "Wasserstein geometry" to "embedding-informed distance." See Attribution Audit in LEDGER.md.

**Structural stability (v0.7.5.0):** COVID comparison validates that task-distance geometry is structural, not contingent. Aggregate coefficients invariant (Δα < 1%); teleworkable occupations show elevated hiring standards post-COVID (δ₄ < 0), consistent with applicant pool expansion enabling selectivity.

---

## Decision Authority

| Decision Type | Engineer Authority | Action Required |
|---------------|-------------------|-----------------|
| Implementation detail | Full | Proceed (e.g., which library, pandas method) |
| File creation/structure | Full | Proceed, document in commit |
| Sample filter or subset | Partial | Document in experiment JSON; flag if >10% exclusion |
| Metric definition change | **None** | STOP → Return to Lead Researcher with proposal |
| Spec deviation | **None** | STOP → Return per MS10 |
| LEDGER Hard Constraint change | **None** | STOP → Requires Lead Researcher approval |
| New claim or finding | **None** | Report results; Lead Researcher adds to Claim Registry |

**When uncertain:** If you're asking yourself "is this a deviation?", it probably is. Return to Lead Researcher.

---

## After Completing Any v0.x.y.z Increment

1. [ ] Save experiment results to `outputs/experiments/[name]_v[version].json`
2. [ ] Update LEDGER.md → Module Validation Checkpoints (results)
3. [ ] Update LEDGER.md → Artifact Registry (new files)
4. [ ] Update LEDGER.md → Version History (one-line entry)
5. [ ] Update LEDGER.md → Header version number
6. [ ] Verify all artifact paths are correct
7. [ ] Commit with message: `v0.x.y.z: [one-line summary]`

---

## LEDGER.md Update Authority

| Section | Engineer Can Update | Requires Lead Researcher |
|---------|--------------------|-----------------------|
| Header (version, date) | ✓ Increment | — |
| Hard Constraints | — | ✓ |
| Methodology Standards | — | ✓ |
| Violations Log | ✓ Add entries | — |
| Claim Registry | — | ✓ |
| Referee Challenge Table | — | ✓ |
| Module Validation Checkpoints | ✓ Add results | — |
| Graveyard | ✓ Add entries | — |
| Frontier | ✓ Update status | ✓ Add new items |
| Artifact Registry | ✓ Add entries | — |
| Version History | ✓ Add entries | — |

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

## Versioning Convention

Experiments run at **x.y.0.z** increments (e.g., 0.7.0.2, 0.7.0.3). After experiment batch completes, paper updates to **x.y.1** and codebase to **x.y.1.0**.

**During experiments:** Only `LEDGER.md` tracks results. Do not update `README.md`, `CLAUDE.md`, or `paper/main.tex`.

**Upon confirmation:** Paper consolidates validated findings; `README.md` and `CLAUDE.md` update alongside the paper. Not at engineer discretion.

---

## Version

**0.7.5.1** — Codebase polish for external review

---

## SPEC.md Requirements

Every sprint spec must include:

### Required Sections
1. **Objective**: One sentence describing success outcome
2. **Phase Structure**: Version → Deliverable → Gate → Timeline table
3. **Detailed Specifications**: For each deliverable
4. **Stop-and-Return Conditions**: Explicit blockers requiring escalation
5. **Risk Assessment**: What could go wrong

### Each Deliverable Specification Must Include
- **Objective**: What we're testing/building
- **Method**: Concrete implementation steps
- **Data requirements**: What inputs, where to get them
- **Acceptance criteria**: How to verify completion
- **Contingency**: "If X is infeasible, STOP and return with [specific information]"

### Out of Scope Declaration
Every spec should explicitly state what is NOT part of this sprint to prevent scope creep.

### Example Contingency Block
```
**If primary method infeasible:**
1. STOP implementation
2. Document: What's missing, what was attempted
3. Return to Lead Researcher with:
   - Blocker description
   - Proposed alternative (if any)
   - DO NOT implement alternative without approval
```

---

## Future: Agentic Architecture

The current workflow uses copy-paste between Claude Web (Researchers, Writer) and Claude Code (Engineer). A future implementation should follow Anthropic's recommended patterns:

1. **Orchestrator-worker pattern**: Lead Researcher as orchestrator with task-specific workers
2. **Filesystem-mediated handoffs**: Agents write to shared artifacts rather than passing context through conversation
3. **Single-writer principle**: Each document section owned by one agent role
4. **Structured handoff protocol**: Typed handoffs (TASK_COMPLETE, BLOCKED, ESCALATION) with verification checklists

See Anthropic's multi-agent documentation for implementation guidance when ready to build this system.
