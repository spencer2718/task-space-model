# SPEC.md — Current Experimental Phase

**Phase:** 0.7.2.x (Robustness Extension)
**Status:** Active

---

## Completed Phases

| Phase | Objective | Key Finding | Reference |
|-------|-----------|-------------|-----------|
| 0.7.0.x | Shock integration + demand probe | Demand ρ=0.80; geometry modest | LEDGER.md |
| 0.7.1.x | Multiverse + performance battery | 81/81 win rate; MPR=0.74 | LEDGER.md |

---

## Current Phase: 0.7.2.x

### P1: Institutional Robustness Check

**Objective:** Test whether T Module robustness holds when institutional channel includes licensing variation.

**Motivation:** Current I Module uses Job Zone + Certification only. CPS licensing supplement enables direct interaction test.

**Design:**
1. Extract licensing status from CPS supplement (IPUMS)
2. Construct d_inst variant with licensing component
3. Re-run multiverse grid with licensing-augmented I
4. Compare win rates: current I vs. augmented I

**Success criterion:** T Module maintains ≥80% win rate with augmented I.

---

## Consolidation Criteria for v0.7.3

- [ ] P1 completed (licensing interaction)
- [ ] Paper Section 5.6 updated with licensing results
- [ ] No regression in T Module robustness

---

## Methodology

All experiments must comply with MS1-MS9 in LEDGER.md.

Key standards:
- MS7: Language policy (robust vs. validated)
- MS8: Performance battery required
- MS9: Multiverse gate (≥80% win rate)

See LEDGER.md for full definitions.
