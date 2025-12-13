# Task Space Model

A geometric framework for measuring labor market exposure to technological shocks.

**Version 0.3.7**

---

## What This Is

This project develops a measurement framework for studying how technological change affects labor markets. The key idea: occupations are probability distributions over an underlying space of work activities. When automation affects certain activities, the impact propagates through the geometry of that space, affecting occupations proportionally to their overlap with the shock.

This approach allows for:
- Continuous gradients of exposure across occupations (vs binary "exposed/unexposed")
- Spillover effects between related activities
- Empirical validation against worker mobility and wage data

---

## Current Status

**v0.3.7**: Theoretical framework complete. The paper (`paper/main.tex`) contains:
- Mathematical definitions (Section 3)
- Empirical strategy with O*NET operationalization (Section 4)
- Phase I/II evaluation plan
- Literature review and extensions

**Next (v0.4)**: Empirical implementation of Section 4 pipeline.

---

## Repository Structure

```
paper/
    main.tex         # Theoretical framework and empirical strategy
    references.bib   # Bibliography
src/task_space/      # Implementation (v0.4)
tests/               # API probes and test scripts
outputs/             # Generated figures and tables
```

---

## Quick Start

The theoretical framework is in `paper/main.tex`. Compile with:
```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

For the v0.4 implementation, you'll need O*NET API access:
1. Register at https://services.onetcenter.org/developer/
2. Create `.env` with `ONET_API_KEY=your_key_here`
3. Test connectivity: `PYTHONPATH=src python tests/test_auth.py`

---

## The Framework in Brief

1. **Activity Domain**: Work activities form a metric space where distance encodes reallocation friction.

2. **Occupation Measures**: Each occupation is a probability distribution over activities, constructed from O*NET importance and level ratings.

3. **Shock Propagation**: Technology shocks are activity-level profiles that spread via a distance-based kernel.

4. **Exposure Measurement**: Occupation exposure is the integral of the propagated shock against the occupation's activity distribution.

See `paper/main.tex` Section 3 for formal definitions and Section 4 for empirical operationalization.

---

## License

Research use only.
