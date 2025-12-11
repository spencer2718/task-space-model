# Task Space Model

A computational framework for modeling how technological shocks propagate through labor markets via task-level dynamics.

**Phase I Status: PASSED** -- O*NET-based task geometry correctly separates robotics vs software automation shocks.

## Overview

This project implements a geometric model of task-level technological change. The core insight: occupations are not primitive objects but *distributions over an underlying task manifold*. When automation hits a region of task space, the impact diffuses outward according to the manifold geometry, affecting nearby occupations proportionally to their exposure.

### The Model in Three Parts

1. **Task Manifold** (`manifold.py`): The space of all tasks, equipped with a metric that captures "how similar" two tasks are. We construct this empirically from O*NET Work Activities and Abilities data. Each occupation is a probability distribution over this space.

2. **Diffusion Dynamics** (`dynamics.py`): Technology shocks propagate via a heat equation on the manifold. A robotics shock centered on "Handling and Moving Objects" diffuses to nearby manual tasks; a software shock centered on "Analyzing Data" diffuses to nearby cognitive tasks. The dynamics follow:
   ```
   A_{t+1} = A_t + K_d[I_t]
   ```
   where K_d is an exponential diffusion kernel and I_t is the shock field.

3. **Exposure Measurement** (`analysis.py`): Each occupation's displacement exposure is the integral of the automation field against its task distribution: `D_j = <A_t, rho_j>`. High exposure means the occupation sits in a heavily-automated region of task space.

### Why This Matters

Traditional labor economics models (routine/non-routine, manual/cognitive) use discrete categories. This model treats task space as *continuous*, allowing:
- Smooth gradients of automation exposure across occupations
- Spillover effects: automating one task affects nearby tasks
- Empirical identification: the diffusion parameter sigma can be estimated from wage covariance data

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up O*NET API credentials
echo "ONET_API_KEY=your_key_here" > .env

# Run the Phase I validation test
PYTHONPATH=src python tests/test_phase_1.py
```

Expected output: Robotics shocks primarily affect Electricians, Machinists, Truck Drivers; Software shocks primarily affect Software Developers, Accountants.

---

## Developer Guide

### Repository Structure

```
task-space-model/
    src/task_space/
        manifold.py      # Task domain: O*NET API + manifold construction
        dynamics.py      # Diffusion operator and state evolution
        analysis.py      # Occupation exposure computation
    tests/
        test_auth.py     # API connectivity probe
        test_phase_1.py  # Shock propagation sanity check
        probe_level.py   # Level vs Importance investigation
    paper/
        main.tex         # Theoretical foundations (Definitions 1.1-1.6)
    notebooks/           # Visualization scripts
    outputs/             # Generated plots
    .cache/              # Disk cache for API responses (git-ignored)
```

### Key Classes

**`OnetManifold`** -- Fetches Work Activities and Abilities from O*NET V2 API, constructs task vectors.
```python
from task_space.manifold import OnetManifold

m = OnetManifold()
m.load_data()  # Fetches 10 default occupations
# m.task_vectors: (n_occupations, n_features) matrix
# m.task_ids: list of SOC codes
```

**`DynamicsEngine`** -- Builds diffusion kernel, propagates shocks.
```python
from task_space.dynamics import DynamicsEngine

engine = DynamicsEngine(sigma=2.0)  # Diffusion length scale
kernel = engine.build_diffusion_kernel(m.task_vectors)
A_t, C_t = engine.evolve(A_t, C_t, kernel, shock_vector)
```

### API Configuration

1. Register at https://services.onetcenter.org/developer/
2. Generate an API key
3. Create `.env` in project root:
   ```
   ONET_API_KEY=your_key_here
   ```

**Importance vs Level scores:**
- *Importance* (0-100): How frequently a task is performed
- *Level* (0-100): How complex/difficult the task is

Use `include_level=True` for better discrimination (e.g., Software Developer vs Data Entry Clerk both have high Importance for "Working with Computers", but very different Levels). Note: Level enrichment requires ~4000 additional API calls.

### Caching

API responses are cached to `.cache/onet/` on first load. Subsequent runs with the same config load instantly. Use `use_cache=False` to force refresh.

### Theory-to-Code Mapping

| Paper Section | Code | Mathematical Object |
|---------------|------|---------------------|
| Definition 1.1 (Task Domain) | `OnetManifold` | Metric-measure space (T, d, mu) |
| Definition 1.4 (Shock Field) | `DynamicsEngine.create_shock_vector()` | Innovation input I_t |
| Definition 1.5 (Diffusion Operator) | `DynamicsEngine.build_diffusion_kernel()` | Integral operator K_d with kernel exp(-d/sigma) |
| Definition 1.6 (Displacement Dynamics) | `DynamicsEngine.evolve()` | dA/dt = K_d[I_t], monotonicity enforced |
| Section 3.3 (Exposure Functionals) | `Nowcaster.compute_exposures()` | D_j = <A_t, rho_j> |

---

## Remark on Coordinate Charts

> The O*NET database provides one specific *coordinate chart* for the abstract task manifold -- it is a measurement instrument, not the territory itself. The mathematical structures (metric, measure, operators) are defined independently of any particular empirical realization. Alternative charts could be constructed from job postings, time-use surveys, or direct task elicitation. The theorems hold for the abstract space; O*NET merely provides convenient numerical handles.

---

## License

Research use only.
