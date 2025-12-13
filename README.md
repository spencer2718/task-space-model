# Task Space Model

A geometric framework for measuring labor market exposure to technological shocks.

**Version 0.4.0**

---

## What This Is

This project develops a measurement framework for studying how technological change affects labor markets. The key idea: occupations are probability distributions over an underlying space of work activities. When automation affects certain activities, the impact propagates through the geometry of that space, affecting occupations proportionally to their overlap with the shock.

This approach allows for:
- Continuous gradients of exposure across occupations (vs binary "exposed/unexposed")
- Spillover effects between related activities
- Empirical validation against worker mobility and wage data

---

## Current Status

**v0.4.0** (Current): Core empirical pipeline implemented.
- Activity domain: 41 Generalized Work Activities (GWAs) from O*NET 30.0
- Occupation measures: 894 occupations as probability distributions over activities
- Activity distances: Recipe X (rating-cooccurrence geometry with PCA)
- Kernel matrix: Row-normalized exponential kernel for shock propagation
- Exposure computation: Occupation-level exposure functionals
- Phase I diagnostics: Measure coherence, distance statistics

**Next**: External validation (worker mobility, wage comovement), Phase II experiments.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download O*NET database (required)
# Get db_30_0_excel.zip from https://www.onetcenter.org/database.html
# Extract to data/onet/db_30_0_excel/

# Run pipeline test
PYTHONPATH=src python tests/test_pipeline.py
```

---

## Repository Structure

```
paper/
    main.tex         # Theoretical framework and empirical strategy
    references.bib   # Bibliography
src/task_space/
    data.py          # O*NET file loading
    domain.py        # Activity domain and occupation measures
    distances.py     # Recipe X activity distances
    kernel.py        # Kernel matrix and exposure computation
    diagnostics.py   # Phase I coherence checks
data/onet/           # O*NET database files (not in git)
tests/               # Test scripts
outputs/             # Generated figures and tables
```

---

## Usage

```python
from task_space import (
    build_activity_domain,
    build_occupation_measures,
    compute_activity_distances,
    build_kernel_matrix,
    compute_occupation_exposure,
    create_shock_profile,
)

# Build the manifold
domain = build_activity_domain()
measures = build_occupation_measures()
distances = compute_activity_distances(measures)

# Set up kernel with median distance as bandwidth
from task_space import distance_percentiles
sigma = distance_percentiles(distances)['p50']
kernel = build_kernel_matrix(distances, sigma=sigma)

# Create a shock profile and compute exposures
shock = create_shock_profile(
    activity_ids=distances.activity_ids,
    target_activities={'4.A.1.a.1': 1.0},  # Target "Getting Information"
)
result = compute_occupation_exposure(measures, kernel, shock)

# result.exposures contains E_j for each occupation
```

---

## The Framework in Brief

1. **Activity Domain**: 41 GWAs form a metric space where distance encodes reallocation friction.

2. **Occupation Measures**: Each occupation is a probability distribution over activities, constructed from O*NET Importance ratings.

3. **Shock Propagation**: Technology shocks spread via exponential kernel: k(d) = exp(-d/sigma).

4. **Exposure Measurement**: E_j = sum_a rho_j(a) * A(a), where A is the propagated shock field.

See `paper/main.tex` Section 3 for formal definitions and Section 4 for empirical strategy.

---

## License

Research use only.
