# Archived Test Scripts

Historical scripts documenting the research process. Not run by pytest.

These may reference deprecated modules in `src/task_space/_legacy/`.

## Contents

### Phase 1: Kernel Calibration (Dec 13)
| Script | Purpose |
|--------|---------|
| `run_phase1_diagnostics.py` | Diagnosed kernel collapse issue |
| `run_phase1_fix.py` | Implemented σ = 0.223 fix |
| `run_semantic_vs_random.py` | Validated semantic > random post-fix |

### Phase 2: Representation Comparison (Dec 13-14)
| Script | Purpose |
|--------|---------|
| `run_phase2_comparison.py` | 10-representation comparison |
| `run_phase2_robustness.py` | Cross-validation and permutation tests |
| `rerun_permutation_tests.py` | Significance testing |
| `robustness_checks.py` | Additional robustness diagnostics |

### Phase A-D: Early Development (Dec 13)
| Script | Purpose |
|--------|---------|
| `run_phase_a.py` | Initial pipeline testing |
| `run_phase_b.py` | Distance computation verification |
| `run_phase_c.py` | Shock propagation testing |
| `run_phase_d.py` | Full pipeline integration |

### Data Validation (Dec 11-13)
| Script | Purpose |
|--------|---------|
| `test_auth.py` | O*NET API authentication probe |
| `probe_level.py` | O*NET Level score investigation |
| `test_pipeline.py` | Basic pipeline tests |
| `test_validation.py` | Validation infrastructure tests |

### Crosswalk & DWA (Dec 13)
| Script | Purpose |
|--------|---------|
| `run_crosswalk_tests.py` | O*NET ↔ SOC mapping tests |
| `test_crosswalk.py` | Crosswalk unit tests |
| `run_dwa_validation.py` | DWA (Detailed Work Activities) validation |
| `test_dwa.py` | DWA tests |
| `audit_validation.py` | Data quality audit |

## Note

These scripts were used during development to diagnose issues and validate changes. They are preserved for historical documentation but are not maintained.

For current tests, see:
- `tests/unit/` - Fast unit tests
- `tests/integration/` - Slower integration tests
