# Legacy Modules

Deprecated modules retained for historical reference. Do not import in new code.

| Module | Replaced By | Deprecation Version |
|--------|-------------|---------------------|
| data.py | data/onet.py, data/oes.py | v0.6.0 |
| distances.py | similarity/distances.py | v0.6.0 |
| crosswalk.py | data/crosswalk.py | v0.6.0 |
| comparison.py | validation/regression.py | v0.6.2 |
| diagnostics_v061.py | validation/diagnostics.py | v0.6.2 |

Safe to delete after v0.7.0 release.

## Why These Were Deprecated

- **data.py, distances.py, crosswalk.py**: Replaced by modular architecture in `data/` and `similarity/` subdirectories
- **comparison.py**: Phase 2 diagnostic code (10-representation comparison) - one-time research artifact
- **diagnostics_v061.py**: Phase 1 diagnostic code (kernel collapse debugging) - one-time research artifact
