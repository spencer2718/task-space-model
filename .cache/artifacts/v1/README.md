# Cache Artifacts v1

Cached embeddings and distance matrices. Safe to delete — will be regenerated on first use.

## Structure

```
.cache/artifacts/v1/
├── embeddings/              # Text embeddings (sentence-transformers)
│   └── all-mpnet-base-v2_*.npz
├── distances/               # Distance matrices (cosine)
│   └── cosine_*.npz
└── mobility/                # CPS mobility analysis artifacts
    ├── d_sem_census.npz     # Semantic distances (Census level)
    ├── d_inst_census.npz    # Institutional distances (Census level)
    └── onet_to_census_improved.csv  # Crosswalk mapping
```

## File Details

### embeddings/

NPZ files containing `embeddings` (N×768 float32) and `activity_ids` arrays.

- Hash suffix = MD5 of activity ID list
- Model: `all-mpnet-base-v2` (sentence-transformers)
- Large file (~5.7M) = full 2,087 DWA activities

### distances/

NPZ files containing `distances` (N×N float32) and `activity_ids` arrays.

- Hash suffix = MD5 of activity ID list
- Metric: cosine distance (1 - cosine_similarity)

### mobility/

CPS conditional logit analysis artifacts.

| File | Shape | Contents |
|------|-------|----------|
| `d_sem_census.npz` | ~450×450 | Semantic distance matrix (Census codes) |
| `d_inst_census.npz` | ~450×450 | Institutional distance matrix (Census codes) |
| `onet_to_census_improved.csv` | ~900 rows | O*NET → Census 2010 crosswalk |

## Regeneration

To regenerate all artifacts:

```python
from task_space.data import clear_cache
clear_cache()  # Deletes all cached files
```

Individual modules regenerate their cache on first access.

## Version Notes

- v1: Initial cache format (v0.6.0+)
- Embeddings use `all-mpnet-base-v2` model
- σ = 0.223 (occupation-level), σ = 0.0096 (activity-level)
