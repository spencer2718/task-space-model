"""
Data loading and caching for task-space model.

Submodules:
    onet: O*NET database loading
    oes: OES wage data loading
    crosswalk: O*NET-SOC to SOC crosswalks
    artifacts: Unified cache for embeddings and distances
"""

from .onet import (
    load_onet_data,
    load_work_activities,
    load_content_model_reference,
    load_dwa_reference,
    load_tasks_to_dwas,
    load_task_ratings,
    get_dwa_titles,
    get_task_ratings,
    get_gwa_ids,
    get_occupation_codes,
)

from .oes import (
    load_oes_year,
    load_oes_panel,
    compute_wage_comovement,
    WageComovement,
)

from .crosswalk import (
    onet_to_soc,
    soc_to_onet_pattern,
    build_onet_oes_crosswalk,
    aggregate_occupation_measures,
    OnetOesCrosswalk,
)

from .artifacts import (
    get_embeddings,
    get_distance_matrix,
    clear_cache,
)

__all__ = [
    # O*NET
    'load_onet_data',
    'load_work_activities',
    'load_content_model_reference',
    'load_dwa_reference',
    'load_tasks_to_dwas',
    'load_task_ratings',
    'get_dwa_titles',
    'get_task_ratings',
    'get_gwa_ids',
    'get_occupation_codes',
    # OES
    'load_oes_year',
    'load_oes_panel',
    'compute_wage_comovement',
    'WageComovement',
    # Crosswalk
    'onet_to_soc',
    'soc_to_onet_pattern',
    'build_onet_oes_crosswalk',
    'aggregate_occupation_measures',
    'OnetOesCrosswalk',
    # Artifacts
    'get_embeddings',
    'get_distance_matrix',
    'clear_cache',
]
