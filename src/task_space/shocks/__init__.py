"""
Shock profiles and propagation for Phase II experiments.

Submodules:
    registry: Registry pattern for extensible shock profiles
    profiles: Built-in shock profiles (uniform, gaussian, v1, v2, rbtc)
    propagation: I_t -> A_t -> E_j pipeline
"""

from .registry import (
    register_shock,
    get_shock,
    list_shocks,
    describe_shock,
    RegisteredShock,
    SHOCK_REGISTRY,
)

from .propagation import (
    propagate_shock,
    compute_exposure_from_shock,
    exposure_stats,
    PropagationResult,
)

# Import profiles to register built-in shocks
from . import profiles

__all__ = [
    # Registry
    'register_shock',
    'get_shock',
    'list_shocks',
    'describe_shock',
    'RegisteredShock',
    'SHOCK_REGISTRY',
    # Propagation
    'propagate_shock',
    'compute_exposure_from_shock',
    'exposure_stats',
    'PropagationResult',
]
