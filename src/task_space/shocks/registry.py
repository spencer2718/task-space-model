"""
Registry pattern for extensible shock profiles.

To add a new shock:
1. Define a function that takes (domain, **kwargs) and returns np.ndarray
2. Decorate with @register_shock(name, description, ...)
3. Use in config: shock: "your_name"
"""

from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

# Type: (domain, **kwargs) -> (n_activities,) shock intensity array
ShockProfileFn = Callable[..., np.ndarray]


@dataclass
class RegisteredShock:
    name: str
    fn: ShockProfileFn
    description: str
    required_args: list[str]
    optional_args: dict[str, Any]


SHOCK_REGISTRY: dict[str, RegisteredShock] = {}


def register_shock(
    name: str,
    description: str = "",
    required_args: list[str] = None,
    optional_args: dict[str, Any] = None,
):
    """Decorator to register a shock profile function."""
    def decorator(fn: ShockProfileFn) -> ShockProfileFn:
        SHOCK_REGISTRY[name] = RegisteredShock(
            name=name,
            fn=fn,
            description=description,
            required_args=required_args or [],
            optional_args=optional_args or {},
        )
        return fn
    return decorator


def get_shock(name: str) -> RegisteredShock:
    """Retrieve registered shock by name."""
    if name not in SHOCK_REGISTRY:
        available = list(SHOCK_REGISTRY.keys())
        raise ValueError(f"Unknown shock '{name}'. Available: {available}")
    return SHOCK_REGISTRY[name]


def list_shocks() -> list[str]:
    """List all registered shock profile names."""
    return list(SHOCK_REGISTRY.keys())


def describe_shock(name: str) -> dict:
    """Get description of a registered shock."""
    shock = get_shock(name)
    return {
        'name': shock.name,
        'description': shock.description,
        'required_args': shock.required_args,
        'optional_args': shock.optional_args,
    }


def _reset_registry():
    """
    Clear all registered shocks.

    FOR TESTING ONLY. Do not use in production code.
    """
    SHOCK_REGISTRY.clear()


def _restore_default_shocks():
    """
    Reset registry and reload default shocks.

    FOR TESTING ONLY.
    """
    _reset_registry()
    # Force reload profiles to re-register defaults
    import importlib
    from . import profiles
    importlib.reload(profiles)
