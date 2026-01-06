"""
Utility modules for task-space experiments and scripts.

This package provides common utilities to reduce code duplication
across experiment scripts.

Submodules:
    experiments: Experiment output saving and path utilities
"""

from .experiments import (
    get_output_path,
    save_experiment_output,
    ensure_project_on_path,
)

__all__ = [
    "get_output_path",
    "save_experiment_output",
    "ensure_project_on_path",
]
