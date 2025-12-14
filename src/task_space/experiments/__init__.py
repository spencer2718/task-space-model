"""
Experiment infrastructure for task-space model.

Submodules:
    config: Experiment configuration schema
    runner: Generic experiment execution
"""

from .config import ExperimentConfig
from .runner import run_experiment

__all__ = [
    'ExperimentConfig',
    'run_experiment',
]
