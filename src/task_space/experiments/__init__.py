"""
Module: experiments
Oracle role: Infrastructure for T module validation experiments
Key functions: run_experiment (executes validation), ExperimentConfig (defines specs)

Provides YAML-driven experiment execution for testing different similarity 
measures and shock profiles. Ensures reproducible validation following 
Methodology Standards (MS1-MS10).
"""

from .config import ExperimentConfig
from .runner import run_experiment

__all__ = [
    'ExperimentConfig',
    'run_experiment',
]
