"""
Experiment utilities for task-space model.

This module provides common utilities for experiment scripts:
- Output path generation
- JSON result saving
- Path setup for development environments

These utilities eliminate the need for copy-pasted output blocks in scripts.

Usage:
    from task_space.utils.experiments import (
        get_output_path,
        save_experiment_output,
    )

    # Get output path
    path = get_output_path("my_experiment_v0702")

    # Save results (creates directories, writes JSON with indent=2)
    result_path = save_experiment_output(
        "my_experiment_v0702",
        {"metric": 0.95, "n_samples": 1000},
    )
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Union


# =============================================================================
# Constants
# =============================================================================

# Canonical output directory for experiment results
EXPERIMENTS_OUTPUT_DIR = Path("outputs/experiments")


# =============================================================================
# Path Utilities
# =============================================================================

def get_output_path(experiment_name: str) -> Path:
    """
    Get canonical output path for an experiment.

    Returns path like: outputs/experiments/{experiment_name}.json

    This function does NOT create directories or write files.
    Use save_experiment_output for that.

    Args:
        experiment_name: Name of the experiment (without .json extension)

    Returns:
        Path object for the experiment output file

    Example:
        >>> path = get_output_path("sensitivity_v0702")
        >>> print(path)
        outputs/experiments/sensitivity_v0702.json
    """
    # Strip .json if accidentally provided
    if experiment_name.endswith(".json"):
        experiment_name = experiment_name[:-5]

    return EXPERIMENTS_OUTPUT_DIR / f"{experiment_name}.json"


def save_experiment_output(
    experiment_name: str,
    payload: Mapping[str, Any],
    *,
    add_timestamp: bool = False,
) -> Path:
    """
    Save experiment output to canonical location.

    Implements the standard 5-line JSON output pattern:
    1. Compute path: outputs/experiments/{name}.json
    2. Create parent directories if needed
    3. Write JSON with indent=2

    Args:
        experiment_name: Name of the experiment (without .json extension)
        payload: Dictionary-like object to serialize as JSON
        add_timestamp: If True, adds 'saved_at' field with UTC timestamp

    Returns:
        Path to the saved file

    Raises:
        TypeError: If payload is not JSON-serializable

    Example:
        >>> output = {
        ...     "experiment": "my_test",
        ...     "result": 0.95,
        ...     "n_samples": 1000,
        ... }
        >>> path = save_experiment_output("my_test_v0702", output)
        >>> print(f"Saved to: {path}")
    """
    output_path = get_output_path(experiment_name)

    # Ensure parent directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare payload
    if add_timestamp:
        payload = dict(payload)  # Make a copy to avoid mutating original
        payload["saved_at"] = datetime.now(timezone.utc).isoformat()

    # Write JSON with indent=2 (canonical format)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_serializer)

    return output_path


def _json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for common types.

    Handles:
    - datetime objects -> ISO format strings
    - Path objects -> strings
    - numpy types -> Python native types
    """
    import numpy as np

    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# =============================================================================
# Development Utilities
# =============================================================================

def ensure_project_on_path() -> None:
    """
    Ensure project src directory is on Python path.

    This is a NOOP if the project is installed in editable mode (pip install -e).
    Provided as an optional helper for ad-hoc script execution during development.

    Implementation is minimal and should NOT be relied upon for CI/production.

    Usage (at top of script):
        from task_space.utils.experiments import ensure_project_on_path
        ensure_project_on_path()  # Safe to call; NOOP if already installed

    Note:
        Prefer installing the package in editable mode:
            pip install -e ".[dev]"
    """
    # Try to import task_space to check if already available
    try:
        import task_space
        return  # Already importable, nothing to do
    except ImportError:
        pass

    # Find src directory relative to common script locations
    # This handles: scripts/foo.py, notebooks/foo.ipynb, tests/foo.py
    cwd = Path.cwd()

    # Check common project structures
    possible_src_paths = [
        cwd / "src",
        cwd.parent / "src",
        cwd.parent.parent / "src",
    ]

    for src_path in possible_src_paths:
        if (src_path / "task_space").is_dir():
            src_str = str(src_path)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return

    # Could not find src directory - caller may need to handle this


def get_experiment_timestamp() -> str:
    """
    Get a formatted timestamp for experiment metadata.

    Returns ISO 8601 formatted UTC timestamp with 'Z' suffix.

    Example:
        >>> ts = get_experiment_timestamp()
        >>> print(ts)
        2024-12-22T15:30:45.123456Z
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Constants
    "EXPERIMENTS_OUTPUT_DIR",
    # Path utilities
    "get_output_path",
    "save_experiment_output",
    # Development utilities
    "ensure_project_on_path",
    "get_experiment_timestamp",
]
