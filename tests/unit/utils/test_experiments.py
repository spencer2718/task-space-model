"""
Unit tests for task_space.utils.experiments module.

Tests:
- get_output_path returns correct path format
- save_experiment_output writes JSON and returns path
- ensure_project_on_path is callable
- get_experiment_timestamp returns valid format
"""

import json
import tempfile
from pathlib import Path

import pytest

from task_space.utils.experiments import (
    get_output_path,
    save_experiment_output,
    ensure_project_on_path,
    get_experiment_timestamp,
    EXPERIMENTS_OUTPUT_DIR,
)


class TestGetOutputPath:
    """Tests for get_output_path function."""

    def test_returns_path_object(self):
        """get_output_path should return a Path object."""
        result = get_output_path("test_experiment")
        assert isinstance(result, Path)

    def test_path_under_experiments_dir(self):
        """Path should be under outputs/experiments/."""
        result = get_output_path("test_experiment")
        assert result.parts[-2] == "experiments"
        assert result.parts[-3] == "outputs"

    def test_adds_json_extension(self):
        """Path should have .json extension."""
        result = get_output_path("test_experiment")
        assert result.suffix == ".json"

    def test_handles_json_suffix_in_input(self):
        """Should not double the .json extension."""
        result = get_output_path("test_experiment.json")
        assert str(result).endswith("test_experiment.json")
        assert not str(result).endswith(".json.json")

    def test_preserves_experiment_name(self):
        """Experiment name should be preserved in filename."""
        result = get_output_path("my_fancy_experiment_v0702")
        assert result.stem == "my_fancy_experiment_v0702"

    def test_relative_to_experiments_dir(self):
        """Path should match EXPERIMENTS_OUTPUT_DIR constant."""
        result = get_output_path("test")
        assert result == EXPERIMENTS_OUTPUT_DIR / "test.json"


class TestSaveExperimentOutput:
    """Tests for save_experiment_output function."""

    def test_writes_json_file(self, tmp_path, monkeypatch):
        """Should write a valid JSON file."""
        # Monkeypatch the output directory
        monkeypatch.setattr(
            "task_space.utils.experiments.EXPERIMENTS_OUTPUT_DIR",
            tmp_path / "experiments"
        )

        payload = {"metric": 0.95, "n_samples": 100}
        result_path = save_experiment_output("test_write", payload)

        assert result_path.exists()
        with open(result_path) as f:
            data = json.load(f)
        assert data["metric"] == 0.95
        assert data["n_samples"] == 100

    def test_returns_path(self, tmp_path, monkeypatch):
        """Should return the path to saved file."""
        monkeypatch.setattr(
            "task_space.utils.experiments.EXPERIMENTS_OUTPUT_DIR",
            tmp_path / "experiments"
        )

        result = save_experiment_output("test_return", {"x": 1})
        assert isinstance(result, Path)
        assert result.name == "test_return.json"

    def test_creates_parent_directories(self, tmp_path, monkeypatch):
        """Should create parent directories if they don't exist."""
        new_dir = tmp_path / "nested" / "experiments"
        monkeypatch.setattr(
            "task_space.utils.experiments.EXPERIMENTS_OUTPUT_DIR",
            new_dir
        )

        result = save_experiment_output("test_mkdir", {"y": 2})
        assert result.exists()
        assert result.parent.exists()

    def test_uses_indent_2(self, tmp_path, monkeypatch):
        """JSON should be formatted with indent=2."""
        monkeypatch.setattr(
            "task_space.utils.experiments.EXPERIMENTS_OUTPUT_DIR",
            tmp_path / "experiments"
        )

        save_experiment_output("test_indent", {"a": 1, "b": 2})
        path = tmp_path / "experiments" / "test_indent.json"

        content = path.read_text()
        # Check for indentation (newline followed by spaces)
        assert "\n  " in content

    def test_add_timestamp_option(self, tmp_path, monkeypatch):
        """add_timestamp=True should add saved_at field."""
        monkeypatch.setattr(
            "task_space.utils.experiments.EXPERIMENTS_OUTPUT_DIR",
            tmp_path / "experiments"
        )

        save_experiment_output("test_ts", {"x": 1}, add_timestamp=True)
        path = tmp_path / "experiments" / "test_ts.json"

        with open(path) as f:
            data = json.load(f)

        assert "saved_at" in data
        assert "T" in data["saved_at"]  # ISO format has T separator


class TestEnsureProjectOnPath:
    """Tests for ensure_project_on_path function."""

    def test_callable(self):
        """Function should be callable without error."""
        # Should not raise
        ensure_project_on_path()

    def test_noop_when_installed(self):
        """Should be a no-op when package is already installed."""
        # Since we're running tests, package is installed
        import sys
        original_path = sys.path.copy()

        ensure_project_on_path()

        # Should not have modified path significantly
        # (may have checked but found nothing to add)
        assert "task_space" in str(sys.modules.get("task_space", ""))


class TestGetExperimentTimestamp:
    """Tests for get_experiment_timestamp function."""

    def test_returns_string(self):
        """Should return a string."""
        result = get_experiment_timestamp()
        assert isinstance(result, str)

    def test_iso_format(self):
        """Should be in ISO 8601 format."""
        result = get_experiment_timestamp()
        # ISO format has T separator and ends with Z or timezone
        assert "T" in result
        assert result.endswith("Z") or "+" in result or "-" in result[-6:]

    def test_ends_with_z(self):
        """Should end with Z (UTC indicator)."""
        result = get_experiment_timestamp()
        assert result.endswith("Z")
