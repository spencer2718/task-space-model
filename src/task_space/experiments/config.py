"""
Experiment configuration.

Configs are YAML files that specify:
- Data sources
- Similarity measure (registry-based)
- Shock profile (registry-based, optional)
- Validation parameters
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
import yaml


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    # Identity
    name: str
    version: str = "0.6.3"
    description: str = ""

    # Data
    onet_path: Path = field(default_factory=lambda: Path("data/onet/db_30_0_excel"))
    oes_path: Path = field(default_factory=lambda: Path("data/external/oes"))
    output_dir: Path = field(default_factory=lambda: Path("outputs/experiments"))

    # Similarity (registry key + args)
    similarity: str = "wasserstein"  # Default changed from normalized_kernel per HC1
    similarity_args: dict[str, Any] = field(default_factory=dict)

    # Shock (registry key + args, optional for validation-only)
    shock: Optional[str] = None
    shock_args: dict[str, Any] = field(default_factory=dict)

    # Validation
    target: str = "wage_comovement"
    oes_years: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023)
    cluster_by: str = "origin"

    # Controls
    controls: list[str] = field(default_factory=list)

    # Robustness
    run_permutation: bool = True
    n_permutations: int = 1000
    run_cv: bool = True
    n_folds: int = 5
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path) -> 'ExperimentConfig':
        with open(path) as f:
            data = yaml.safe_load(f)

        # Path conversion
        for key in ('onet_path', 'oes_path', 'output_dir'):
            if key in data and data[key]:
                data[key] = Path(data[key])

        # Tuple conversion
        if 'oes_years' in data:
            data['oes_years'] = tuple(data['oes_years'])

        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        data = asdict(self)
        for key in ('onet_path', 'oes_path', 'output_dir'):
            data[key] = str(data[key])
        data['oes_years'] = list(data['oes_years'])

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to dict with paths as strings."""
        data = asdict(self)
        for key in ('onet_path', 'oes_path', 'output_dir'):
            data[key] = str(data[key])
        data['oes_years'] = list(data['oes_years'])
        return data
