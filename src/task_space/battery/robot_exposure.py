"""
Robot exposure classification based on Webb (2020) methodology.

Robots are particularly suited to perform routine manual tasks such as:
- Welding, assembling, handling tools
- Moving, loading, packaging objects
- Machine tending, equipment operation
- Physical manipulation in structured environments

This module classifies DWAs as robot-exposed and computes occupation-level
robot exposure scores for Test C'.
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from task_space.data.onet import load_onet_data
from task_space.data.artifacts import get_embeddings


# Keywords indicating robot-capable tasks (Webb 2020, Acemoglu-Restrepo 2020)
# Focus on physical manipulation in structured environments
ROBOT_KEYWORDS = [
    # Core physical manipulation (high specificity)
    r"\bweld",           # Welding
    r"\bassembl",        # Assembling
    r"\bload\b",         # Loading (not "download")
    r"\bunload",         # Unloading
    r"\bpackag",         # Packaging
    r"\bstack",          # Stacking
    r"\blift",           # Lifting
    r"\bcarry",          # Carrying
    # Manufacturing operations
    r"\bfasten",         # Fastening
    r"\bseal",           # Sealing
    r"\bpaint\b",        # Painting (not "repaint")
    r"\bspray",          # Spraying
    r"\bcut(ting)?\b",   # Cutting
    r"\bdrill",          # Drilling
    r"\bgrind",          # Grinding
    r"\bpolish",         # Polishing
    r"\bmold",           # Molding
    r"\bsolder",         # Soldering
    # Material handling
    r"\bposition\b",     # Positioning materials
    r"\bplace\b",        # Placing items
    r"\bsort\b",         # Sorting
    r"\bpack\b",         # Packing
    r"\bfill\b",         # Filling
    # Machine tending (specific)
    r"\btend\b",         # Tending machines
    r"machine.*(operat|tend)", # Operating machines
    r"operat.*(machine|press|lathe|mill)",
]


class RobotExposure:
    """
    Robot exposure measure based on task-level classification.

    Identifies DWAs that robots can perform, then aggregates to occupation level.
    """

    def __init__(
        self,
        onet_path: Optional[Path] = None,
        keyword_threshold: int = 1,  # Min keyword matches to classify as robot-exposed
    ):
        """
        Initialize robot exposure classifier.

        Args:
            onet_path: Path to O*NET data (uses default if None)
            keyword_threshold: Minimum keyword matches for robot classification
        """
        self.keyword_threshold = keyword_threshold

        # Load O*NET data
        onet_data = load_onet_data(onet_path)
        self.dwa_reference = onet_data["dwa_reference"]
        self.tasks_to_dwas = onet_data["tasks_to_dwas"]

        # Build DWA descriptions
        self._dwa_descriptions = {
            row["DWA ID"]: row["DWA Title"]
            for _, row in self.dwa_reference.iterrows()
        }

        # Classify DWAs
        self._robot_dwas = self._identify_robot_dwas()

        # Precompute occupation -> DWA mappings
        self._occ_dwas = self._build_occ_dwa_mapping()

    def _identify_robot_dwas(self) -> set[str]:
        """
        Identify DWAs that are robot-exposed based on keyword matching.

        Returns:
            Set of DWA IDs classified as robot-exposed
        """
        robot_dwas = set()
        pattern = re.compile("|".join(ROBOT_KEYWORDS), re.IGNORECASE)

        for dwa_id, description in self._dwa_descriptions.items():
            matches = pattern.findall(description)
            if len(matches) >= self.keyword_threshold:
                robot_dwas.add(dwa_id)

        return robot_dwas

    def _build_occ_dwa_mapping(self) -> dict[str, set[str]]:
        """
        Build occupation -> DWA set mapping from tasks_to_dwas.

        Each occupation is linked to the DWAs that describe its tasks.
        """
        occ_dwas = {}

        for _, row in self.tasks_to_dwas.iterrows():
            occ = row["O*NET-SOC Code"]
            dwa_id = row["DWA ID"]

            if occ not in occ_dwas:
                occ_dwas[occ] = set()
            if dwa_id:
                occ_dwas[occ].add(dwa_id)

        return occ_dwas

    @property
    def robot_dwas(self) -> set[str]:
        """Get set of robot-exposed DWA IDs."""
        return self._robot_dwas

    @property
    def n_robot_dwas(self) -> int:
        """Number of robot-exposed DWAs."""
        return len(self._robot_dwas)

    @property
    def robot_dwa_descriptions(self) -> list[tuple[str, str]]:
        """Get (ID, description) for all robot-exposed DWAs."""
        return [
            (dwa_id, self._dwa_descriptions[dwa_id])
            for dwa_id in sorted(self._robot_dwas)
            if dwa_id in self._dwa_descriptions
        ]

    def continuous_exposure(self, occ_soc: str) -> float:
        """
        Compute continuous robot exposure for an occupation.

        Robot exposure = fraction of occupation's DWAs that are robot-exposed

        Args:
            occ_soc: O*NET-SOC occupation code

        Returns:
            Robot exposure score (0-1 scale)
        """
        if occ_soc not in self._occ_dwas:
            return np.nan

        dwas = self._occ_dwas[occ_soc]
        if len(dwas) == 0:
            return 0.0

        robot_count = len(dwas & self._robot_dwas)

        return robot_count / len(dwas)

    def discrete_exposure(self, occ_soc: str, tercile: int = 2) -> int:
        """
        Compute discrete (binary or tercile) robot exposure.

        Args:
            occ_soc: O*NET-SOC occupation code
            tercile: Which tercile is "high exposure" (0, 1, or 2)

        Returns:
            1 if high exposure, 0 otherwise
        """
        score = self.continuous_exposure(occ_soc)
        if np.isnan(score):
            return np.nan

        # Get all scores to determine tercile thresholds
        all_scores = [
            self.continuous_exposure(occ)
            for occ in self._occ_task_weights.keys()
        ]
        all_scores = [s for s in all_scores if not np.isnan(s)]

        # Compute tercile threshold
        threshold = np.percentile(all_scores, 100 * (tercile + 1) / 3)

        return 1 if score >= threshold else 0

    def compute_all_exposures(self) -> pd.DataFrame:
        """
        Compute robot exposure for all occupations.

        Returns:
            DataFrame with columns: onet_soc, robot_exposure, robot_exposed_binary
        """
        results = []

        for occ in self._occ_dwas.keys():
            exposure = self.continuous_exposure(occ)
            results.append({
                "onet_soc": occ,
                "robot_exposure": exposure,
            })

        df = pd.DataFrame(results)

        # Add binary classification (top tercile among non-zero exposures)
        # This is more meaningful since many occupations have zero robot exposure
        nonzero = df["robot_exposure"] > 0
        if nonzero.sum() > 0:
            # Top tercile of non-zero exposures
            threshold = df.loc[nonzero, "robot_exposure"].quantile(2/3)
            df["robot_exposed_binary"] = (df["robot_exposure"] >= threshold).astype(int)
        else:
            df["robot_exposed_binary"] = 0

        return df

    def get_exposure_stats(self) -> dict:
        """Get summary statistics for robot exposure distribution."""
        exposures = self.compute_all_exposures()
        exp = exposures["robot_exposure"].dropna()

        return {
            "n_occupations": len(exp),
            "mean": float(exp.mean()),
            "std": float(exp.std()),
            "min": float(exp.min()),
            "p25": float(exp.quantile(0.25)),
            "median": float(exp.median()),
            "p75": float(exp.quantile(0.75)),
            "max": float(exp.max()),
            "n_high_exposure": int((exposures["robot_exposed_binary"] == 1).sum()),
        }


def load_robot_exposure() -> RobotExposure:
    """Load robot exposure classifier with default settings."""
    return RobotExposure()


if __name__ == "__main__":
    # Quick test
    robot = RobotExposure()

    print(f"Robot-exposed DWAs: {robot.n_robot_dwas}")
    print("\nSample robot DWAs:")
    for dwa_id, desc in robot.robot_dwa_descriptions[:10]:
        print(f"  {dwa_id}: {desc}")

    print("\nExposure statistics:")
    stats = robot.get_exposure_stats()
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
