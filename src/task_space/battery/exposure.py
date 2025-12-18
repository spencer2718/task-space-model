"""
Exposure measures for retrospective battery tests.

This module defines the ExposureMeasure interface for computing discrete (benchmark)
and continuous (geometry-based) exposure measures for occupations or units.

Tests A/B/C from Appendix A will subclass ExposureMeasure:
- Test A (Computer): Autor-Katz-Krueger computer adoption
- Test B (RSH/CSH): Routine-biased technical change (1980-2005)
- Test C (Robot): Acemoglu-Restrepo robot exposure
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd


class ExposureType(Enum):
    """Type of exposure measure."""
    DISCRETE = "discrete"     # Binary or categorical (benchmark)
    CONTINUOUS = "continuous"  # Our geometry-based measure


@dataclass
class ExposureMetadata:
    """
    Metadata describing an exposure measure.

    Attributes:
        name: Short identifier (e.g., "rsh_1980_2005", "robot_exposure")
        description: Full description of the measure
        source: Data source citation
        unit_of_analysis: What the measure is computed over (occupation, industry, commuting zone)
        time_period: Relevant time period (e.g., "1980-2005")
        n_units: Number of units with valid exposure
        coverage: Fraction of target population with valid exposure
    """
    name: str
    description: str
    source: str
    unit_of_analysis: str
    time_period: Optional[str] = None
    n_units: Optional[int] = None
    coverage: Optional[float] = None


@dataclass
class ExposureResult:
    """
    Container for exposure measure results.

    Attributes:
        discrete: Dict mapping unit_id -> discrete exposure value
        continuous: Dict mapping unit_id -> continuous exposure value
        metadata: ExposureMetadata for this measure
        correlation: Correlation between discrete and continuous (diagnostic)
    """
    discrete: Dict[str, float]
    continuous: Dict[str, float]
    metadata: ExposureMetadata
    correlation: Optional[float] = None


class ExposureMeasure(ABC):
    """
    Abstract base class for exposure measures.

    Subclass this for each retrospective test:
    - ComputerExposure (Test A): Computer adoption shock
    - RSHExposure (Test B): Routine-skill-hypothesis displacement
    - RobotExposure (Test C): Industrial robot exposure

    The key distinction:
    - discrete_exposure(): Benchmark measure from prior literature
    - continuous_exposure(): Our geometry-based measure (Wasserstein distance)

    Example usage:
        >>> exposure = RSHExposure(onet_path, distance_matrix)
        >>> result = exposure.compute_all()
        >>> discrete = result.discrete  # RTI from Autor-Dorn
        >>> continuous = result.continuous  # Our semantic exposure
    """

    @abstractmethod
    def discrete_exposure(self, unit_id: str) -> float:
        """
        Compute discrete (benchmark) exposure for a unit.

        This is the exposure measure from prior literature that we
        compare against. May be binary, categorical, or continuous
        depending on the test.

        Args:
            unit_id: Occupation code (occ1990dd, SOC, Census) or other unit

        Returns:
            Discrete exposure value

        Raises:
            KeyError: If unit_id not found in exposure data
        """
        pass

    @abstractmethod
    def continuous_exposure(self, unit_id: str) -> float:
        """
        Compute continuous (geometry-based) exposure for a unit.

        This is our measure based on Wasserstein distance in task space.
        Captures how exposed the unit is based on its task composition.

        Args:
            unit_id: Occupation code or other unit

        Returns:
            Continuous exposure value (higher = more exposed)

        Raises:
            KeyError: If unit_id not found
        """
        pass

    @abstractmethod
    def metadata(self) -> ExposureMetadata:
        """
        Return metadata describing this exposure measure.

        Returns:
            ExposureMetadata with name, source, unit of analysis, etc.
        """
        pass

    def compute_all(self) -> ExposureResult:
        """
        Compute exposure for all units.

        Returns:
            ExposureResult with discrete and continuous dicts
        """
        meta = self.metadata()
        unit_ids = self.get_unit_ids()

        discrete = {}
        continuous = {}

        for uid in unit_ids:
            try:
                discrete[uid] = self.discrete_exposure(uid)
                continuous[uid] = self.continuous_exposure(uid)
            except KeyError:
                continue

        # Compute correlation as diagnostic
        common_ids = set(discrete.keys()) & set(continuous.keys())
        if len(common_ids) > 2:
            d_vals = np.array([discrete[u] for u in common_ids])
            c_vals = np.array([continuous[u] for u in common_ids])
            corr = np.corrcoef(d_vals, c_vals)[0, 1]
        else:
            corr = None

        return ExposureResult(
            discrete=discrete,
            continuous=continuous,
            metadata=meta,
            correlation=corr,
        )

    @abstractmethod
    def get_unit_ids(self) -> list[str]:
        """
        Return list of all unit IDs for which exposure can be computed.

        Returns:
            List of unit identifiers
        """
        pass

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert exposure to DataFrame for analysis.

        Returns:
            DataFrame with columns: unit_id, discrete_exposure, continuous_exposure
        """
        result = self.compute_all()
        common_ids = set(result.discrete.keys()) & set(result.continuous.keys())

        rows = []
        for uid in common_ids:
            rows.append({
                'unit_id': uid,
                'discrete_exposure': result.discrete[uid],
                'continuous_exposure': result.continuous[uid],
            })

        return pd.DataFrame(rows)


# =============================================================================
# Placeholder subclasses - to be implemented with actual data
# =============================================================================

class ComputerExposure(ExposureMeasure):
    """
    Test A: Computer adoption exposure (1984-1997).

    Discrete: Autor-Katz-Krueger (1998) computer use by occupation
    Continuous: Task-space distance to "computerized" task centroid

    NOT YET IMPLEMENTED - requires CPS computer supplement data.
    """

    def __init__(self):
        raise NotImplementedError(
            "ComputerExposure requires CPS computer supplement data (1984, 1989, 1993, 1997). "
            "See Autor, Katz & Krueger (1998) 'Computing Inequality'."
        )

    def discrete_exposure(self, unit_id: str) -> float:
        pass

    def continuous_exposure(self, unit_id: str) -> float:
        pass

    def metadata(self) -> ExposureMetadata:
        pass

    def get_unit_ids(self) -> list[str]:
        pass


class RSHExposure(ExposureMeasure):
    """
    Test B: Routine-biased technical change exposure (1980-2005).

    Discrete: RTI (Routine Task Intensity) from Autor-Dorn
    Continuous: Task-space distance to "routine-displaced" task centroid

    NOT YET IMPLEMENTED - requires mapping O*NET → occ1990dd.
    """

    def __init__(self):
        raise NotImplementedError(
            "RSHExposure requires O*NET → occ1990dd crosswalk. "
            "Dorn replication files available at data/external/dorn_replication/"
        )

    def discrete_exposure(self, unit_id: str) -> float:
        pass

    def continuous_exposure(self, unit_id: str) -> float:
        pass

    def metadata(self) -> ExposureMetadata:
        pass

    def get_unit_ids(self) -> list[str]:
        pass


class RobotExposure(ExposureMeasure):
    """
    Test C: Industrial robot exposure (1990-2007).

    Discrete: Acemoglu-Restrepo robot exposure (industry × region)
    Continuous: Task-space distance to "automatable manual" task centroid

    NOT YET IMPLEMENTED - requires IFR robot data and industry crosswalk.
    """

    def __init__(self):
        raise NotImplementedError(
            "RobotExposure requires IFR robot adoption data and occupation-industry crosswalk. "
            "See Acemoglu & Restrepo (2020) 'Robots and Jobs'."
        )

    def discrete_exposure(self, unit_id: str) -> float:
        pass

    def continuous_exposure(self, unit_id: str) -> float:
        pass

    def metadata(self) -> ExposureMetadata:
        pass

    def get_unit_ids(self) -> list[str]:
        pass
