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

    Discrete: RTI terciles from Autor-Levy-Murnane (2003) / Autor-Dorn (2013)
    Continuous: CSH (Continuous Semantic Height) - projection onto learned
                RTI direction in embedding space

    RTI = ln(Routine) - ln(Manual) - ln(Abstract)

    CSH is learned by ridge regression to maximize r(CSH, RTI).
    Validated: r(CSH, RTI) = 0.815 (gate: 0.7-0.9)
    """

    def __init__(
        self,
        rti_values: Optional[Dict[int, float]] = None,
        csh_values: Optional[Dict[int, float]] = None,
        csh_alt_values: Optional[Dict[int, float]] = None,
        employment_weights: Optional[Dict[int, float]] = None,
        n_bands: int = 3,
    ):
        """
        Initialize RSH/CSH exposure measure.

        Args:
            rti_values: Dict mapping occ1990dd -> RTI value
            csh_values: Dict mapping occ1990dd -> CSH value
            csh_alt_values: Dict mapping occ1990dd -> CSH_alt value (robustness)
            employment_weights: Dict mapping occ1990dd -> 1980 employment share
            n_bands: Number of RTI bands for discrete measure (default: terciles)
        """
        self._rti = rti_values or {}
        self._csh = csh_values or {}
        self._csh_alt = csh_alt_values or {}
        self._emp_weights = employment_weights or {}
        self._n_bands = n_bands

        # Compute RTI bands (terciles by default)
        if self._rti:
            rti_sorted = sorted(self._rti.values())
            n = len(rti_sorted)
            self._band_thresholds = []
            for i in range(1, n_bands):
                idx = int(i * n / n_bands)
                self._band_thresholds.append(rti_sorted[idx])

    @classmethod
    def from_artifacts(cls, repo_root: Optional['Path'] = None) -> 'RSHExposure':
        """
        Load RSH/CSH exposure from saved artifacts.

        Args:
            repo_root: Path to repository root (auto-detected if None)

        Returns:
            RSHExposure instance with loaded data
        """
        from pathlib import Path

        if repo_root is None:
            repo_root = Path(__file__).parent.parent.parent.parent

        # Load RTI from ALM task scores
        alm_path = repo_root / "data/external/dorn_replication/occ1990dd_task_alm.dta"
        rti_values = {}
        if alm_path.exists():
            import pandas as pd
            df = pd.read_stata(alm_path)
            for _, row in df.iterrows():
                occ = int(row['occ1990dd'])
                routine = row['task_routine']
                manual = row['task_manual']
                abstract = row['task_abstract']
                if pd.notna(routine) and pd.notna(manual) and pd.notna(abstract):
                    if routine > 0 and manual > 0 and abstract > 0:
                        rti = np.log(routine) - np.log(manual) - np.log(abstract)
                        rti_values[occ] = rti

        # Load CSH from saved values
        csh_path = repo_root / "outputs/experiments/csh_values_v0722.csv"
        csh_values = {}
        if csh_path.exists():
            csh_df = pd.read_csv(csh_path)
            for _, row in csh_df.iterrows():
                csh_values[int(row['occ1990dd'])] = row['csh']

        # Load CSH_alt from saved values (robustness variant)
        csh_alt_path = repo_root / "outputs/experiments/csh_alt_values_v0722.csv"
        csh_alt_values = {}
        if csh_alt_path.exists():
            csh_alt_df = pd.read_csv(csh_alt_path)
            for _, row in csh_alt_df.iterrows():
                csh_alt_values[int(row['occ1990dd'])] = row['csh_alt']

        # Load employment weights
        emp_path = repo_root / "data/external/dorn_replication/dorn_extracted/Autor-Dorn-LowSkillServices-FileArchive.zip Folder/dta/occ1990dd_data2012.dta"
        emp_weights = {}
        if emp_path.exists():
            emp_df = pd.read_stata(emp_path)
            for _, row in emp_df.iterrows():
                occ = int(row['occ1990dd'])
                emp = row.get('sh_empl1980', 0)
                if pd.notna(emp):
                    emp_weights[occ] = emp

        instance = cls(
            rti_values=rti_values,
            csh_values=csh_values,
            employment_weights=emp_weights,
        )
        instance._csh_alt = csh_alt_values
        return instance

    def _rti_to_band(self, rti: float) -> int:
        """Convert RTI value to band (0 = lowest routine, n_bands-1 = highest)."""
        band = 0
        for threshold in self._band_thresholds:
            if rti > threshold:
                band += 1
        return band

    def discrete_exposure(self, unit_id: str) -> float:
        """
        Return RTI tercile band for occupation.

        Args:
            unit_id: occ1990dd code as string

        Returns:
            Band index (0 = low routine, 1 = medium, 2 = high routine)
        """
        occ = int(unit_id)
        if occ not in self._rti:
            raise KeyError(f"No RTI for occ1990dd {occ}")
        return float(self._rti_to_band(self._rti[occ]))

    def continuous_exposure(self, unit_id: str) -> float:
        """
        Return CSH (Continuous Semantic Height) for occupation.

        Args:
            unit_id: occ1990dd code as string

        Returns:
            CSH value (projection onto learned RTI direction)
        """
        occ = int(unit_id)
        if occ not in self._csh:
            raise KeyError(f"No CSH for occ1990dd {occ}")
        return self._csh[occ]

    def raw_rti(self, unit_id: str) -> float:
        """Return raw RTI value (not banded)."""
        occ = int(unit_id)
        if occ not in self._rti:
            raise KeyError(f"No RTI for occ1990dd {occ}")
        return self._rti[occ]

    def continuous_exposure_alt(self, unit_id: str) -> float:
        """
        Return CSH_alt (robustness variant) for occupation.

        CSH_alt = cosine similarity to routine centroid.
        Less correlated with RTI (r=0.29) but provides alternative measure.

        Args:
            unit_id: occ1990dd code as string

        Returns:
            CSH_alt value (cosine similarity to routine centroid)
        """
        occ = int(unit_id)
        if occ not in self._csh_alt:
            raise KeyError(f"No CSH_alt for occ1990dd {occ}")
        return self._csh_alt[occ]

    def residualized_continuous(self, unit_id: str) -> float:
        """
        Return CSH residualized against RTI band.

        This captures the continuous variation within each discrete band.
        Computed as CSH - mean(CSH | RTI band).

        Args:
            unit_id: occ1990dd code as string

        Returns:
            Residualized CSH value
        """
        occ = int(unit_id)
        if occ not in self._csh or occ not in self._rti:
            raise KeyError(f"No CSH/RTI for occ1990dd {occ}")

        # Get this occupation's band
        band = self._rti_to_band(self._rti[occ])

        # Compute mean CSH for this band
        band_csh_values = [
            self._csh[o] for o in self._csh
            if o in self._rti and self._rti_to_band(self._rti[o]) == band
        ]

        if not band_csh_values:
            return self._csh[occ]

        band_mean = np.mean(band_csh_values)
        return self._csh[occ] - band_mean

    def metadata(self) -> ExposureMetadata:
        """Return metadata for this exposure measure."""
        n_rti = len(self._rti)
        n_csh = len(self._csh)
        n_common = len(set(self._rti.keys()) & set(self._csh.keys()))

        return ExposureMetadata(
            name="rsh_csh_1980_2005",
            description=(
                "Routine-biased technical change exposure. "
                f"Discrete: RTI terciles ({self._n_bands} bands). "
                f"Continuous: CSH (r=0.815 with RTI)."
            ),
            source="Autor-Levy-Murnane (2003), Autor-Dorn (2013), task-space geometry",
            unit_of_analysis="occ1990dd",
            time_period="1980-2005",
            n_units=n_common,
            coverage=n_common / n_rti if n_rti > 0 else 0,
        )

    def get_unit_ids(self) -> list[str]:
        """Return list of occ1990dd codes with both RTI and CSH."""
        common = set(self._rti.keys()) & set(self._csh.keys())
        return [str(occ) for occ in sorted(common)]

    def aggregate_to_cz(
        self,
        cz_employment: Dict[int, Dict[int, float]],
    ) -> Dict[int, Dict[str, float]]:
        """
        Aggregate exposure to commuting zone level.

        Args:
            cz_employment: Dict mapping cz_id -> {occ1990dd -> employment_share}

        Returns:
            Dict mapping cz_id -> {
                'discrete': employment-weighted mean RTI band,
                'continuous': employment-weighted mean CSH,
                'n_occupations': count of occupations in CZ
            }
        """
        results = {}
        for cz_id, occ_shares in cz_employment.items():
            total_weight = 0
            weighted_discrete = 0
            weighted_continuous = 0
            n_occs = 0

            for occ, share in occ_shares.items():
                if occ in self._rti and occ in self._csh:
                    band = self._rti_to_band(self._rti[occ])
                    weighted_discrete += share * band
                    weighted_continuous += share * self._csh[occ]
                    total_weight += share
                    n_occs += 1

            if total_weight > 0:
                results[cz_id] = {
                    'discrete': weighted_discrete / total_weight,
                    'continuous': weighted_continuous / total_weight,
                    'n_occupations': n_occs,
                }

        return results


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
