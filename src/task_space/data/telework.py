"""
Telework data from Dingel & Neiman (2020).

"How Many Jobs Can be Done at Home?"
https://github.com/jdingel/DingelNeiman-workathome

Provides occupation-level telework feasibility scores (0-1 scale).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# Default path to telework data
DEFAULT_TELEWORK_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "data"
    / "external"
    / "dingel_neiman"
    / "onet_teleworkable_blscodes.csv"
)


@dataclass
class TeleworkData:
    """
    Telework feasibility scores from Dingel & Neiman (2020).

    Attributes:
        scores: DataFrame with columns [soc_code, occupation_title, teleworkable]
        source_file: Path to source data
        n_occupations: Number of occupations
        mean_teleworkable: Average telework feasibility (0-1)
    """

    scores: pd.DataFrame
    source_file: str
    n_occupations: int
    mean_teleworkable: float


def load_telework(path: Optional[Path] = None) -> TeleworkData:
    """
    Load Dingel-Neiman telework feasibility scores.

    Args:
        path: Path to CSV file. Defaults to data/external/dingel_neiman/onet_teleworkable_blscodes.csv

    Returns:
        TeleworkData with occupation-level telework scores (0-1 scale)

    Raises:
        FileNotFoundError: If data file not found
    """
    path = Path(path) if path else DEFAULT_TELEWORK_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Telework data not found at {path}.\n"
            "Download from: https://github.com/jdingel/DingelNeiman-workathome\n"
            "Expected file: onet_teleworkable_blscodes.csv"
        )

    df = pd.read_csv(path)

    # Standardize column names
    df = df.rename(
        columns={
            "OCC_CODE": "soc_code",
            "OES_TITLE": "occupation_title",
        }
    )

    # Ensure teleworkable is numeric (handles any string values)
    df["teleworkable"] = pd.to_numeric(df["teleworkable"], errors="coerce")

    # Drop any rows with missing values
    df = df.dropna(subset=["soc_code", "teleworkable"])

    return TeleworkData(
        scores=df[["soc_code", "occupation_title", "teleworkable"]],
        source_file=str(path),
        n_occupations=len(df),
        mean_teleworkable=float(df["teleworkable"].mean()),
    )


def get_telework_by_soc(soc_code: str) -> Optional[float]:
    """
    Get telework score for a single SOC code.

    Args:
        soc_code: 6-digit SOC code (e.g., "11-1011")

    Returns:
        Telework feasibility score (0-1) or None if not found
    """
    data = load_telework()
    match = data.scores[data.scores["soc_code"] == soc_code]
    if len(match) == 0:
        return None
    return float(match["teleworkable"].iloc[0])
