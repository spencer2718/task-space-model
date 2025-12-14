"""
O*NET database file loading and filtering.

Loads Work Activities.xlsx and related files from the O*NET 30.0 database.
Applies suppression rules per O*NET documentation.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


# Default path to O*NET database files
DEFAULT_ONET_PATH = Path(__file__).parent.parent.parent.parent / "data" / "onet" / "db_30_0_excel"


def load_onet_data(
    onet_path: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load core O*NET data files.

    Returns dict with keys: work_activities, dwa_reference, tasks_to_dwas, task_ratings
    """
    onet_path = onet_path or DEFAULT_ONET_PATH
    return {
        'work_activities': load_work_activities(onet_path),
        'dwa_reference': load_dwa_reference(onet_path),
        'tasks_to_dwas': load_tasks_to_dwas(onet_path),
        'task_ratings': load_task_ratings(onet_path),
    }


def load_work_activities(
    onet_path: Optional[Path] = None,
    scale_id: str = "IM",
    filter_suppressed: bool = True,
) -> pd.DataFrame:
    """
    Load GWA ratings from Work Activities.xlsx.

    Args:
        onet_path: Path to O*NET database directory. Defaults to data/onet/db_30_0_excel.
        scale_id: Scale to load. "IM" for Importance (default), "LV" for Level.
        filter_suppressed: If True, exclude rows where Recommend Suppress = "Y".

    Returns:
        DataFrame with columns: O*NET-SOC Code, Element ID, Element Name, Data Value, N, Standard Error
    """
    onet_path = onet_path or DEFAULT_ONET_PATH
    filepath = onet_path / "Work Activities.xlsx"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Work Activities.xlsx not found at {filepath}. "
            f"Download O*NET database from https://www.onetcenter.org/database.html"
        )

    df = pd.read_excel(filepath)

    # Filter to requested scale
    df = df[df["Scale ID"] == scale_id].copy()

    # Apply suppression filter
    if filter_suppressed:
        df = df[df["Recommend Suppress"] != "Y"]

    # Select relevant columns
    columns = [
        "O*NET-SOC Code",
        "Element ID",
        "Element Name",
        "Data Value",
        "N",
        "Standard Error",
    ]
    df = df[columns].copy()

    return df


def load_content_model_reference(onet_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Content Model Reference.xlsx for activity descriptions.

    Returns DataFrame with Element ID, Element Name, Description for GWAs.
    GWA Element IDs start with "4.A".
    """
    onet_path = onet_path or DEFAULT_ONET_PATH
    filepath = onet_path / "Content Model Reference.xlsx"

    if not filepath.exists():
        raise FileNotFoundError(f"Content Model Reference.xlsx not found at {filepath}")

    df = pd.read_excel(filepath)

    # Filter to work activities (Element ID starts with 4.A)
    df = df[df["Element ID"].str.startswith("4.A", na=False)].copy()

    return df


def load_dwa_reference(onet_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load DWA Reference.xlsx for DWA hierarchy (GWA -> IWA -> DWA).

    Returns DataFrame with columns: Element ID (GWA), IWA ID, DWA ID, DWA Title.
    """
    onet_path = onet_path or DEFAULT_ONET_PATH
    filepath = onet_path / "DWA Reference.xlsx"

    if not filepath.exists():
        raise FileNotFoundError(f"DWA Reference.xlsx not found at {filepath}")

    return pd.read_excel(filepath)


def load_tasks_to_dwas(onet_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Tasks to DWAs.xlsx for task-DWA mappings.

    Returns DataFrame with columns: O*NET-SOC Code, Task ID, DWA ID, DWA Title.
    """
    onet_path = onet_path or DEFAULT_ONET_PATH
    filepath = onet_path / "Tasks to DWAs.xlsx"

    if not filepath.exists():
        raise FileNotFoundError(f"Tasks to DWAs.xlsx not found at {filepath}")

    return pd.read_excel(filepath)


def load_task_ratings(
    onet_path: Optional[Path] = None,
    scale_id: str = "IM",
    filter_suppressed: bool = True,
) -> pd.DataFrame:
    """
    Load Task Ratings.xlsx for task importance ratings.

    Args:
        onet_path: Path to O*NET database directory.
        scale_id: Scale to load. "IM" for Importance, "RT" for Relevance, "FT" for Frequency.
        filter_suppressed: If True, exclude suppressed rows.

    Returns:
        DataFrame with columns: O*NET-SOC Code, Task ID, Data Value.
    """
    onet_path = onet_path or DEFAULT_ONET_PATH
    filepath = onet_path / "Task Ratings.xlsx"

    if not filepath.exists():
        raise FileNotFoundError(f"Task Ratings.xlsx not found at {filepath}")

    df = pd.read_excel(filepath)

    # Filter to requested scale
    df = df[df["Scale ID"] == scale_id].copy()

    # Apply suppression filter
    if filter_suppressed and "Recommend Suppress" in df.columns:
        df = df[df["Recommend Suppress"] != "Y"]

    columns = ["O*NET-SOC Code", "Task ID", "Data Value"]
    df = df[columns].copy()

    return df


def get_dwa_titles(onet_path: Optional[Path] = None) -> dict[str, str]:
    """
    Get mapping from DWA ID to DWA Title.

    Returns:
        Dict mapping DWA ID -> DWA Title
    """
    dwa_ref = load_dwa_reference(onet_path)
    return dict(zip(dwa_ref['DWA ID'], dwa_ref['DWA Title']))


def get_task_ratings(onet_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Get task importance ratings for DWA derivation.

    Returns:
        DataFrame with O*NET-SOC Code, Task ID, Data Value (importance)
    """
    return load_task_ratings(onet_path, scale_id="IM")


def get_gwa_ids(onet_path: Optional[Path] = None) -> list[str]:
    """
    Get list of all GWA Element IDs from Work Activities file.

    Returns sorted list of unique GWA IDs (e.g., ['4.A.1.a.1', '4.A.1.a.2', ...]).
    """
    df = load_work_activities(onet_path, scale_id="IM", filter_suppressed=False)
    gwa_ids = sorted(df["Element ID"].unique().tolist())
    return gwa_ids


def get_occupation_codes(onet_path: Optional[Path] = None) -> list[str]:
    """
    Get list of all occupation codes with Work Activities data.

    Returns sorted list of O*NET-SOC codes.
    """
    df = load_work_activities(onet_path, scale_id="IM", filter_suppressed=True)
    codes = sorted(df["O*NET-SOC Code"].unique().tolist())
    return codes
