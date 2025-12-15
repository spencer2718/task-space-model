"""
AI Occupational Exposure (AIOE) scores from Felten, Raj, Seamans.

Source: https://github.com/AIOE-Data/AIOE
Citation: Felten E, Raj M, Seamans R (2021) Occupational, industry, and
         geographic exposure to artificial intelligence: A novel dataset
         and its potential uses. Strategic Management Journal 42(12):2195-2217.

Updated 2023 for generative AI (Language Modeling AIOE).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# Default path to AIOE data
DEFAULT_AIOE_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "data" / "external" / "aioe" / "aioe_extracted" / "AIOE-main"
)


@dataclass
class AIOEData:
    """
    AIOE scores by occupation.

    Attributes:
        scores: DataFrame with columns [soc_code, occupation_title, aioe, lm_aioe]
        source_file: Path to source Excel file
        n_occupations: Number of occupations with scores
        has_lm_aioe: Whether Language Modeling AIOE is included
    """
    scores: pd.DataFrame
    source_file: str
    n_occupations: int
    has_lm_aioe: bool


def load_aioe(
    path: Optional[Path] = None,
    include_lm: bool = True,
) -> AIOEData:
    """
    Load AIOE scores from Excel files.

    Args:
        path: Directory containing AIOE Excel files.
              Defaults to data/external/aioe/aioe_extracted/AIOE-main/
        include_lm: Whether to include Language Modeling AIOE scores.
                    Default True.

    Returns:
        AIOEData with scores DataFrame containing:
        - soc_code: 6-digit SOC (e.g., "11-1011")
        - occupation_title: string
        - aioe: standard AIOE score (approximately standardized)
        - lm_aioe: Language Modeling AIOE (if include_lm=True)

    Raises:
        FileNotFoundError: If AIOE data files not found

    Example:
        >>> data = load_aioe()
        >>> data.scores.head()
           soc_code            occupation_title      aioe   lm_aioe
        0   11-1011              Chief Executives  1.334246  1.308912
        1   11-1021  General and Operations Mgrs  0.574877  0.677615
    """
    path = Path(path) if path else DEFAULT_AIOE_PATH

    # Load main AIOE file
    main_file = path / "AIOE_DataAppendix.xlsx"
    if not main_file.exists():
        raise FileNotFoundError(
            f"AIOE data not found at {main_file}. "
            "Download from https://github.com/AIOE-Data/AIOE and extract."
        )

    df = pd.read_excel(main_file, sheet_name="Appendix A")

    # Standardize column names
    df = df.rename(columns={
        "SOC Code": "soc_code",
        "Occupation Title": "occupation_title",
        "AIOE": "aioe",
    })

    # Load Language Modeling AIOE if requested
    has_lm = False
    if include_lm:
        lm_file = path / "Language Modeling AIOE and AIIE.xlsx"
        if lm_file.exists():
            lm_df = pd.read_excel(lm_file, sheet_name="LM AIOE")
            lm_df = lm_df.rename(columns={
                "SOC Code": "soc_code",
                "Language Modeling AIOE": "lm_aioe",
            })
            # Merge on SOC code
            df = df.merge(
                lm_df[["soc_code", "lm_aioe"]],
                on="soc_code",
                how="left"
            )
            has_lm = True

    # Ensure consistent types
    df["soc_code"] = df["soc_code"].astype(str).str.strip()
    df["aioe"] = pd.to_numeric(df["aioe"], errors="coerce")
    if has_lm:
        df["lm_aioe"] = pd.to_numeric(df["lm_aioe"], errors="coerce")

    return AIOEData(
        scores=df,
        source_file=str(main_file),
        n_occupations=len(df),
        has_lm_aioe=has_lm,
    )


def get_aioe_by_soc(
    soc_code: str,
    aioe_data: Optional[AIOEData] = None,
    use_lm: bool = False,
) -> Optional[float]:
    """
    Get AIOE score for a specific SOC code.

    Args:
        soc_code: 6-digit SOC code (e.g., "11-1011")
        aioe_data: Pre-loaded AIOEData. If None, loads fresh.
        use_lm: If True, return Language Modeling AIOE instead.

    Returns:
        AIOE score, or None if not found
    """
    if aioe_data is None:
        aioe_data = load_aioe(include_lm=use_lm)

    col = "lm_aioe" if use_lm else "aioe"
    match = aioe_data.scores[aioe_data.scores["soc_code"] == soc_code]

    if len(match) == 0:
        return None
    return float(match[col].iloc[0])
