"""
Institutional distance computation for occupation mobility analysis.

Implements d_inst(i,j) = |zone_i - zone_j| + gamma * |cert_i - cert_j|

Where:
- zone: O*NET Job Zone (1-5 scale, preparation required)
- cert: Certification importance (Element ID 2.D.4.a)

References:
- O*NET Job Zones: https://www.onetonline.org/help/online/zones
- Decomposition: paper/main.tex Definition 3.6 (Effective Distance Decomposition)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


@dataclass
class InstitutionalDistanceResult:
    """
    Result from institutional distance matrix construction.

    Attributes:
        matrix: (n_occ, n_occ) symmetric distance matrix
        occupations: List of O*NET-SOC codes (row/column labels)
        zone_vector: Job zone values for each occupation (1-5 scale)
        cert_vector: Raw certification importance values
        cert_normalized: Certification values normalized to [0, 4] range
        n_occupations: Number of occupations
        n_imputed_cert: Number of occupations with imputed certification
        cert_coverage: Fraction of occupations with direct certification data
        gamma: Weight on certification difference
        assumptions: List of modeling assumptions
    """
    matrix: np.ndarray
    occupations: List[str]
    zone_vector: np.ndarray
    cert_vector: np.ndarray
    cert_normalized: np.ndarray
    n_occupations: int
    n_imputed_cert: int
    cert_coverage: float
    gamma: float
    assumptions: List[str] = field(default_factory=lambda: [
        "Symmetric distance: d(i,j) = d(j,i)",
        "Certification missing → median imputed",
        "Job zone difference has equal weight to normalized certification difference",
        "Linear additivity: d_inst = d_zone + gamma * d_cert",
    ])


def load_job_zones(onet_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load O*NET Job Zones data.

    Args:
        onet_path: Path to O*NET database directory. If None, uses default.

    Returns:
        DataFrame with O*NET-SOC Code, Title, Job Zone columns.
    """
    if onet_path is None:
        onet_path = Path("data/onet/db_30_0_excel")

    jz = pd.read_excel(onet_path / "Job Zones.xlsx")
    return jz[["O*NET-SOC Code", "Title", "Job Zone"]].copy()


def load_certification_importance(onet_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load certification importance ratings from O*NET.

    Element ID 2.D.4.a = "Required Certifications and Licenses"
    Scale: 1-5 (Not Important → Extremely Important)

    Args:
        onet_path: Path to O*NET database directory.

    Returns:
        DataFrame with O*NET-SOC Code and cert_importance columns.
    """
    if onet_path is None:
        onet_path = Path("data/onet/db_30_0_excel")

    ete = pd.read_excel(onet_path / "Education, Training, and Experience.xlsx")
    cert = ete[ete["Element ID"] == "2.D.4.a"].copy()

    cert_clean = cert[["O*NET-SOC Code", "Data Value"]].rename(
        columns={"Data Value": "cert_importance"}
    )
    return cert_clean


def build_institutional_distance_matrix(
    onet_path: Optional[Path] = None,
    gamma: float = 1.0,
) -> InstitutionalDistanceResult:
    """
    Build institutional distance matrix from O*NET Job Zones and certification data.

    Implements: d_inst(i,j) = |zone_i - zone_j| + gamma * |cert_i_norm - cert_j_norm|

    Where cert_norm is certification importance rescaled to [0, 4] to match
    the job zone difference range (max zone diff = 4).

    Args:
        onet_path: Path to O*NET database directory.
        gamma: Weight on certification difference (default 1.0 = equal weight).

    Returns:
        InstitutionalDistanceResult with matrix and metadata.

    Example:
        >>> result = build_institutional_distance_matrix()
        >>> result.matrix.shape
        (923, 923)
        >>> result.assumptions[0]
        'Symmetric distance: d(i,j) = d(j,i)'
    """
    # Load data
    jz = load_job_zones(onet_path)
    cert = load_certification_importance(onet_path)

    # Merge
    df = jz.merge(cert, on="O*NET-SOC Code", how="left")

    # Track imputation
    n_missing_cert = df["cert_importance"].isna().sum()
    n_total = len(df)

    # Fill missing certification with median (conservative choice)
    cert_median = df["cert_importance"].median()
    df["cert_importance"] = df["cert_importance"].fillna(cert_median)

    # Normalize certification to [0, 4] scale to match zone range
    # Zone range: 1-5, so max difference = 4
    # Certification range: 1-5, normalize to 0-4
    cert_min = df["cert_importance"].min()
    cert_max = df["cert_importance"].max()
    if cert_max > cert_min:
        df["cert_normalized"] = (df["cert_importance"] - cert_min) / (cert_max - cert_min) * 4
    else:
        df["cert_normalized"] = 0.0

    # Extract vectors
    n_occ = len(df)
    occ_codes = df["O*NET-SOC Code"].tolist()
    zones = df["Job Zone"].values.astype(float)
    certs_raw = df["cert_importance"].values
    certs_norm = df["cert_normalized"].values

    # Build distance matrix
    # Zone distance: |zone_i - zone_j| ranges from 0 to 4
    zone_dist = np.abs(zones[:, None] - zones[None, :])

    # Cert distance: |cert_i - cert_j| also ranges 0 to 4 (after normalization)
    cert_dist = np.abs(certs_norm[:, None] - certs_norm[None, :])

    # Combined distance
    d_inst = zone_dist + gamma * cert_dist

    return InstitutionalDistanceResult(
        matrix=d_inst,
        occupations=occ_codes,
        zone_vector=zones,
        cert_vector=certs_raw,
        cert_normalized=certs_norm,
        n_occupations=n_occ,
        n_imputed_cert=int(n_missing_cert),
        cert_coverage=1.0 - (n_missing_cert / n_total),
        gamma=gamma,
    )


def compute_institutional_distance(
    occ_i: str,
    occ_j: str,
    result: InstitutionalDistanceResult,
) -> float:
    """
    Look up institutional distance between two occupations.

    Args:
        occ_i: O*NET-SOC code for origin occupation
        occ_j: O*NET-SOC code for destination occupation
        result: Pre-computed InstitutionalDistanceResult

    Returns:
        Institutional distance d_inst(i, j)

    Raises:
        ValueError: If occupation code not found
    """
    try:
        idx_i = result.occupations.index(occ_i)
        idx_j = result.occupations.index(occ_j)
    except ValueError as e:
        raise ValueError(f"Occupation code not found: {e}")

    return result.matrix[idx_i, idx_j]


def get_zone_difference(
    occ_i: str,
    occ_j: str,
    result: InstitutionalDistanceResult,
) -> int:
    """
    Get job zone difference between two occupations.

    Useful for analyzing upward vs downward mobility.
    Positive = moving up (higher zone), Negative = moving down.

    Args:
        occ_i: Origin O*NET-SOC code
        occ_j: Destination O*NET-SOC code
        result: Pre-computed result

    Returns:
        zone_j - zone_i (positive = upward mobility)
    """
    idx_i = result.occupations.index(occ_i)
    idx_j = result.occupations.index(occ_j)
    return int(result.zone_vector[idx_j] - result.zone_vector[idx_i])
