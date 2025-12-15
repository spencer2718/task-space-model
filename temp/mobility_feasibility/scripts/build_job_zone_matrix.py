"""
Task 1: Build Job Zone Institutional Distance Matrix

Constructs d_inst(i,j) = |zone[i] - zone[j]| + gamma * cert_distance[i,j]

Where cert_distance is based on certification importance differential.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Paths
ONET_DIR = Path("data/onet/db_30_0_excel")
OUTPUT_DIR = Path("temp/mobility_feasibility/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_job_zones():
    """Load O*NET Job Zones data."""
    jz = pd.read_excel(ONET_DIR / "Job Zones.xlsx")
    return jz[["O*NET-SOC Code", "Title", "Job Zone"]].copy()


def load_certification():
    """Load certification importance data."""
    ete = pd.read_excel(ONET_DIR / "Education, Training, and Experience.xlsx")
    cert = ete[ete["Element ID"] == "2.D.4.a"].copy()

    # Use Data Value as certification importance (scale ~1-5)
    cert_clean = cert[["O*NET-SOC Code", "Data Value"]].rename(
        columns={"Data Value": "cert_importance"}
    )
    return cert_clean


def build_institutional_distance_matrix(gamma=1.0):
    """
    Build institutional distance matrix.

    d_inst[i,j] = |zone[i] - zone[j]| + gamma * |cert[i] - cert[j]|

    Args:
        gamma: weight on certification difference (default 1.0)

    Returns:
        dict with matrix and metadata
    """
    # Load data
    jz = load_job_zones()
    cert = load_certification()

    # Merge
    df = jz.merge(cert, on="O*NET-SOC Code", how="left")

    # Fill missing certification with median (conservative choice)
    cert_median = df["cert_importance"].median()
    df["cert_importance"] = df["cert_importance"].fillna(cert_median)

    # Normalize certification to [0, 1] scale (matches zone range of 4)
    cert_min, cert_max = df["cert_importance"].min(), df["cert_importance"].max()
    df["cert_normalized"] = (df["cert_importance"] - cert_min) / (cert_max - cert_min) * 4

    n_occ = len(df)
    occ_codes = df["O*NET-SOC Code"].values
    zones = df["Job Zone"].values
    certs = df["cert_normalized"].values

    # Build distance matrix
    # Zone distance: |zone_i - zone_j| ranges from 0 to 4
    zone_dist = np.abs(zones[:, None] - zones[None, :])

    # Cert distance: |cert_i - cert_j| also ranges ~0 to 4 (after normalization)
    cert_dist = np.abs(certs[:, None] - certs[None, :])

    # Combined distance
    d_inst = zone_dist + gamma * cert_dist

    return {
        "d_inst_matrix": d_inst,
        "zone_vector": zones,
        "cert_vector": df["cert_importance"].values,
        "cert_normalized": certs,
        "occ_codes": occ_codes,
        "occ_titles": df["Title"].values,
        "gamma": gamma,
        "n_occ": n_occ,
    }


def print_diagnostics(result):
    """Print diagnostic information."""
    print("=" * 60)
    print("JOB ZONE INSTITUTIONAL DISTANCE MATRIX - DIAGNOSTICS")
    print("=" * 60)

    print(f"\nTotal occupations: {result['n_occ']}")
    print(f"Gamma (cert weight): {result['gamma']}")

    print("\n--- Job Zone Distribution ---")
    zones = result["zone_vector"]
    for z in range(1, 6):
        count = np.sum(zones == z)
        pct = count / len(zones) * 100
        print(f"  Zone {z}: {count:4d} occupations ({pct:5.1f}%)")

    print("\n--- Certification Importance Stats ---")
    certs = result["cert_vector"]
    print(f"  Min:    {np.min(certs):.2f}")
    print(f"  Max:    {np.max(certs):.2f}")
    print(f"  Mean:   {np.mean(certs):.2f}")
    print(f"  Median: {np.median(certs):.2f}")

    # Missing data
    jz = load_job_zones()
    cert = load_certification()
    n_with_cert = len(cert["O*NET-SOC Code"].unique())
    pct_with_cert = n_with_cert / len(jz) * 100
    print(f"\n--- Coverage ---")
    print(f"  Occupations with cert data: {n_with_cert} / {len(jz)} ({pct_with_cert:.1f}%)")

    print("\n--- Distance Matrix Stats ---")
    d = result["d_inst_matrix"]
    # Upper triangle only (exclude diagonal)
    upper = d[np.triu_indices_from(d, k=1)]
    print(f"  Min distance:    {np.min(upper):.3f}")
    print(f"  Max distance:    {np.max(upper):.3f}")
    print(f"  Mean distance:   {np.mean(upper):.3f}")
    print(f"  Median distance: {np.median(upper):.3f}")

    # Zone-only vs combined
    zone_only = np.abs(zones[:, None] - zones[None, :])
    zone_upper = zone_only[np.triu_indices_from(zone_only, k=1)]
    print(f"\n--- Zone-Only Distance Stats (for comparison) ---")
    print(f"  Mean zone distance: {np.mean(zone_upper):.3f}")

    print("\n" + "=" * 60)


def main():
    print("Building Job Zone institutional distance matrix...")

    # Build with gamma=1.0
    result = build_institutional_distance_matrix(gamma=1.0)

    # Print diagnostics
    print_diagnostics(result)

    # Save
    output_path = OUTPUT_DIR / "job_zone_matrix.npz"
    np.savez(
        output_path,
        d_inst_matrix=result["d_inst_matrix"],
        zone_vector=result["zone_vector"],
        cert_vector=result["cert_vector"],
        cert_normalized=result["cert_normalized"],
        occ_codes=result["occ_codes"],
        occ_titles=result["occ_titles"],
        gamma=result["gamma"],
    )
    print(f"\nSaved to: {output_path}")

    # Also save as CSV for inspection
    df_out = pd.DataFrame({
        "occ_code": result["occ_codes"],
        "title": result["occ_titles"],
        "zone": result["zone_vector"],
        "cert_importance": result["cert_vector"],
        "cert_normalized": result["cert_normalized"],
    })
    csv_path = OUTPUT_DIR / "job_zone_data.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
