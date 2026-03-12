#!/usr/bin/env python3
"""
D2: RTI-Centroid Correlation (v0.7.12.2)

Recomputes RTI-distance correlation using centroid (cosine embedding)
at Census level, then crosswalked to occ1990dd for correlation with
Dorn's RTI.

Usage:
    python scripts/run_rti_centroid_v07122.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from task_space.mobility.io import load_centroid_census


# Paths (matching original RTI script)
DORN_PATH = Path("data/external/dorn_replication")
DORN_ARCHIVE_PATH = DORN_PATH / "dorn_extracted" / "Autor-Dorn-LowSkillServices-FileArchive.zip Folder"
IPUMS_PATH = Path("data/external/ipums")


def main():
    print("=== D2: RTI-Centroid Correlation (v0.7.12.2) ===")

    # Load centroid matrix at Census level
    d_cent, census_codes = load_centroid_census()
    n = d_cent.shape[0]
    print(f"Centroid matrix: {d_cent.shape}")

    # Compute mean centroid distance for each Census occupation
    # Diagonal is zero, so sum(axis=1) / (n-1) is correct
    mean_distances = d_cent.sum(axis=1) / (n - 1)
    census_exposure = pd.DataFrame({
        "occ2010": census_codes,
        "mean_centroid_distance": mean_distances,
    })
    print(f"Census exposures: {len(census_exposure)}")

    # Load IPUMS crosswalk: Census 2010 -> occ1990
    print("Loading IPUMS crosswalk...")
    ipums_xw = pd.read_excel(IPUMS_PATH / "cps_1992-2002-occ2010-xwalk.xlsx")
    ipums_xw = ipums_xw[["1990 Census code", "OCC2010"]].copy()
    ipums_xw.columns = ["occ1990", "occ2010"]
    ipums_xw["occ1990"] = pd.to_numeric(ipums_xw["occ1990"], errors="coerce")
    ipums_xw["occ2010"] = pd.to_numeric(ipums_xw["occ2010"], errors="coerce")
    ipums_xw = ipums_xw.dropna()
    ipums_xw["occ1990"] = ipums_xw["occ1990"].astype(int)
    ipums_xw["occ2010"] = ipums_xw["occ2010"].astype(int)
    print(f"  {len(ipums_xw)} mappings, {ipums_xw['occ2010'].nunique()} unique occ2010")

    # Load Dorn crosswalk: occ1990 -> occ1990dd
    print("Loading Dorn crosswalk...")
    dorn_xw = pd.read_stata(DORN_ARCHIVE_PATH / "crosswalks" / "occ1990_occ1990dd.dta")
    dorn_xw.columns = ["occ1990", "occ1990dd"]
    dorn_xw["occ1990"] = dorn_xw["occ1990"].astype(int)
    dorn_xw["occ1990dd"] = dorn_xw["occ1990dd"].astype(int)
    print(f"  {len(dorn_xw)} mappings")

    # Crosswalk Census 2010 -> occ1990 -> occ1990dd
    # First merge Census exposure with IPUMS (occ2010 -> occ1990)
    merged = census_exposure.merge(ipums_xw, on="occ2010", how="inner")
    print(f"After Census->occ1990: {len(merged)} rows")

    # Then merge with Dorn (occ1990 -> occ1990dd)
    merged = merged.merge(dorn_xw, on="occ1990", how="inner")
    print(f"After occ1990->occ1990dd: {len(merged)} rows")

    # Average mean_centroid_distance when multiple Census codes map to same occ1990dd
    exposure_dd = merged.groupby("occ1990dd")["mean_centroid_distance"].mean().reset_index()
    exposure_dd.columns = ["occ1990dd", "semantic_exposure"]
    print(f"Unique occ1990dd: {len(exposure_dd)}")

    # Load Dorn RTI
    print("Loading Dorn RTI...")
    rti = pd.read_stata(DORN_PATH / "occ1990dd_task_alm.dta")
    rti["rti"] = rti["task_routine"] - (rti["task_abstract"] + rti["task_manual"]) / 2
    print(f"RTI data: {len(rti)} occ1990dd codes")

    # Merge
    final = exposure_dd.merge(rti, on="occ1990dd", how="inner")
    n_matched = len(final)
    print(f"Matched for correlation: {n_matched}")

    # Compute correlations
    sem_exp = final["semantic_exposure"].values

    corrs_pearson = {}
    pvals = {}
    for col in ["rti", "task_routine", "task_abstract", "task_manual"]:
        r, p = pearsonr(sem_exp, final[col].values)
        corrs_pearson[col if col != "rti" else "rti_composite"] = round(float(r), 6)
        pvals[col if col != "rti" else "rti_composite"] = float(p)
        print(f"  r(centroid, {col}) = {r:.4f} (p={p:.4f})")

    rho_spearman, p_spearman = spearmanr(sem_exp, final["rti"].values)
    print(f"  Spearman ρ(centroid, RTI) = {rho_spearman:.4f} (p={p_spearman:.4f})")
    print(f"RTI-centroid r = {corrs_pearson['rti_composite']:.4f} (Wasserstein was -0.052)")

    # Save
    result = {
        "experiment": "rti_centroid_v07122",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "distance_metric": "cosine_embed (centroid)",
        "n_matched_occ1990dd": n_matched,
        "correlations_pearson": corrs_pearson,
        "pvalues": pvals,
        "spearman_rti": round(float(rho_spearman), 6),
        "comparison_to_wasserstein": {
            "wasserstein_rti_r": -0.052,
            "source": "paper Table (RTI correlation)",
        },
    }

    out_path = "outputs/experiments/rti_centroid_v07122.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
