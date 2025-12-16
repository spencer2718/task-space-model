#!/usr/bin/env python3
"""
Path C: RTI Construct Validity (v0.6.8.1)

Correlates Wasserstein-based semantic exposure with Dorn's pre-computed RTI
at occ1990dd level to establish construct validity against the canonical
RBTC paradigm.

Research Question: Does our semantic geometry capture the same construct
as Autor-Dorn RTI, or does it capture different structure?

- High correlation (r > 0.6): Geometry captures RTI (validation)
- Medium correlation (r ∈ [0.3, 0.6]): Partial overlap (nuanced)
- Low correlation (r < 0.3): Distinct constructs (novel contribution)

Crosswalk Chain:
    O*NET-SOC → Census 2010 → occ1990 → occ1990dd

Data Sources:
- Wasserstein distances: .cache/artifacts/v1/wasserstein/d_wasserstein_onet.npz
- Dorn RTI: data/external/dorn_replication/occ1990dd_task_alm.dta
- IPUMS crosswalk: data/external/ipums/cps_1992-2002-occ2010-xwalk.xlsx
- Dorn occ1990→occ1990dd: crosswalks/occ1990_occ1990dd.dta

Output: outputs/experiments/path_c_rti_construct_validity_v0681.json

Usage:
    PYTHONPATH=src python scripts/experiments/path_c_rti_construct_validity.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# Paths
CACHE_PATH = Path(".cache/artifacts/v1")
DORN_PATH = Path("data/external/dorn_replication")
DORN_ARCHIVE_PATH = DORN_PATH / "dorn_extracted" / "Autor-Dorn-LowSkillServices-FileArchive.zip Folder"
IPUMS_PATH = Path("data/external/ipums")
OUTPUT_PATH = Path("outputs/experiments/path_c_rti_construct_validity_v0681.json")


def load_dorn_rti() -> pd.DataFrame:
    """Load Dorn's pre-computed RTI at occ1990dd level."""
    rti = pd.read_stata(DORN_PATH / "occ1990dd_task_alm.dta")

    # Compute RTI = routine - (abstract + manual) / 2
    # Note: Dorn uses log scale already, so we compute composite
    rti['rti'] = rti['task_routine'] - (rti['task_abstract'] + rti['task_manual']) / 2

    print(f"Loaded RTI data: {len(rti)} occ1990dd codes")
    print(f"  task_abstract: mean={rti['task_abstract'].mean():.3f}, std={rti['task_abstract'].std():.3f}")
    print(f"  task_routine:  mean={rti['task_routine'].mean():.3f}, std={rti['task_routine'].std():.3f}")
    print(f"  task_manual:   mean={rti['task_manual'].mean():.3f}, std={rti['task_manual'].std():.3f}")
    print(f"  rti:           mean={rti['rti'].mean():.3f}, std={rti['rti'].std():.3f}")

    return rti


def load_ipums_crosswalk() -> pd.DataFrame:
    """Load IPUMS OCC2010 ↔ 1990 Census code crosswalk."""
    xw = pd.read_excel(IPUMS_PATH / "cps_1992-2002-occ2010-xwalk.xlsx")

    # Clean up column names and types
    xw = xw[['1990 Census code', 'OCC2010']].copy()
    xw.columns = ['occ1990', 'occ2010']

    # Convert to numeric, handling any non-numeric entries
    xw['occ1990'] = pd.to_numeric(xw['occ1990'], errors='coerce')
    xw['occ2010'] = pd.to_numeric(xw['occ2010'], errors='coerce')
    xw = xw.dropna()

    xw['occ1990'] = xw['occ1990'].astype(int)
    xw['occ2010'] = xw['occ2010'].astype(int)

    print(f"Loaded IPUMS crosswalk: {len(xw)} mappings")
    print(f"  Unique occ1990: {xw['occ1990'].nunique()}")
    print(f"  Unique occ2010: {xw['occ2010'].nunique()}")

    return xw


def load_dorn_occ1990_to_occ1990dd() -> pd.DataFrame:
    """Load Dorn's occ1990 → occ1990dd mapping."""
    xw = pd.read_stata(DORN_ARCHIVE_PATH / "crosswalks" / "occ1990_occ1990dd.dta")
    xw.columns = ['occ1990', 'occ1990dd']
    xw['occ1990'] = xw['occ1990'].astype(int)
    xw['occ1990dd'] = xw['occ1990dd'].astype(int)

    print(f"Loaded Dorn occ1990→occ1990dd: {len(xw)} mappings")
    print(f"  Unique occ1990: {xw['occ1990'].nunique()}")
    print(f"  Unique occ1990dd: {xw['occ1990dd'].nunique()}")

    return xw


def load_onet_to_census() -> pd.DataFrame:
    """Load our O*NET-SOC → Census 2010 crosswalk."""
    xw = pd.read_csv(CACHE_PATH / "mobility" / "onet_to_census_improved.csv")
    xw = xw[xw['matched'] == True].copy()
    xw['census_2010'] = xw['census_2010'].astype(int)

    print(f"Loaded O*NET→Census2010: {len(xw)} mappings")
    print(f"  Unique O*NET codes: {xw['onet_soc'].nunique()}")
    print(f"  Unique Census codes: {xw['census_2010'].nunique()}")

    return xw


def load_wasserstein_distances() -> tuple[np.ndarray, list]:
    """Load Wasserstein distance matrix at O*NET level."""
    data = np.load(CACHE_PATH / "wasserstein" / "d_wasserstein_onet.npz", allow_pickle=True)
    d_wass = data['distance_matrix']
    onet_codes = data['occupation_codes'].tolist()

    print(f"Loaded Wasserstein distances: {d_wass.shape}")
    print(f"  mean={d_wass.mean():.4f}, std={d_wass.std():.4f}")

    return d_wass, onet_codes


def build_onet_to_occ1990dd_crosswalk(
    onet_to_census: pd.DataFrame,
    ipums_xw: pd.DataFrame,
    dorn_xw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build full crosswalk: O*NET-SOC → Census 2010 → occ1990 → occ1990dd

    Returns DataFrame with columns: onet_soc, census2010, occ1990, occ1990dd
    """
    print("\nBuilding crosswalk chain...")

    # Step 1: O*NET → Census 2010 (already have)
    step1 = onet_to_census[['onet_soc', 'census_2010']].copy()
    step1.columns = ['onet_soc', 'occ2010']

    # Step 2: Census 2010 → occ1990 (invert IPUMS crosswalk)
    # Note: Multiple 1990 codes may map to same 2010 code
    # We need to go backward: for each occ2010, find corresponding occ1990
    occ2010_to_occ1990 = ipums_xw.groupby('occ2010')['occ1990'].apply(list).to_dict()

    # Expand step1 to include all possible occ1990 codes
    rows = []
    for _, row in step1.iterrows():
        occ2010 = row['occ2010']
        if occ2010 in occ2010_to_occ1990:
            for occ1990 in occ2010_to_occ1990[occ2010]:
                rows.append({
                    'onet_soc': row['onet_soc'],
                    'occ2010': occ2010,
                    'occ1990': occ1990,
                })

    step2 = pd.DataFrame(rows)
    print(f"  After OCC2010→occ1990: {len(step2)} rows ({step2['onet_soc'].nunique()} O*NET codes)")

    # Step 3: occ1990 → occ1990dd (use Dorn crosswalk)
    step3 = step2.merge(dorn_xw, on='occ1990', how='inner')
    print(f"  After occ1990→occ1990dd: {len(step3)} rows ({step3['onet_soc'].nunique()} O*NET codes)")

    # Many O*NET codes may map to same occ1990dd
    # Keep track of weights (will average later)
    final = step3[['onet_soc', 'occ2010', 'occ1990', 'occ1990dd']].copy()

    return final


def compute_semantic_exposure(
    d_wasserstein: np.ndarray,
    onet_codes: list,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute semantic exposure at occ1990dd level.

    Definition: Mean Wasserstein distance to all other occupations.
    This measures overall position in task space (peripheral = high exposure).

    Aggregation: When multiple O*NET codes map to one occ1990dd,
    take the mean of their mean distances.
    """
    print("\nComputing semantic exposure...")

    # Build O*NET code to index mapping
    onet_to_idx = {code: i for i, code in enumerate(onet_codes)}

    # Compute mean distance for each O*NET occupation (exclude diagonal)
    n = d_wasserstein.shape[0]
    mean_distances = (d_wasserstein.sum(axis=1) - np.diag(d_wasserstein)) / (n - 1)

    # Create O*NET-level exposure dataframe
    onet_exposure = pd.DataFrame({
        'onet_soc': onet_codes,
        'mean_distance': mean_distances,
    })

    # Merge with crosswalk
    merged = crosswalk.merge(onet_exposure, on='onet_soc', how='inner')
    print(f"  Matched {len(merged)} crosswalk rows to O*NET distances")

    # Aggregate to occ1990dd level (mean of means)
    exposure = merged.groupby('occ1990dd').agg({
        'mean_distance': 'mean',
        'onet_soc': 'count',  # Number of O*NET codes contributing
    }).reset_index()
    exposure.columns = ['occ1990dd', 'semantic_exposure', 'n_onet_codes']

    print(f"  Aggregated to {len(exposure)} occ1990dd codes")
    print(f"  Mean O*NET codes per occ1990dd: {exposure['n_onet_codes'].mean():.1f}")

    return exposure


def compute_correlations(
    exposure: pd.DataFrame,
    rti: pd.DataFrame,
) -> dict:
    """Compute correlations between semantic exposure and RTI components."""
    print("\nComputing correlations...")

    # Merge exposure with RTI
    merged = exposure.merge(rti, on='occ1990dd', how='inner')
    print(f"  Matched {len(merged)} occupations for correlation")

    if len(merged) < 30:
        print("  WARNING: Very few matches, correlations may be unstable")

    # Compute correlations
    sem_exp = merged['semantic_exposure']

    corrs = {}
    pvals = {}

    for col in ['rti', 'task_routine', 'task_abstract', 'task_manual']:
        r, p = stats.pearsonr(sem_exp, merged[col])
        corrs[col] = r
        pvals[col] = p
        print(f"  r(semantic_exposure, {col}) = {r:.3f} (p={p:.4f})")

    # Also compute Spearman for robustness
    rho, p_spearman = stats.spearmanr(sem_exp, merged['rti'])
    print(f"  rho_spearman(semantic_exposure, rti) = {rho:.3f} (p={p_spearman:.4f})")

    return {
        'pearson': corrs,
        'pvalues': pvals,
        'spearman_rti': rho,
        'spearman_pvalue': p_spearman,
        'n_matched': len(merged),
    }


def main():
    print("=" * 70)
    print("Path C: RTI Construct Validity Test")
    print("Version: 0.6.8.1")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    rti = load_dorn_rti()
    print()

    ipums_xw = load_ipums_crosswalk()
    print()

    dorn_xw = load_dorn_occ1990_to_occ1990dd()
    print()

    onet_to_census = load_onet_to_census()
    print()

    d_wasserstein, onet_codes = load_wasserstein_distances()
    print()

    # Build crosswalk chain
    crosswalk = build_onet_to_occ1990dd_crosswalk(
        onet_to_census, ipums_xw, dorn_xw
    )

    # Compute coverage statistics
    n_onet_matched = crosswalk['onet_soc'].nunique()
    n_onet_total = len(onet_codes)
    n_occ1990dd_matched = crosswalk['occ1990dd'].nunique()
    n_occ1990dd_total = len(rti)

    coverage = {
        'onet_matched': n_onet_matched,
        'onet_total': n_onet_total,
        'onet_coverage': n_onet_matched / n_onet_total,
        'occ1990dd_matched': n_occ1990dd_matched,
        'occ1990dd_total': n_occ1990dd_total,
        'occ1990dd_coverage': n_occ1990dd_matched / n_occ1990dd_total,
    }

    print()
    print("Crosswalk coverage:")
    print(f"  O*NET: {n_onet_matched}/{n_onet_total} ({coverage['onet_coverage']:.1%})")
    print(f"  occ1990dd: {n_occ1990dd_matched}/{n_occ1990dd_total} ({coverage['occ1990dd_coverage']:.1%})")

    # Compute semantic exposure
    exposure = compute_semantic_exposure(d_wasserstein, onet_codes, crosswalk)

    # Distribution of semantic exposure
    print()
    print("Semantic exposure distribution:")
    print(f"  mean={exposure['semantic_exposure'].mean():.4f}")
    print(f"  std={exposure['semantic_exposure'].std():.4f}")
    print(f"  min={exposure['semantic_exposure'].min():.4f}")
    print(f"  max={exposure['semantic_exposure'].max():.4f}")

    # Compute correlations
    corr_results = compute_correlations(exposure, rti)

    # Interpretation
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    r_rti = corr_results['pearson']['rti']

    if abs(r_rti) > 0.6:
        interpretation = "high"
        verdict = f"Semantic geometry CAPTURES RTI construct (r = {r_rti:.3f})"
    elif abs(r_rti) > 0.3:
        interpretation = "medium"
        verdict = f"Semantic geometry PARTIALLY CAPTURES RTI construct (r = {r_rti:.3f})"
    else:
        interpretation = "low"
        verdict = f"Semantic geometry is DISTINCT from RTI construct (r = {r_rti:.3f})"

    print()
    print(f"Pearson r(semantic_exposure, RTI) = {r_rti:.3f}")
    print(f"Interpretation: {interpretation} correlation")
    print()
    print(f"VERDICT: {verdict}")
    print()

    # Note about sign
    if r_rti > 0:
        sign_note = "Positive correlation: Higher semantic exposure (peripheral task space position) associates with higher RTI (more routine)."
    else:
        sign_note = "Negative correlation: Higher semantic exposure (peripheral task space position) associates with lower RTI (more abstract)."
    print(f"Sign interpretation: {sign_note}")

    # Build output
    output = {
        "version": "0.6.8.1",
        "timestamp": datetime.now().isoformat(),
        "task": "Path C: RTI Construct Validity",
        "crosswalk_coverage": coverage,
        "n_occ1990dd_matched": corr_results['n_matched'],
        "semantic_exposure_statistics": {
            "mean": float(exposure['semantic_exposure'].mean()),
            "std": float(exposure['semantic_exposure'].std()),
            "min": float(exposure['semantic_exposure'].min()),
            "max": float(exposure['semantic_exposure'].max()),
        },
        "correlations": {
            "semantic_exposure_vs_rti": corr_results['pearson']['rti'],
            "semantic_exposure_vs_routine": corr_results['pearson']['task_routine'],
            "semantic_exposure_vs_abstract": corr_results['pearson']['task_abstract'],
            "semantic_exposure_vs_manual": corr_results['pearson']['task_manual'],
        },
        "pvalues": corr_results['pvalues'],
        "spearman_correlation": {
            "rti": corr_results['spearman_rti'],
            "pvalue": corr_results['spearman_pvalue'],
        },
        "interpretation": interpretation,
        "verdict": verdict,
        "sign_interpretation": sign_note,
        "semantic_exposure_definition": "mean_wasserstein_distance",
        "aggregation_method": "mean",
        "data_sources": {
            "dorn_rti": "occ1990dd_task_alm.dta",
            "wasserstein": "d_wasserstein_onet.npz",
            "ipums_crosswalk": "cps_1992-2002-occ2010-xwalk.xlsx",
            "dorn_crosswalk": "occ1990_occ1990dd.dta",
        },
    }

    # Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH}")

    return output


if __name__ == "__main__":
    output = main()
