"""
Compute CSH_alt: Alternative CSH using distance to routine centroid.

Robustness check for v0.7.2.2 Task 6.

Instead of learning a direction via ridge regression, we:
1. Define a "routine centroid" as the weighted average of high-RTI occupation embeddings
2. Compute CSH_alt = -distance(occupation, routine_centroid)
   (Negative so that higher = more routine, matching CSH sign)

This provides a direction-free alternative that doesn't depend on the ridge regression.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine

# Paths
REPO_ROOT = Path(__file__).parent.parent
CENTROIDS_PATH = REPO_ROOT / ".cache/artifacts/v1/embeddings/occ1990dd_centroids_mpnet.npz"
ALM_PATH = REPO_ROOT / "data/external/dorn_replication/occ1990dd_task_alm.dta"
EMP_PATH = REPO_ROOT / "data/external/dorn_replication/dorn_extracted/Autor-Dorn-LowSkillServices-FileArchive.zip Folder/dta/occ1990dd_data2012.dta"
CSH_PATH = REPO_ROOT / "outputs/experiments/csh_values_v0722.csv"
OUTPUT_PATH = REPO_ROOT / "outputs/experiments/csh_alt_values_v0722.csv"


def load_data():
    """Load centroids, RTI, and employment data."""
    # Centroids
    data = np.load(CENTROIDS_PATH)
    occ_codes = data['occ_codes']
    centroids = data['centroids']
    centroids_dict = dict(zip(occ_codes, centroids))

    # RTI
    df = pd.read_stata(ALM_PATH)
    rti_dict = {}
    for _, row in df.iterrows():
        occ = int(row['occ1990dd'])
        routine = row['task_routine']
        manual = row['task_manual']
        abstract = row['task_abstract']
        if pd.notna(routine) and pd.notna(manual) and pd.notna(abstract):
            if routine > 0 and manual > 0 and abstract > 0:
                rti = np.log(routine) - np.log(manual) - np.log(abstract)
                rti_dict[occ] = rti

    # Employment weights
    emp_dict = {}
    if EMP_PATH.exists():
        emp_df = pd.read_stata(EMP_PATH)
        for _, row in emp_df.iterrows():
            occ = int(row['occ1990dd'])
            emp = row.get('sh_empl1980', 0)
            if pd.notna(emp):
                emp_dict[occ] = emp

    return centroids_dict, rti_dict, emp_dict


def compute_routine_centroid(centroids_dict, rti_dict, emp_dict, top_fraction=0.33):
    """
    Compute the routine task centroid.

    Uses employment-weighted average of top-RTI occupations.

    Args:
        centroids_dict: occ1990dd -> embedding
        rti_dict: occ1990dd -> RTI value
        emp_dict: occ1990dd -> employment share
        top_fraction: Fraction of occupations to use (by RTI rank)

    Returns:
        routine_centroid: 768-dim vector representing "routine" tasks
    """
    # Get occupations with all data
    common = set(centroids_dict.keys()) & set(rti_dict.keys())
    print(f"Found {len(common)} occupations with both centroids and RTI")

    # Sort by RTI (descending)
    sorted_by_rti = sorted(common, key=lambda x: rti_dict[x], reverse=True)

    # Take top fraction
    n_top = int(len(sorted_by_rti) * top_fraction)
    top_routine_occs = sorted_by_rti[:n_top]
    print(f"Using top {n_top} occupations by RTI ({top_fraction*100:.0f}%)")
    print(f"  RTI range: [{rti_dict[top_routine_occs[-1]]:.3f}, {rti_dict[top_routine_occs[0]]:.3f}]")

    # Compute weighted centroid
    total_weight = 0
    weighted_sum = np.zeros(768)

    for occ in top_routine_occs:
        weight = emp_dict.get(occ, 1.0)  # Default to 1 if no employment data
        weighted_sum += weight * centroids_dict[occ]
        total_weight += weight

    routine_centroid = weighted_sum / total_weight
    routine_centroid = routine_centroid / np.linalg.norm(routine_centroid)  # Normalize

    return routine_centroid, top_routine_occs


def compute_csh_alt(centroids_dict, routine_centroid, method='cosine'):
    """
    Compute CSH_alt as distance to routine centroid.

    Args:
        centroids_dict: occ1990dd -> embedding
        routine_centroid: The routine task centroid
        method: 'cosine' or 'euclidean'

    Returns:
        csh_alt_dict: occ1990dd -> CSH_alt value
    """
    csh_alt_dict = {}

    for occ, centroid in centroids_dict.items():
        if method == 'cosine':
            # Cosine similarity (higher = closer to routine)
            sim = 1 - cosine(centroid, routine_centroid)
            csh_alt_dict[occ] = sim
        elif method == 'euclidean':
            # Negative Euclidean distance (higher = closer to routine)
            dist = np.linalg.norm(centroid - routine_centroid)
            csh_alt_dict[occ] = -dist

    return csh_alt_dict


def main():
    print("=" * 60)
    print("CSH_alt Computation (v0.7.2.2 Task 6 - Robustness)")
    print("=" * 60)
    print()

    # Load data
    print("Loading data...")
    centroids_dict, rti_dict, emp_dict = load_data()

    # Compute routine centroid
    print("\nComputing routine centroid...")
    routine_centroid, top_routine_occs = compute_routine_centroid(
        centroids_dict, rti_dict, emp_dict, top_fraction=0.33
    )

    # Compute CSH_alt (cosine similarity)
    print("\nComputing CSH_alt (cosine similarity to routine centroid)...")
    csh_alt_dict = compute_csh_alt(centroids_dict, routine_centroid, method='cosine')

    # Statistics
    csh_alt_vals = list(csh_alt_dict.values())
    print(f"CSH_alt stats: min={min(csh_alt_vals):.4f}, max={max(csh_alt_vals):.4f}, "
          f"mean={np.mean(csh_alt_vals):.4f}, std={np.std(csh_alt_vals):.4f}")

    # Validate correlation with RTI
    common_occs = sorted(set(csh_alt_dict.keys()) & set(rti_dict.keys()))
    csh_alt_vals_common = [csh_alt_dict[o] for o in common_occs]
    rti_vals_common = [rti_dict[o] for o in common_occs]

    r, p = stats.pearsonr(csh_alt_vals_common, rti_vals_common)
    rho, p_rho = stats.spearmanr(csh_alt_vals_common, rti_vals_common)

    print(f"\nCorrelation with RTI:")
    print(f"  Pearson r(CSH_alt, RTI) = {r:.4f} (p = {p:.2e})")
    print(f"  Spearman rho = {rho:.4f} (p = {p_rho:.2e})")

    # Compare with original CSH
    if CSH_PATH.exists():
        csh_df = pd.read_csv(CSH_PATH)
        csh_dict = {int(row['occ1990dd']): row['csh'] for _, row in csh_df.iterrows()}

        common_csh = sorted(set(csh_alt_dict.keys()) & set(csh_dict.keys()))
        csh_vals = [csh_dict[o] for o in common_csh]
        csh_alt_vals_csh = [csh_alt_dict[o] for o in common_csh]

        r_csh, _ = stats.pearsonr(csh_vals, csh_alt_vals_csh)
        print(f"\nCorrelation with original CSH:")
        print(f"  r(CSH, CSH_alt) = {r_csh:.4f}")

    # Save results
    print(f"\nSaving to {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([
        {'occ1990dd': occ, 'csh_alt': csh_alt, 'rti': rti_dict.get(occ)}
        for occ, csh_alt in csh_alt_dict.items()
    ])
    results_df.to_csv(OUTPUT_PATH, index=False)

    # Print top/bottom occupations
    print("\n" + "=" * 60)
    print("TOP 10 OCCUPATIONS BY CSH_alt (most routine-like)")
    print("-" * 60)
    sorted_by_alt = sorted(csh_alt_dict.items(), key=lambda x: x[1], reverse=True)
    for occ, csh_alt in sorted_by_alt[:10]:
        rti = rti_dict.get(occ, float('nan'))
        print(f"  {occ:4d}: CSH_alt={csh_alt:.4f}, RTI={rti:.4f}")

    print("\nBOTTOM 10 OCCUPATIONS BY CSH_alt (least routine-like)")
    print("-" * 60)
    for occ, csh_alt in sorted_by_alt[-10:]:
        rti = rti_dict.get(occ, float('nan'))
        print(f"  {occ:4d}: CSH_alt={csh_alt:.4f}, RTI={rti:.4f}")

    return {
        'routine_centroid': routine_centroid,
        'csh_alt': csh_alt_dict,
        'correlation': r,
    }


if __name__ == '__main__':
    results = main()
