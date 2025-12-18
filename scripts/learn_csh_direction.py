"""
Learn CSH direction vector from RTI.

Task 3 of v0.7.2.2: Find direction v in R^768 that maximizes
correlation between occupation centroid projections and traditional RTI.

CSH_i = centroid_i · v
Goal: maximize r(CSH, RTI)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Paths
REPO_ROOT = Path(__file__).parent.parent
CENTROIDS_PATH = REPO_ROOT / ".cache/artifacts/v1/embeddings/occ1990dd_centroids_mpnet.npz"
ALM_PATH = REPO_ROOT / "data/external/dorn_replication/occ1990dd_task_alm.dta"
OUTPUT_DIR = REPO_ROOT / "outputs/experiments"
DIRECTION_OUTPUT = REPO_ROOT / ".cache/artifacts/v1/embeddings/rti_direction_v0722.npz"


def load_centroids():
    """Load occ1990dd embedding centroids."""
    data = np.load(CENTROIDS_PATH)
    occ_codes = data['occ_codes']
    centroids = data['centroids']
    print(f"Loaded {len(occ_codes)} occ1990dd centroids with {centroids.shape[1]} dimensions")
    return dict(zip(occ_codes, centroids))


def load_rti():
    """Load and compute traditional RTI from ALM scores."""
    df = pd.read_stata(ALM_PATH)

    # RTI = ln(Routine) - ln(Manual) - ln(Abstract)
    # Using ALM task measures (columns: task_routine, task_manual, task_abstract)
    rti_values = {}

    for _, row in df.iterrows():
        occ = int(row['occ1990dd'])
        routine = row['task_routine']
        manual = row['task_manual']
        abstract = row['task_abstract']

        if pd.notna(routine) and pd.notna(manual) and pd.notna(abstract):
            if routine > 0 and manual > 0 and abstract > 0:
                rti = np.log(routine) - np.log(manual) - np.log(abstract)
                rti_values[occ] = rti

    print(f"Computed RTI for {len(rti_values)} occupations")
    return rti_values


def learn_direction_vector(centroids_dict, rti_dict, method='ridge'):
    """
    Learn direction vector v that maximizes correlation between
    centroid projections and RTI.

    Methods:
    - 'simple': v = X^T y / ||X^T y|| (prone to overfitting with p >> n)
    - 'ridge': Ridge regression with cross-validation
    - 'pca': PCA dimensionality reduction + linear regression

    For high-dimensional settings (768 dims, ~260 samples), regularization is essential.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Find common occupations
    common_occs = sorted(set(centroids_dict.keys()) & set(rti_dict.keys()))
    print(f"Found {len(common_occs)} occupations with both centroids and RTI")

    # Build matrices
    X = np.array([centroids_dict[occ] for occ in common_occs])  # (n, 768)
    y = np.array([rti_dict[occ] for occ in common_occs])  # (n,)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"RTI stats: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}, std={y.std():.3f}")

    # Standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    y_mean, y_std_val = y.mean(), y.std()
    y_std = (y - y_mean) / y_std_val

    if method == 'simple':
        # Simple correlation maximization (may overfit)
        v_raw = X_std.T @ y_std
        v = v_raw / np.linalg.norm(v_raw)

    elif method == 'ridge':
        # Ridge regression with cross-validation
        # Test multiple alpha values
        alphas = np.logspace(-2, 4, 50)
        ridge = RidgeCV(alphas=alphas, cv=5)
        ridge.fit(X_std, y_std)
        print(f"Ridge best alpha: {ridge.alpha_:.4f}")

        v_raw = ridge.coef_
        v = v_raw / np.linalg.norm(v_raw)

    elif method == 'pca':
        # PCA + regression: reduce to 50 components, then regress
        n_components = min(50, X.shape[0] - 1)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_std)

        # Simple regression in PCA space
        # v_pca = (X_pca^T X_pca)^{-1} X_pca^T y
        v_pca = np.linalg.lstsq(X_pca, y_std, rcond=None)[0]

        # Transform back to original space
        v_raw = pca.components_.T @ v_pca  # (768,)
        v = v_raw / np.linalg.norm(v_raw)

        print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    print(f"Direction vector norm: {np.linalg.norm(v):.6f}")

    return v, common_occs, X, y


def compute_csh(centroids_dict, v):
    """Compute CSH for all occupations."""
    csh_values = {}
    for occ, centroid in centroids_dict.items():
        csh_values[occ] = float(np.dot(centroid, v))
    return csh_values


def validate_correlation(csh_dict, rti_dict, common_occs):
    """Validate correlation between CSH and RTI."""
    csh_vals = [csh_dict[occ] for occ in common_occs]
    rti_vals = [rti_dict[occ] for occ in common_occs]

    r, p = stats.pearsonr(csh_vals, rti_vals)
    rho, p_rho = stats.spearmanr(csh_vals, rti_vals)

    print(f"\nCorrelation validation:")
    print(f"  Pearson r(CSH, RTI) = {r:.4f} (p = {p:.2e})")
    print(f"  Spearman rho(CSH, RTI) = {rho:.4f} (p = {p_rho:.2e})")
    print(f"  Gate: 0.7 <= r <= 0.9")

    gate_passed = 0.7 <= abs(r) <= 0.9
    print(f"  Gate {'PASSED' if gate_passed else 'FAILED'}")

    return {
        'pearson_r': r,
        'pearson_p': p,
        'spearman_rho': rho,
        'spearman_p': p_rho,
        'n_common': len(common_occs),
        'gate_passed': gate_passed,
    }


def main():
    print("=" * 60)
    print("CSH Direction Vector Learning (v0.7.2.2 Task 3)")
    print("=" * 60)
    print()

    # Load data
    centroids_dict = load_centroids()
    rti_dict = load_rti()

    # Learn direction vector
    print("\nLearning direction vector...")
    v, common_occs, X, y = learn_direction_vector(centroids_dict, rti_dict)

    # Compute CSH
    print("\nComputing CSH...")
    csh_dict = compute_csh(centroids_dict, v)

    # CSH statistics
    csh_vals = list(csh_dict.values())
    print(f"CSH stats: min={min(csh_vals):.4f}, max={max(csh_vals):.4f}, "
          f"mean={np.mean(csh_vals):.4f}, std={np.std(csh_vals):.4f}")

    # Validate correlation
    corr_results = validate_correlation(csh_dict, rti_dict, common_occs)

    # Save direction vector
    print(f"\nSaving direction vector to {DIRECTION_OUTPUT}")
    DIRECTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        DIRECTION_OUTPUT,
        direction=v,
        correlation=corr_results['pearson_r'],
        n_occupations=len(common_occs),
    )

    # Save CSH values
    csh_output = OUTPUT_DIR / "csh_values_v0722.csv"
    csh_df = pd.DataFrame([
        {'occ1990dd': occ, 'csh': csh, 'rti': rti_dict.get(occ)}
        for occ, csh in csh_dict.items()
    ])
    csh_df.to_csv(csh_output, index=False)
    print(f"Saved CSH values to {csh_output}")

    # Print top/bottom occupations by CSH
    print("\n" + "=" * 60)
    print("TOP 10 OCCUPATIONS BY CSH (most routine-like in embedding space)")
    print("-" * 60)
    sorted_by_csh = sorted(csh_dict.items(), key=lambda x: x[1], reverse=True)
    for occ, csh in sorted_by_csh[:10]:
        rti = rti_dict.get(occ, float('nan'))
        print(f"  {occ:4d}: CSH={csh:7.4f}, RTI={rti:7.4f}")

    print("\nBOTTOM 10 OCCUPATIONS BY CSH (least routine-like)")
    print("-" * 60)
    for occ, csh in sorted_by_csh[-10:]:
        rti = rti_dict.get(occ, float('nan'))
        print(f"  {occ:4d}: CSH={csh:7.4f}, RTI={rti:7.4f}")

    # Return results for further use
    return {
        'direction': v,
        'csh_values': csh_dict,
        'rti_values': rti_dict,
        'correlation': corr_results,
        'common_occupations': common_occs,
    }


if __name__ == '__main__':
    results = main()
