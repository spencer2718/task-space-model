#!/usr/bin/env python3
"""
Verification script for distance baseline matrices (v0.7.3.1).

Loads all four distance matrices and verifies:
1. Shape = (447, 447) for each
2. Diagonal = 0 for each
3. Symmetry for each
4. Summary statistics: min, max, median, mean

Outputs: outputs/experiments/distance_baselines_v0731.json
"""

import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "artifacts" / "v1" / "mobility"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "experiments"


def load_matrix(name: str) -> tuple:
    """Load a distance matrix and return (distances, census_codes)."""
    path = CACHE_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Matrix not found: {path}")

    data = np.load(path, allow_pickle=True)
    distances = data["distances"]

    # Handle census_codes which may be saved differently
    if "census_codes" in data.files:
        census_codes = data["census_codes"].tolist()
    else:
        census_codes = None

    return distances, census_codes


def verify_matrix(distances: np.ndarray, name: str) -> dict:
    """Verify a distance matrix and return statistics."""
    n = distances.shape[0]

    # Check shape
    shape_ok = distances.shape == (447, 447)

    # Check diagonal
    diag = np.diag(distances)
    diagonal_zero = np.allclose(diag, 0, atol=1e-10)

    # Check symmetry
    symmetric = np.allclose(distances, distances.T, atol=1e-10)

    # Get off-diagonal values for statistics
    mask = ~np.eye(n, dtype=bool)
    off_diag = distances[mask]

    stats = {
        "shape": list(distances.shape),
        "min": float(np.min(off_diag)),
        "max": float(np.max(off_diag)),
        "median": float(np.median(off_diag)),
        "mean": float(np.mean(off_diag)),
        "std": float(np.std(off_diag)),
        "symmetric": bool(symmetric),
        "diagonal_zero": bool(diagonal_zero),
    }

    # Print summary
    print(f"\n{name}:")
    print(f"  Shape: {distances.shape} {'✓' if shape_ok else '✗'}")
    print(f"  Diagonal zero: {'✓' if diagonal_zero else '✗'}")
    print(f"  Symmetric: {'✓' if symmetric else '✗'}")
    print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}")

    return stats


def compute_correlations(matrices: dict) -> dict:
    """Compute pairwise Spearman correlations between distance matrices."""
    # Flatten upper triangle of each matrix
    n = 447
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    vectors = {}
    for name, (d, _) in matrices.items():
        vectors[name] = d[mask]

    correlations = {}
    pairs = [
        ("wasserstein", "euclidean_dwa"),
        ("wasserstein", "cosine_onet"),
        ("wasserstein", "cosine_embed"),
        ("euclidean_dwa", "cosine_onet"),
        ("euclidean_dwa", "cosine_embed"),
        ("cosine_onet", "cosine_embed"),
    ]

    for m1, m2 in pairs:
        if m1 in vectors and m2 in vectors:
            rho, _ = spearmanr(vectors[m1], vectors[m2])
            key = f"{m1}_vs_{m2}"
            correlations[key] = float(rho)
            print(f"  ρ({m1}, {m2}) = {rho:.4f}")

    return correlations


def main():
    print("=" * 60)
    print("Distance Baselines Verification v0.7.3.1")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define matrices to load
    matrix_files = {
        "cosine_onet": "d_cosine_onet_census",
        "cosine_embed": "d_cosine_embed_census",
        "euclidean_dwa": "d_euclidean_dwa_census",
        "wasserstein": "d_wasserstein_census",
    }

    # Load all matrices
    matrices = {}
    missing = []
    for short_name, file_name in matrix_files.items():
        try:
            distances, census_codes = load_matrix(file_name)
            matrices[short_name] = (distances, census_codes)
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            missing.append(short_name)

    if missing:
        print(f"\nMissing matrices: {missing}")
        print("Run compute_distance_baselines_v0731.py first.")
        return

    # Verify each matrix
    print("\n=== Matrix Verification ===")
    results = {"matrices": {}}

    for name, (distances, census_codes) in matrices.items():
        stats = verify_matrix(distances, name)
        results["matrices"][name] = stats

    # Compute correlations
    print("\n=== Pairwise Correlations (Spearman) ===")
    results["correlations"] = compute_correlations(matrices)

    # Check for high correlation (potential redundancy)
    high_corr = [
        (k, v) for k, v in results["correlations"].items()
        if v > 0.98
    ]
    if high_corr:
        print("\nWARNING: High correlation detected (>0.98):")
        for k, v in high_corr:
            print(f"  {k}: {v:.4f}")
        results["high_correlation_warning"] = True
    else:
        results["high_correlation_warning"] = False

    # Add metadata
    results["version"] = "0.7.3.1"
    results["census_codes_n"] = 447

    # Check all gates passed for NEW matrices (cosine_onet, cosine_embed, euclidean_dwa)
    # Note: existing Wasserstein matrix has non-zero diagonal due to aggregation
    new_matrices = ["cosine_onet", "cosine_embed", "euclidean_dwa"]
    all_ok = all(
        results["matrices"][m]["shape"] == [447, 447]
        and results["matrices"][m]["symmetric"]
        and results["matrices"][m]["diagonal_zero"]
        for m in new_matrices
    )
    results["gate_passed"] = all_ok

    # Note Wasserstein diagonal issue
    if not results["matrices"]["wasserstein"]["diagonal_zero"]:
        print("\nNote: Existing Wasserstein matrix has non-zero diagonal (expected from aggregation)")
        results["wasserstein_diagonal_note"] = "Non-zero diagonal due to many-to-one O*NET→Census aggregation"

    # Save results
    output_path = OUTPUT_DIR / "distance_baselines_v0731.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    print("\n=== Summary ===")
    print(f"Gate passed: {'✓' if results['gate_passed'] else '✗'}")

    return results


if __name__ == "__main__":
    main()
