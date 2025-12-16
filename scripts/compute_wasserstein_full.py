"""
Compute full 894×894 Wasserstein distance matrix.

Expected runtime: 30-60 seconds based on integration test scaling.
Output: .cache/artifacts/v1/wasserstein/d_wasserstein_onet.npz
"""
import numpy as np
from pathlib import Path

from task_space.domain import build_dwa_occupation_measures
from task_space.data import get_dwa_titles
from task_space.data.artifacts import get_embeddings, get_distance_matrix
from task_space.similarity.wasserstein import compute_wasserstein_distances


def main():
    print("=" * 60)
    print("Computing Full Wasserstein Distance Matrix")
    print("=" * 60)

    # Load occupation measures
    print("\n1. Loading DWA occupation measures...")
    measures = build_dwa_occupation_measures()
    occ_matrix = measures.occupation_matrix
    print(f"   Shape: {occ_matrix.shape}")
    print(f"   Sparsity: {(occ_matrix == 0).mean():.1%}")

    # Get activity titles for embeddings
    print("\n2. Loading activity embeddings (MPNet)...")
    dwa_titles = get_dwa_titles()
    activity_titles = [dwa_titles.get(aid, aid) for aid in measures.activity_ids]
    embeddings = get_embeddings(activity_titles, model="all-mpnet-base-v2")
    print(f"   Embedding shape: {embeddings.shape}")

    print("\n3. Computing ground distance matrix...")
    ground = get_distance_matrix(embeddings, metric="cosine")
    print(f"   Ground metric shape: {ground.shape}")
    print(f"   Distance range: [{ground.min():.4f}, {ground.max():.4f}]")

    # Compute Wasserstein
    print("\n4. Computing pairwise Wasserstein distances...")
    n_occ = occ_matrix.shape[0]
    n_pairs = n_occ * (n_occ - 1) // 2
    print(f"   Pairs to compute: {n_pairs:,}")

    result = compute_wasserstein_distances(
        occ_matrix,
        ground,
        n_jobs=-1,
        verbose=True
    )

    # Save results
    print("\n5. Saving results...")
    output_dir = Path(".cache/artifacts/v1/wasserstein")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "d_wasserstein_onet.npz"
    np.savez_compressed(
        output_path,
        distance_matrix=result.distance_matrix,
        occupation_codes=np.array(measures.occupation_codes),
        n_occupations=result.n_occupations,
        n_activities=result.n_activities,
        computation_time=result.computation_time_seconds,
        median_support_size=result.median_support_size,
    )
    print(f"   Saved to: {output_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Matrix shape: {result.distance_matrix.shape}")
    print(f"Computation time: {result.computation_time_seconds:.1f}s")
    print(f"Median support union size: {result.median_support_size:.0f}")

    D = result.distance_matrix
    nonzero = D[np.triu_indices_from(D, k=1)]
    print(f"Distance range: [{nonzero.min():.4f}, {nonzero.max():.4f}]")
    print(f"Mean distance: {nonzero.mean():.4f}")
    print(f"Median distance: {np.median(nonzero):.4f}")

    return result


if __name__ == "__main__":
    main()
