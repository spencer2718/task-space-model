#!/usr/bin/env python3
"""
Phase 1 Diagnostics Runner for v0.6.1.

Runs all Phase 1 diagnostic tasks and generates outputs.

Usage:
    PYTHONPATH=src python tests/run_phase1_diagnostics.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from task_space.domain import build_dwa_activity_domain, build_dwa_occupation_measures
from task_space.distances import compute_text_embedding_distances
from task_space.baseline import compute_binary_overlap
from task_space.diagnostics_v061 import (
    diagnose_distance_distribution,
    verify_similarity_orientation,
    diagnose_kernel_weights,
    correlate_jaccard_semantic,
    compute_semantic_similarity_matrix,
    compute_occupation_semantic_overlap,
    save_phase1_results,
    generate_phase1_summary,
)


def main():
    output_dir = Path("outputs/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1: Implementation Verification")
    print("=" * 60)

    # Step 1: Build activity domain and occupation measures
    print("\n[1/6] Building DWA activity domain and occupation measures...")
    domain = build_dwa_activity_domain()
    measures = build_dwa_occupation_measures()
    print(f"  - {domain.n_activities} activities")
    print(f"  - {len(measures.occupation_codes)} occupations")

    # Step 2: Compute text embeddings and distances
    print("\n[2/6] Computing text embeddings (this may take a minute)...")
    activity_titles = [domain.activity_names[aid] for aid in domain.activity_ids]

    # Use GPU if available
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  - Using device: {device}")

    distances = compute_text_embedding_distances(
        activity_titles=activity_titles,
        activity_ids=domain.activity_ids,
        model_name="all-mpnet-base-v2",
        metric="cosine",
    )

    # Save embeddings for later use
    np.save(output_dir / "activity_embeddings.npy", distances.activity_profiles)
    print(f"  - Saved embeddings: {distances.activity_profiles.shape}")

    # Step 3: Run Task 1.1.1 - Distance Distribution Diagnostics
    print("\n[3/6] Task 1.1.1: Distance Distribution Diagnostics...")
    distance_result = diagnose_distance_distribution(distances.distance_matrix)
    print(f"  - Mean: {distance_result.mean:.4f}")
    print(f"  - Std: {distance_result.std:.4f}")
    print(f"  - CV: {distance_result.coefficient_of_variation:.4f}")
    print(f"  - Effective range: {distance_result.effective_range:.4f}")
    print(f"  - Degenerate: {distance_result.is_degenerate}")
    print(f"  - Clustered: {distance_result.is_clustered}")
    print(f"  - Diagnosis: {distance_result.diagnosis}")

    # Step 4: Run Task 1.1.2 - Similarity Orientation Check
    print("\n[4/6] Task 1.1.2: Similarity Orientation Check...")
    orientation_result = verify_similarity_orientation(
        model_name="all-mpnet-base-v2",
        device=device,
    )
    print(f"  - Mean similar: {orientation_result.mean_similar:.4f}")
    print(f"  - Mean dissimilar: {orientation_result.mean_dissimilar:.4f}")
    print(f"  - Gap: {orientation_result.gap:.4f}")
    print(f"  - Orientation correct: {orientation_result.orientation_correct}")
    print(f"  - Diagnosis: {orientation_result.diagnosis}")

    # Step 5: Run Task 1.1.3 - Kernel Weight Distribution
    print("\n[5/6] Task 1.1.3: Kernel Weight Distribution...")
    kernel_result = diagnose_kernel_weights(distances.distance_matrix)
    print(f"  - Recommended sigma: {kernel_result.recommended_sigma:.4f}")
    print(f"  - All collapsed: {kernel_result.all_collapsed}")
    for label, result in kernel_result.sigma_results.items():
        status = "COLLAPSED" if result.is_collapsed else "OK"
        print(f"    {label}: sigma={result.sigma:.4f}, range={result.weight_range:.4f} [{status}]")
    print(f"  - Diagnosis: {kernel_result.diagnosis}")

    # Step 6: Run Task 1.1.4 - Jaccard-Semantic Correlation
    print("\n[6/6] Task 1.1.4: Jaccard-Semantic Correlation...")

    # Compute binary Jaccard
    jaccard_result = compute_binary_overlap(measures, threshold=0.0)
    jaccard_matrix = jaccard_result.overlap_matrix

    # Compute semantic similarity matrix from activity embeddings
    activity_sim_matrix = compute_semantic_similarity_matrix(distances.activity_profiles)

    # Compute occupation-level semantic overlap
    semantic_overlap = compute_occupation_semantic_overlap(
        measures.occupation_matrix,
        activity_sim_matrix,
    )

    correlation_result = correlate_jaccard_semantic(
        jaccard_matrix=jaccard_matrix,
        semantic_sim_matrix=semantic_overlap,
        occupation_codes=measures.occupation_codes,
    )
    print(f"  - Pearson r: {correlation_result.pearson_r:.4f}")
    print(f"  - Spearman rho: {correlation_result.spearman_rho:.4f}")
    print(f"  - Interpretation: {correlation_result.interpretation}")
    print(f"  - Diagnosis: {correlation_result.diagnosis}")

    # Generate scatter plot
    print("\n[7/7] Generating scatter plot...")
    n = jaccard_matrix.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    jaccard_vals = jaccard_matrix[triu_indices]
    semantic_vals = semantic_overlap[triu_indices]

    # Sample for visualization (too many points otherwise)
    n_sample = min(10000, len(jaccard_vals))
    np.random.seed(42)
    sample_idx = np.random.choice(len(jaccard_vals), n_sample, replace=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        jaccard_vals[sample_idx],
        semantic_vals[sample_idx],
        alpha=0.3,
        s=5,
        c="steelblue",
    )
    ax.set_xlabel("Binary Jaccard Overlap")
    ax.set_ylabel("Semantic Overlap")
    ax.set_title(f"Jaccard vs Semantic Overlap (r = {correlation_result.pearson_r:.3f})")

    # Add trend line
    z = np.polyfit(jaccard_vals, semantic_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(jaccard_vals.min(), jaccard_vals.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f"Linear fit")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "jaccard_semantic_scatter.png", dpi=150)
    plt.close()
    print(f"  - Saved: {output_dir}/jaccard_semantic_scatter.png")

    # Save all results
    print("\n" + "=" * 60)
    print("Saving results...")
    save_phase1_results(
        output_dir=output_dir,
        distance_result=distance_result,
        orientation_result=orientation_result,
        kernel_result=kernel_result,
        correlation_result=correlation_result,
    )

    # Generate and save summary
    summary = generate_phase1_summary(
        output_dir=output_dir,
        distance_result=distance_result,
        orientation_result=orientation_result,
        kernel_result=kernel_result,
        correlation_result=correlation_result,
    )

    with open(output_dir / "phase1_summary.md", "w") as f:
        f.write(summary)

    print(f"\nOutputs saved to: {output_dir}")
    print("  - distance_distribution.json")
    print("  - similarity_orientation.json")
    print("  - kernel_weights.json")
    print("  - jaccard_semantic_correlation.json")
    print("  - jaccard_semantic_scatter.png")
    print("  - phase1_summary.md")

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"\nSummary written to: {output_dir}/phase1_summary.md")
    print("\nNext steps:")
    if distance_result.is_degenerate or distance_result.is_clustered:
        print("  - FIX: Distance distribution issues detected")
    elif not orientation_result.orientation_correct:
        print("  - FIX: Similarity orientation issues detected")
    elif kernel_result.all_collapsed:
        print("  - FIX: Kernel weight collapse detected")
    elif correlation_result.interpretation == "sign_flip":
        print("  - FIX: Sign flip detected in Jaccard-semantic correlation")
    else:
        print("  - No bugs found. Proceed to Phase 2 (Alternative Distances)")


if __name__ == "__main__":
    main()
