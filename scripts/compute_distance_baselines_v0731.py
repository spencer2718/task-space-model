#!/usr/bin/env python3
"""
Compute distance baseline matrices for v0.7.3.1.

Computes three distance measures at Census 2010 code level (447×447):
- D1: Cosine distance on O*NET DWA importance vectors
- D2: Cosine distance on MPNet occupation centroid embeddings
- D3: Euclidean distance on normalized DWA weights

Uses existing infrastructure:
- domain.py: build_dwa_occupation_measures() for DWA importance
- artifacts.py: get_embeddings() for MPNet embeddings (HC: never compute elsewhere)
- census_crosswalk.py: aggregate_distances_to_census() for O*NET → Census

Outputs to:
- .cache/artifacts/v1/mobility/d_cosine_onet_census.npz
- .cache/artifacts/v1/mobility/d_cosine_embed_census.npz
- .cache/artifacts/v1/mobility/d_euclidean_dwa_census.npz
"""

import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from task_space.domain import build_dwa_occupation_measures
from task_space.data.artifacts import get_embeddings
from task_space.data.onet import load_dwa_reference
from task_space.mobility.census_crosswalk import (
    load_census_onet_crosswalk,
    aggregate_distances_to_census,
)


CACHE_DIR = Path(__file__).parent.parent / ".cache" / "artifacts" / "v1" / "mobility"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "experiments"


def compute_d1_cosine_onet(occ_measures, onet_codes):
    """
    D1: Cosine distance on O*NET DWA importance vectors.

    Uses raw (unnormalized) DWA importance ratings.
    Distance = 1 - cosine_similarity(occ_i, occ_j)
    """
    print("\n=== D1: Cosine distance on O*NET importance vectors ===")
    print(f"Matrix shape: {occ_measures.raw_matrix.shape}")

    # Use raw importance values (not normalized to probability)
    importance_matrix = occ_measures.raw_matrix

    # Cosine distance
    d_cosine = cosine_distances(importance_matrix)

    print(f"Cosine distance range: [{d_cosine.min():.4f}, {d_cosine.max():.4f}]")
    print(f"Diagonal zero check: {np.allclose(np.diag(d_cosine), 0)}")

    return d_cosine


def compute_d2_cosine_embed(occ_measures, onet_codes):
    """
    D2: Cosine distance on MPNet occupation centroid embeddings.

    Compute occupation centroid as importance-weighted average of DWA embeddings.
    Uses get_embeddings() from artifacts.py (HC: never compute embeddings elsewhere).
    """
    print("\n=== D2: Cosine distance on MPNet embeddings ===")

    # Get DWA titles for embedding
    dwa_ref = load_dwa_reference()
    dwa_titles = dict(zip(dwa_ref["DWA ID"], dwa_ref["DWA Title"]))

    # Align DWA IDs with occupation matrix columns
    dwa_ids = occ_measures.activity_ids
    dwa_texts = [dwa_titles.get(dwa_id, dwa_id) for dwa_id in dwa_ids]

    print(f"Getting embeddings for {len(dwa_texts)} DWAs...")

    # Get embeddings via canonical function (HC: never compute elsewhere)
    dwa_embeddings = get_embeddings(dwa_texts, model="all-mpnet-base-v2")
    print(f"DWA embeddings shape: {dwa_embeddings.shape}")

    # Compute occupation centroids: importance-weighted average of DWA embeddings
    # Use raw importance as weights (not normalized to probability)
    importance_matrix = occ_measures.raw_matrix

    # Normalize weights per occupation for weighted average
    weights = importance_matrix.copy()
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    weights = weights / row_sums

    # Compute centroids: (n_occ, n_dwa) @ (n_dwa, embed_dim) = (n_occ, embed_dim)
    occupation_centroids = weights @ dwa_embeddings
    print(f"Occupation centroids shape: {occupation_centroids.shape}")

    # Cosine distance between centroids
    d_cosine_embed = cosine_distances(occupation_centroids)

    print(f"Cosine embedding distance range: [{d_cosine_embed.min():.4f}, {d_cosine_embed.max():.4f}]")
    print(f"Diagonal zero check: {np.allclose(np.diag(d_cosine_embed), 0)}")

    return d_cosine_embed


def compute_d3_euclidean_dwa(occ_measures, onet_codes):
    """
    D3: Euclidean distance on normalized DWA weights.

    Normalize vectors to sum to 1 (treat as probability distributions).
    This is the closest analogue to Cortes-Gallipoli's DOT Euclidean.
    """
    print("\n=== D3: Euclidean distance on DWA weights ===")

    # Use normalized probability distributions (rows sum to 1)
    prob_matrix = occ_measures.occupation_matrix

    # Verify normalization
    row_sums = prob_matrix.sum(axis=1)
    print(f"Row sums range: [{row_sums.min():.6f}, {row_sums.max():.6f}]")

    # Euclidean distance
    d_euclidean = euclidean_distances(prob_matrix)

    print(f"Euclidean distance range: [{d_euclidean.min():.4f}, {d_euclidean.max():.4f}]")
    print(f"Diagonal zero check: {np.allclose(np.diag(d_euclidean), 0)}")

    return d_euclidean


def aggregate_to_census(d_onet, onet_codes, crosswalk):
    """Aggregate O*NET-level distance matrix to Census 2010 level."""
    d_census, census_codes = aggregate_distances_to_census(
        d_onet, onet_codes, crosswalk, aggregation="mean"
    )
    # Zero the diagonal - d(x,x) = 0 by definition
    # Non-zero diagonal arises from many-to-one O*NET → Census aggregation
    np.fill_diagonal(d_census, 0)
    return d_census, census_codes


def main():
    print("=" * 60)
    print("Distance Baselines Computation v0.7.3.1")
    print("=" * 60)

    start_time = time.time()

    # Ensure output directories exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load DWA occupation measures
    print("\nLoading DWA occupation measures...")
    occ_measures = build_dwa_occupation_measures(normalize=True)
    onet_codes = occ_measures.occupation_codes
    print(f"Loaded {len(onet_codes)} O*NET occupations × {len(occ_measures.activity_ids)} DWAs")

    # Load crosswalk
    print("\nLoading Census crosswalk...")
    crosswalk = load_census_onet_crosswalk()
    print(f"Crosswalk: {crosswalk.n_matched} O*NET → {crosswalk.n_census} Census codes")
    print(f"Coverage: {crosswalk.coverage:.1%}")

    # Compute D1: Cosine on O*NET importance
    d1_onet = compute_d1_cosine_onet(occ_measures, onet_codes)
    d1_census, census_codes = aggregate_to_census(d1_onet, onet_codes, crosswalk)
    print(f"Aggregated to Census: {d1_census.shape}")

    # Compute D2: Cosine on MPNet embeddings
    d2_onet = compute_d2_cosine_embed(occ_measures, onet_codes)
    d2_census, _ = aggregate_to_census(d2_onet, onet_codes, crosswalk)
    print(f"Aggregated to Census: {d2_census.shape}")

    # Compute D3: Euclidean on DWA weights
    d3_onet = compute_d3_euclidean_dwa(occ_measures, onet_codes)
    d3_census, _ = aggregate_to_census(d3_onet, onet_codes, crosswalk)
    print(f"Aggregated to Census: {d3_census.shape}")

    # Save matrices
    print("\n=== Saving matrices ===")

    np.savez_compressed(
        CACHE_DIR / "d_cosine_onet_census.npz",
        distances=d1_census,
        census_codes=census_codes,
        metric="cosine_onet",
        description="Cosine distance on O*NET DWA importance vectors",
    )
    print(f"Saved: {CACHE_DIR / 'd_cosine_onet_census.npz'}")

    np.savez_compressed(
        CACHE_DIR / "d_cosine_embed_census.npz",
        distances=d2_census,
        census_codes=census_codes,
        metric="cosine_embed",
        description="Cosine distance on MPNet occupation centroid embeddings",
    )
    print(f"Saved: {CACHE_DIR / 'd_cosine_embed_census.npz'}")

    np.savez_compressed(
        CACHE_DIR / "d_euclidean_dwa_census.npz",
        distances=d3_census,
        census_codes=census_codes,
        metric="euclidean_dwa",
        description="Euclidean distance on normalized DWA weight distributions",
    )
    print(f"Saved: {CACHE_DIR / 'd_euclidean_dwa_census.npz'}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    return d1_census, d2_census, d3_census, census_codes


if __name__ == "__main__":
    main()
