#!/usr/bin/env python3
"""
Phase B: SAE Training

Trains a Sparse Autoencoder to decompose MPNet embeddings
into interpretable sparse features.

Usage:
    PYTHONPATH=src python tests/run_phase_b.py
"""

from pathlib import Path
import sys
import time

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from task_space import (
    build_dwa_activity_domain,
    SAEConfig,
    train_sae,
    save_sae,
    extract_sparse_features,
)


def get_embeddings(domain, cache_path: Path) -> np.ndarray:
    """Load or compute MPNet embeddings for DWA titles."""
    if cache_path.exists():
        print(f"      Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print("      Computing MPNet embeddings (first run downloads ~420MB model)...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-mpnet-base-v2')

    # Get DWA titles in order
    titles = [domain.activity_names[aid] for aid in domain.activity_ids]

    # Encode
    embeddings = model.encode(titles, show_progress_bar=True)

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"      Embeddings cached to {cache_path}")

    return embeddings


def main():
    print("=" * 60)
    print("Phase B: SAE Training")
    print("=" * 60)

    # Output directories
    output_dir = Path("outputs/phase_b")
    model_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "dwa_embeddings.npy"

    # Step 1: Load DWA domain
    print("\n[1/4] Loading DWA activity domain...")
    domain = build_dwa_activity_domain()
    print(f"      {domain.n_activities} activities loaded")

    # Step 2: Get embeddings
    print("\n[2/4] Getting MPNet embeddings...")
    embeddings = get_embeddings(domain, cache_path)
    print(f"      Embeddings shape: {embeddings.shape}")

    # Step 3: Train SAE
    print("\n[3/4] Training Sparse Autoencoder...")
    print(f"      Architecture: {embeddings.shape[1]} → 16384 → {embeddings.shape[1]}")
    print(f"      Target L0: 10-20 active features per input")

    config = SAEConfig(
        input_dim=embeddings.shape[1],
        hidden_dim=16384,
        learning_rate=1e-3,
        lambda_l1=0.005,
        noise_std=0.01,
        epochs=500,
        batch_size=64,
        early_stopping_patience=50,
        target_l0_min=10,
        target_l0_max=20,
    )

    start_time = time.time()
    model, training_log = train_sae(embeddings, config, verbose=True)
    elapsed = time.time() - start_time

    print(f"\n      Training complete in {elapsed:.1f}s")
    print(f"      Final L0: {training_log.l0_values[-1]:.1f}")
    print(f"      Final λ: {training_log.final_lambda:.4f}")
    print(f"      Best epoch: {training_log.best_epoch}")

    # Step 4: Save model and extract features
    print("\n[4/4] Saving model and extracting features...")
    save_sae(model, training_log, model_dir, "sae_v1")

    # Extract sparse features
    features = extract_sparse_features(model, embeddings, threshold=0.01)
    np.save(output_dir / "dwa_sparse_features.npy", features)

    # Compute feature statistics
    active_per_input = (features > 0).sum(axis=1)
    active_per_feature = (features > 0).sum(axis=0)
    n_dead_features = (active_per_feature == 0).sum()

    print(f"\n      Sparse features shape: {features.shape}")
    print(f"      Mean active features per DWA: {active_per_input.mean():.1f}")
    print(f"      Dead features (never activate): {n_dead_features} / {features.shape[1]}")

    # Save summary
    summary = {
        "embeddings_shape": list(embeddings.shape),
        "features_shape": list(features.shape),
        "training_time_seconds": elapsed,
        "final_l0": training_log.l0_values[-1],
        "final_lambda": training_log.final_lambda,
        "best_epoch": training_log.best_epoch,
        "converged": training_log.converged,
        "mean_active_per_dwa": float(active_per_input.mean()),
        "dead_features": int(n_dead_features),
        "final_reconstruction_loss": training_log.reconstruction_losses[-1],
    }

    import json
    with open(output_dir / "phase_b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("PHASE B COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {model_dir / 'sae_v1.pt'}")
    print(f"Features saved to: {output_dir / 'dwa_sparse_features.npy'}")
    print("\n→ Proceed to Phase C (Feature Inspection)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
