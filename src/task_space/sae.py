"""
Sparse Autoencoder (SAE) for activity embedding decomposition.

Phase B of the Sparsity Hypothesis (v0.5.0):
Decomposes dense MPNet embeddings into sparse interpretable features.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SAEConfig:
    """SAE training configuration."""
    input_dim: int = 768
    hidden_dim: int = 16384
    learning_rate: float = 1e-3
    lambda_l1: float = 0.005
    noise_std: float = 0.01
    epochs: int = 500
    batch_size: int = 64
    early_stopping_patience: int = 50
    target_l0_min: int = 10
    target_l0_max: int = 20


@dataclass
class TrainingLog:
    """Training metrics log."""
    epochs: list[int]
    losses: list[float]
    reconstruction_losses: list[float]
    sparsity_losses: list[float]
    l0_values: list[float]
    final_lambda: float
    converged: bool
    best_epoch: int


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for decomposing dense embeddings.

    Architecture:
        Encoder: Linear(input_dim → hidden_dim) + ReLU
        Decoder: Linear(hidden_dim → input_dim)

    The ReLU activation enforces non-negativity in the sparse representation.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 16384):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse representation."""
        return torch.relu(self.encoder(x))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode sparse representation to reconstruction."""
        return self.decoder(f)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input embeddings, shape (batch, input_dim)

        Returns:
            Tuple of (reconstruction, sparse_features)
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


def sae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    f: torch.Tensor,
    lambda_l1: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SAE loss with L1 sparsity penalty.

    L = ||x - x_hat||² + λ * ||f||₁

    Args:
        x: Original input
        x_hat: Reconstruction
        f: Sparse features
        lambda_l1: Sparsity penalty weight

    Returns:
        Tuple of (total_loss, reconstruction_loss, sparsity_loss)
    """
    reconstruction_loss = torch.mean((x - x_hat) ** 2)
    sparsity_loss = torch.mean(torch.abs(f))
    total_loss = reconstruction_loss + lambda_l1 * sparsity_loss
    return total_loss, reconstruction_loss, sparsity_loss


def compute_l0(f: torch.Tensor, threshold: float = 0.01) -> float:
    """Compute average L0 (number of active features per input)."""
    return (f > threshold).float().sum(dim=1).mean().item()


def get_device() -> torch.device:
    """Get available device with fallback."""
    if torch.cuda.is_available():
        try:
            # Test actual GPU operation
            torch.zeros(1, device='cuda')
            return torch.device('cuda')
        except RuntimeError:
            print("GPU detected but not functional, falling back to CPU")
    return torch.device('cpu')


def train_sae(
    embeddings: np.ndarray,
    config: Optional[SAEConfig] = None,
    verbose: bool = True,
) -> tuple[SparseAutoencoder, TrainingLog]:
    """
    Train Sparse Autoencoder on embeddings.

    Args:
        embeddings: Input embeddings, shape (n_samples, input_dim)
        config: Training configuration
        verbose: Print progress

    Returns:
        Tuple of (trained_model, training_log)
    """
    if config is None:
        config = SAEConfig()

    device = get_device()
    if verbose:
        print(f"Training on device: {device}")

    # Prepare data
    dataset = torch.FloatTensor(embeddings).to(device)
    n_samples = dataset.shape[0]

    # Initialize model
    model = SparseAutoencoder(config.input_dim, config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training state
    lambda_l1 = config.lambda_l1
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    # Logging
    log_epochs = []
    log_losses = []
    log_recon = []
    log_sparse = []
    log_l0 = []

    for epoch in range(config.epochs):
        model.train()

        # Shuffle data
        perm = torch.randperm(n_samples)
        dataset_shuffled = dataset[perm]

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_sparse = 0.0
        epoch_l0 = 0.0
        n_batches = 0

        for i in range(0, n_samples, config.batch_size):
            batch = dataset_shuffled[i:i + config.batch_size]

            # Add noise for regularization
            noisy_input = batch + torch.randn_like(batch) * config.noise_std

            # Forward pass
            x_hat, f = model(noisy_input)

            # Compute loss (reconstruction on clean input)
            loss, recon_loss, sparse_loss = sae_loss(batch, x_hat, f, lambda_l1)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_sparse += sparse_loss.item()
            epoch_l0 += compute_l0(f)
            n_batches += 1

        # Average metrics
        epoch_loss /= n_batches
        epoch_recon /= n_batches
        epoch_sparse /= n_batches
        epoch_l0 /= n_batches

        # Log
        log_epochs.append(epoch)
        log_losses.append(epoch_loss)
        log_recon.append(epoch_recon)
        log_sparse.append(epoch_sparse)
        log_l0.append(epoch_l0)

        # Adaptive lambda tuning based on L0
        if epoch > 0 and epoch % 50 == 0:
            if epoch_l0 > config.target_l0_max:
                lambda_l1 *= 1.2  # Increase sparsity penalty
                if verbose:
                    print(f"  L0={epoch_l0:.1f} > {config.target_l0_max}, increasing λ to {lambda_l1:.4f}")
            elif epoch_l0 < config.target_l0_min:
                lambda_l1 *= 0.8  # Decrease sparsity penalty
                if verbose:
                    print(f"  L0={epoch_l0:.1f} < {config.target_l0_min}, decreasing λ to {lambda_l1:.4f}")

        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # Progress
        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:4d}: loss={epoch_loss:.4f}, recon={epoch_recon:.4f}, L0={epoch_l0:.1f}")

        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    converged = patience_counter >= config.early_stopping_patience

    training_log = TrainingLog(
        epochs=log_epochs,
        losses=log_losses,
        reconstruction_losses=log_recon,
        sparsity_losses=log_sparse,
        l0_values=log_l0,
        final_lambda=lambda_l1,
        converged=converged,
        best_epoch=best_epoch,
    )

    return model, training_log


def save_sae(
    model: SparseAutoencoder,
    training_log: TrainingLog,
    output_dir: Path,
    model_name: str = "sae_v1",
) -> None:
    """
    Save trained SAE model and training log.

    Args:
        model: Trained SparseAutoencoder
        training_log: Training metrics
        output_dir: Directory to save to
        model_name: Name for the model file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / f"{model_name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': model.input_dim,
        'hidden_dim': model.hidden_dim,
    }, model_path)

    # Save training log
    log_path = output_dir / f"{model_name}_training_log.json"
    log_dict = {
        'epochs': training_log.epochs,
        'losses': training_log.losses,
        'reconstruction_losses': training_log.reconstruction_losses,
        'sparsity_losses': training_log.sparsity_losses,
        'l0_values': training_log.l0_values,
        'final_lambda': training_log.final_lambda,
        'converged': training_log.converged,
        'best_epoch': training_log.best_epoch,
    }
    with open(log_path, 'w') as f:
        json.dump(log_dict, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Training log saved to {log_path}")


def load_sae(model_path: Path) -> SparseAutoencoder:
    """
    Load trained SAE model.

    Args:
        model_path: Path to saved model (.pt file)

    Returns:
        Loaded SparseAutoencoder
    """
    checkpoint = torch.load(model_path, map_location='cpu')

    model = SparseAutoencoder(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def extract_sparse_features(
    model: SparseAutoencoder,
    embeddings: np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """
    Extract sparse features from embeddings using trained SAE.

    Args:
        model: Trained SparseAutoencoder
        embeddings: Input embeddings, shape (n_samples, input_dim)
        threshold: Threshold for hard sparsity (set values below to 0)

    Returns:
        Sparse features, shape (n_samples, hidden_dim)
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        x = torch.FloatTensor(embeddings).to(device)
        features = model.encode(x)

        # Hard threshold for true sparsity
        features[features < threshold] = 0

        return features.cpu().numpy()
