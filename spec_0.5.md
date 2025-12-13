# Specification v0.5.0: The Sparsity Hypothesis

**Document ID:** SPEC-v0.5.0  
**Status:** APPROVED  
**Date:** 2025-12-13  
**Authors:** Principal Investigator, Research Lead  
**Assignee:** Software Engineering Agent  

---

## 1. Executive Summary

Version 0.4.2.1 established that dense semantic embeddings fail to predict wage comovement—and that random distances outperform carefully-computed text embeddings. This specification defines a pivot from continuous dense geometry to sparse discrete features via Sparse Autoencoder (SAE) decomposition.

**Core Hypothesis:** The labor market is sparse. Economic substitutability is binary ("you have the skill or you don't"), not continuous. Dense embeddings impose the wrong representational format.

**Deliverable:** A validated comparison between Raw Binary Overlap (baseline) and SAE-derived Sparse Overlap against wage comovement.

---

## 2. Background

### 2.1 The v0.4.2 Failure Mode

The v0.4.2 validation appeared to pass (β > 0, t ≈ 5.17), but robustness checks revealed:

| Check | Result | Implication |
|-------|--------|-------------|
| Permutation Test | p = 0.31 | Effect indistinguishable from shuffled data |
| Placebo Test | β_random / β_real = 2.5x | Random distances work *better* |
| σ-Collinearity | r = 0.999 | Bandwidth parameter is inert |

**Diagnosis:** The correlation exists but is driven by occupation-activity matrix structure (occupations sharing activities have correlated wages), not the geometric properties of the activity space.

### 2.2 The Sparsity Hypothesis

MPNet embeddings encode semantic similarity, but compress distinct economic capabilities into overlapping regions (polysemanticity). "Python programming" and "SQL queries" become entangled in a shared "Technical/Computer" direction.

An SAE projects the 768-dimensional dense space into a 16,384-dimensional sparse space where distinct capabilities occupy orthogonal directions. This converts continuous similarity into discrete feature intersection.

**Theoretical justification:** The SAE does not add knowledge—it adds resolution. If MPNet distinguishes "welding" from "typing" anywhere in its representation (even superposed), the SAE can disentangle them.

---

## 3. Execution Plan

### 3.1 Phase A: Raw Binary Baseline (Priority 1)

**Objective:** Establish the floor. Does counting shared activities predict wage comovement?

#### 3.1.1 Data Preparation

Load the occupation-activity weight matrix W (shape: J × N_dwa, where J ≈ 700-900 occupations, N_dwa = 2,087 activities).

**Source files:**
- `data/onet/db_30_0_excel/Task Ratings.xlsx`
- `data/onet/db_30_0_excel/Tasks to DWAs.xlsx`
- `data/onet/db_30_0_excel/DWA Reference.xlsx`

#### 3.1.2 Binary Matrix Construction

```python
# Binarize at threshold τ (default: τ = 0, i.e., any positive weight)
B[j, a] = 1 if W[j, a] > τ else 0
```

#### 3.1.3 Jaccard Overlap Computation

For each occupation pair (i, j):

```
BinaryOverlap_{i,j} = |A_i ∩ A_j| / |A_i ∪ A_j|
```

**Vectorized implementation:**
```python
intersection = B @ B.T  # (J × J) matrix of |A_i ∩ A_j|
row_sums = B.sum(axis=1, keepdims=True)
union = row_sums + row_sums.T - intersection
binary_overlap = intersection / union
```

#### 3.1.4 Validation Regression

```
WageComovement_{i,j} = α + β · BinaryOverlap_{i,j} + ε_{i,j}
```

Standard errors clustered by origin occupation.

**Output:** β̂, SE, t-stat, p-value, R²

#### 3.1.5 Decision Gate

| Binary Result | Interpretation | Action |
|---------------|----------------|--------|
| β ≈ 0, p > 0.10 | O*NET structure doesn't predict wages | Investigate data quality |
| β > 0, p < 0.10 | Baseline established | SAE must beat this R² |

---

### 3.2 Phase B: SAE Training

**Objective:** Train a Sparse Autoencoder to decompose MPNet embeddings into interpretable features.

#### 3.2.1 Embedding Extraction

Extract MPNet embeddings for all 2,087 DWA titles:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(dwa_titles)  # Shape: (2087, 768)
```

**Cache location:** `outputs/phase_i_dwa/dwa_embeddings.npy`

#### 3.2.2 Architecture Specification

| Component | Specification |
|-----------|---------------|
| Input dimension | 768 |
| Hidden dimension | 16,384 (expansion factor ≈ 21×) |
| Activation | ReLU (enforces non-negativity) |
| Sparsity target | L₀ ≈ 10-20 active features per input |
| Output dimension | 768 (reconstruction) |

**Architecture:**
```
Encoder: Linear(768 → 16384) + ReLU
Decoder: Linear(16384 → 768)
```

#### 3.2.3 Loss Function

```
L = ||x - x̂||² + λ · ||f||₁
```

Where:
- x is the input embedding
- x̂ is the reconstruction
- f is the sparse hidden representation
- λ is the sparsity penalty (tuning parameter)

**Tuning procedure:**
1. Start with λ = 0.005
2. Monitor average L₀ (number of non-zero activations per input)
3. Adjust λ to achieve L₀ ∈ [10, 20]
4. If L₀ > 20: increase λ
5. If L₀ < 10: decrease λ

#### 3.2.4 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam | Standard for small models |
| Learning rate | 1e-3 | Aggressive for small dataset |
| Batch size | 64 | Small dataset allows this |
| Epochs | 500 | ~16k steps, fast convergence |
| Data augmentation | Gaussian noise (σ=0.01) | Prevent memorization |
| Early stopping | 50 epochs without improvement | Prevent overfitting |

#### 3.2.5 Hardware Considerations

**Primary path: CPU training**

The training task is small (2,087 samples × 768 dimensions × 16,384 hidden). CPU training is expected to complete in 5-15 minutes. This is the recommended path due to ROCm compatibility complexities on Ubuntu 24.04.

```python
device = torch.device('cpu')
```

**Optional: ROCm GPU acceleration**

If GPU acceleration is desired, note the following Ubuntu 24.04 compatibility issues:

1. **ROCm 6.1 is NOT supported on Ubuntu 24.04** — use ROCm 6.2+ or 7.1+
2. **Docker is the recommended approach** for stable ROCm+PyTorch:
   ```bash
   docker run -it \
     --device=/dev/kfd --device=/dev/dri \
     --group-add video --ipc=host --shm-size 8G \
     rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
   ```
3. **Some GPUs require HSA_OVERRIDE_GFX_VERSION:**
   - RDNA2 (RX 6000 series): `export HSA_OVERRIDE_GFX_VERSION=10.3.0`
   - RDNA3 (RX 7000 series): `export HSA_OVERRIDE_GFX_VERSION=11.0.0`
4. **Known issues:** PyTorch source builds on Ubuntu 24.04 have reported RUNPATH problems (pytorch/pytorch#137858)

**Fallback strategy:**
```python
def get_device():
    if torch.cuda.is_available():
        try:
            # Test actual GPU operation
            torch.zeros(1, device='cuda')
            return torch.device('cuda')
        except RuntimeError:
            print("GPU detected but not functional, falling back to CPU")
    return torch.device('cpu')
```

#### 3.2.6 Deliverables

- `models/sae_v1.pt` — Trained model checkpoint
- `outputs/phase_i_dwa/sae_training_log.json` — Training metrics (loss, L₀ per epoch)

---

### 3.3 Phase C: Feature Extraction and Inspection

**Objective:** Extract sparse features and validate interpretability.

#### 3.3.1 Feature Extraction

```python
# Load trained SAE
sae = load_model('models/sae_v1.pt')

# Extract sparse features for all DWAs
with torch.no_grad():
    features = sae.encode(embeddings)  # Shape: (2087, 16384)

# Hard threshold for true sparsity
features[features < 0.01] = 0
```

**Output:** `outputs/phase_i_dwa/dwa_sparse_features.npy`

#### 3.3.2 Feature Interpretability Audit

For the top 20 most active features (by total activation across all DWAs):

```python
for feature_idx in top_20_features:
    activations = features[:, feature_idx]
    top_dwas = np.argsort(activations)[-5:][::-1]
    print(f"Feature {feature_idx}:")
    for dwa_idx in top_dwas:
        print(f"  {dwa_titles[dwa_idx]}: {activations[dwa_idx]:.3f}")
```

**Pass criteria:**
- ≥10/20 features show semantic coherence (e.g., Feature N clusters "Python", "Java", "C++")
- <10/20 coherent features indicates bag-of-words failure (flag but continue)

**Fail indicators:**
- Features activate on function words ("the", "and", "of")
- Features map 1:1 to specific tokens rather than concepts

#### 3.3.3 Deliverables

- `outputs/phase_i_dwa/feature_interpretability.txt` — Human-readable feature audit
- `outputs/phase_i_dwa/interpretability_score.json` — { "coherent_features": N, "total_inspected": 20 }

---

### 3.4 Phase D: SAE Validation

**Objective:** Test whether sparse features outperform raw binary overlap.

#### 3.4.1 Occupation-Level Feature Aggregation

Aggregate DWA-level features to occupation level using the occupation-activity weight matrix:

```python
# W: (J × 2087) occupation-activity weights
# F: (2087 × 16384) sparse DWA features
# W_sparse: (J × 16384) occupation-level features

W_sparse = W @ F
W_sparse = W_sparse / W_sparse.sum(axis=1, keepdims=True)  # Normalize to distribution
```

#### 3.4.2 SAE Jaccard Overlap

Binarize sparse occupation features and compute Jaccard:

```python
B_sparse = (W_sparse > threshold).astype(float)
intersection = B_sparse @ B_sparse.T
row_sums = B_sparse.sum(axis=1, keepdims=True)
union = row_sums + row_sums.T - intersection
sae_overlap = intersection / union
```

**Threshold selection:** Use median non-zero activation value, or tune to match average L₀ from training.

#### 3.4.3 Validation Regression

```
WageComovement_{i,j} = α + β · SAE_Overlap_{i,j} + ε_{i,j}
```

#### 3.4.4 Comparative Analysis

| Metric | Binary Baseline | SAE | Δ |
|--------|-----------------|-----|---|
| β | ? | ? | ? |
| SE | ? | ? | ? |
| t-stat | ? | ? | ? |
| p-value | ? | ? | ? |
| R² | ? | ? | ? |

**Success criteria:**

| Criterion | Threshold | Pass/Fail |
|-----------|-----------|-----------|
| SAE β > 0 | p < 0.05 | Required |
| SAE β > Binary β | Δ > 20% | Required for "SAE adds value" |
| SAE R² > Binary R² | Δ > 0.05% | Required for "SAE adds value" |
| Features interpretable | ≥10/20 coherent | Required for theoretical interest |

#### 3.4.5 Deliverables

- `outputs/phase_i_dwa/baseline_vs_sae_comparison.json` — Full comparison table
- `outputs/phase_i_dwa/sae_validation_verdict.md` — Narrative interpretation

---

## 4. File Deliverables Summary

### 4.1 Scripts

| Script | Purpose | Priority |
|--------|---------|----------|
| `src/task_space/baseline.py` | Raw Binary Overlap computation and validation | P1 |
| `src/task_space/train_sae.py` | SAE training loop | P2 |
| `src/task_space/inspect_sae.py` | Feature interpretability audit | P2 |
| `src/task_space/validate_sae.py` | SAE validation and comparison | P2 |

### 4.2 Data Outputs

| File | Contents |
|------|----------|
| `outputs/phase_i_dwa/dwa_embeddings.npy` | Cached MPNet embeddings (2087 × 768) |
| `outputs/phase_i_dwa/binary_overlap.npy` | Raw binary overlap matrix (J × J) |
| `outputs/phase_i_dwa/dwa_sparse_features.npy` | SAE sparse features (2087 × 16384) |
| `outputs/phase_i_dwa/sae_overlap.npy` | SAE-based overlap matrix (J × J) |
| `models/sae_v1.pt` | Trained SAE checkpoint |

### 4.3 Analysis Outputs

| File | Contents |
|------|----------|
| `outputs/phase_i_dwa/baseline_results.json` | Binary baseline regression results |
| `outputs/phase_i_dwa/sae_training_log.json` | Training metrics |
| `outputs/phase_i_dwa/feature_interpretability.txt` | Feature audit (human-readable) |
| `outputs/phase_i_dwa/baseline_vs_sae_comparison.json` | Comparative analysis |
| `outputs/phase_i_dwa/sae_validation_verdict.md` | Final verdict with interpretation |

---

## 5. Interpretation Guide

### 5.1 Outcome Matrix

| Binary | SAE | Features | Interpretation | Next Step |
|--------|-----|----------|----------------|-----------|
| FAIL | FAIL | — | O*NET data doesn't predict wage comovement | Investigate alternative validation targets |
| **PASS** | FAIL | — | Binary structure works; SAE adds noise | Abandon SAE, proceed with binary overlap |
| FAIL | **PASS** | Coherent | SAE captures economic structure that binary misses | **WIN CONDITION** — Proceed to Phase II with SAE |
| PASS | **PASS (better)** | Coherent | SAE adds value beyond counting | **WIN CONDITION** — Proceed to Phase II with SAE |
| PASS | PASS (better) | Incoherent | SAE works but features are bag-of-words | Proceed with caution; theoretical story weakened |

### 5.2 What Constitutes "Better"?

SAE is considered "better" if:
- β_SAE > 1.2 × β_Binary (20% improvement in effect size)
- R²_SAE > R²_Binary + 0.0005 (0.05 percentage point improvement)

Both conditions must be met.

---

## 6. Version Control

After completing all phases:

1. Increment version to **v0.5.0**
2. Update `README.md` with validation status
3. Update `CLAUDE.md` with lessons learned
4. If SAE passes: Document as "Sparsity Hypothesis validated"
5. If SAE fails: Document failure mode and recommended pivots

---

## 7. Timeline Estimates

| Phase | Estimated Duration |
|-------|-------------------|
| Phase A (Binary Baseline) | 30-60 minutes |
| Phase B (SAE Training) | 15-30 minutes (CPU) |
| Phase C (Feature Inspection) | 30 minutes |
| Phase D (SAE Validation) | 30-60 minutes |
| Documentation | 30 minutes |
| **Total** | **2.5-4 hours** |

---

## 8. Appendix: SAE Implementation Reference

### 8.1 Minimal SAE Architecture

```python
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=16384):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        return torch.relu(self.encoder(x))
    
    def decode(self, f):
        return self.decoder(f)
    
    def forward(self, x):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

def sae_loss(x, x_hat, f, lambda_l1=0.005):
    reconstruction_loss = torch.mean((x - x_hat) ** 2)
    sparsity_loss = torch.mean(torch.abs(f))
    return reconstruction_loss + lambda_l1 * sparsity_loss
```

### 8.2 Training Loop Skeleton

```python
def train_sae(embeddings, epochs=500, lr=1e-3, lambda_l1=0.005, noise_std=0.01):
    model = SparseAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = torch.FloatTensor(embeddings)
    
    for epoch in range(epochs):
        # Add noise for regularization
        noisy_input = dataset + torch.randn_like(dataset) * noise_std
        
        x_hat, f = model(noisy_input)
        loss = sae_loss(dataset, x_hat, f, lambda_l1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Monitor sparsity
        l0 = (f > 0.01).float().sum(dim=1).mean().item()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f}, L0={l0:.1f}")
    
    return model
```

---

**END OF SPECIFICATION**