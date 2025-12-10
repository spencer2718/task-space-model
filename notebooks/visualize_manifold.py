import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from task_space.manifold import SyntheticManifold
from task_space.dynamics import DynamicsEngine


def ensure_outputs_dir():
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    print("Initializing Synthetic Task Space...")
    # 1. Setup Manifold (The Territory)
    # Using 1000 tasks creates a nice dense cloud.
    # 15 centers simulates 15 distinct 'Occupation Types' (e.g. Coding, Manual, Care).
    manifold = SyntheticManifold()
    manifold.load_data(n_samples=1000, n_features=50, n_clusters=15)
    adj_matrix = manifold.build_graph(k=10)

    # 2. Setup Dynamics (The Physics)
    engine = DynamicsEngine(decay_rate=0.2)
    A_t = np.zeros(manifold.n_tasks)
    C_t = np.ones(manifold.n_tasks)

    # 3. Apply a Shock
    # We target "Cluster 5" specifically to see if the geometry preserves locality.
    # Ref: Definition 3.4 (Locality of Operators)
    print("Applying technological shock to Cluster 5...")

    # We cheat slightly and find points belonging to cluster 5 using the blob generation logic
    # (In the real app, we'd filter by task ID)
    labels = manifold.task_ids  # These are just strings in the current implementation
    # For synthetic viz, we just shock a slice of the array which corresponds to a blob
    # since make_blobs returns grouped data usually.
    target_indices = range(300, 360)

    shock = engine.create_shock_vector(manifold.n_tasks, target_indices, magnitude=2.0)

    # Evolve 5 times to let the shock diffuse
    for _ in range(5):
        A_t, C_t = engine.evolve(A_t, C_t, adj_matrix, shock)

    # 4. Dimensionality Reduction (The Camera)
    print("Computing PCA projections...")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(manifold.task_vectors)

    output_dir = ensure_outputs_dir()

    # 5. Plot 2D
    print("Generating 2D Plot...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=A_t, cmap='inferno', alpha=0.6, s=15)
    plt.colorbar(scatter, label='Displacement Intensity $A_t$')
    plt.title('Task Space Projection (2D PCA)\nColor represents Automation Exposure')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, alpha=0.3)

    save_path_2d = os.path.join(output_dir, 'manifold_2d.png')
    plt.savefig(save_path_2d, dpi=150)
    print(f"Saved: {save_path_2d}")

    # 6. Plot 3D
    print("Generating 3D Plot...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Use the same color map (Inferno) - hot colors = high automation
    p = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=A_t, cmap='inferno', alpha=0.6, s=15)

    fig.colorbar(p, label='Displacement Intensity $A_t$')
    ax.set_title('Task Space Manifold (3D PCA)')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')

    save_path_3d = os.path.join(output_dir, 'manifold_3d.png')
    plt.savefig(save_path_3d, dpi=150)
    print(f"Saved: {save_path_3d}")


if __name__ == "__main__":
    main()
