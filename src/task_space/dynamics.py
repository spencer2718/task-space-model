import numpy as np
from scipy import sparse


class DynamicsEngine:
    """
    Implements the displacement dynamics from Definition 1.6.

    The displacement field A_t evolves according to:
        dA_t/dt = K_d[I_t] + epsilon_t

    where K_d is the diffusion operator (Definition 1.5) with exponential kernel:
        k(d(x,y); sigma) = exp(-d(x,y) / sigma)

    In discrete time with adjacency matrix approximation:
        A_{t+1} = A_t + K @ I_t

    where K is the row-normalized adjacency with exponential distance weighting.
    """

    def __init__(self, sigma=1.0):
        """
        Args:
            sigma: Diffusion length scale. Controls how far shocks travel.
                   Small sigma = localized shocks; large sigma = broad diffusion.
        """
        self.sigma = sigma

    def create_shock_vector(self, n_tasks, target_indices, magnitude=1.0):
        """
        Creates an exogenous investment input I_t (Definition 1.4).
        Represents R&D/innovation intensity at specific tasks.
        """
        shock = np.zeros(n_tasks)
        shock[target_indices] = magnitude
        return shock

    def build_diffusion_kernel(self, task_vectors):
        """
        Constructs the diffusion operator K_d using the exponential kernel.

        K[i,j] = exp(-d(i,j) / sigma) where d is Euclidean distance.
        Row-normalized so each row sums to 1.

        This implements Definition 1.5 with kernel from Equation (1).
        """
        from scipy.spatial.distance import cdist

        # Compute pairwise Euclidean distances
        distances = cdist(task_vectors, task_vectors, metric='euclidean')

        # Apply exponential kernel: k(d) = exp(-d/sigma)
        kernel = np.exp(-distances / self.sigma)

        # Row-normalize to create a proper diffusion operator
        row_sums = kernel.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        kernel_normalized = kernel / row_sums

        return kernel_normalized

    def apply_diffusion_operator(self, kernel, shock_vector):
        """
        Apply the diffusion operator K_d to a shock field I_t.

        Implements: [K_d I](x) = integral of k(d(x,y)) * I(y) dmu(y)
        In discrete form: K @ I
        """
        return kernel @ shock_vector

    def evolve(self, current_A, current_C, kernel, shock_vector):
        """
        Updates the technological state fields (A_t, C_t).

        Implements Definition 1.6 (Displacement Dynamics):
            dA_t/dt = K_d[I_t]
            => A_{t+1} = A_t + K_d[I_t]  (discrete time)

        With monotonicity constraint: dA/dt >= 0 (Equation 2b).

        Args:
            current_A: Current displacement field A_t
            current_C: Current complementarity field C_t
            kernel: Diffusion operator K_d (precomputed)
            shock_vector: Innovation input I_t

        Returns:
            (next_A, next_C): Updated technological state
        """
        # Apply diffusion operator to shock: K_d[I_t]
        diffused_shock = self.apply_diffusion_operator(kernel, shock_vector)

        # Update A: linear diffusion (Definition 1.6, Equation 2)
        # dA/dt = K_d[I_t], so A_{t+1} = A_t + K_d[I_t]
        delta_A = diffused_shock

        # Enforce monotonicity constraint (Equation 2b): dA/dt >= 0
        delta_A = np.maximum(delta_A, 0)

        next_A = current_A + delta_A

        # Update C: complementarity grows with diffused innovation
        # (reduced-form specification; paper focuses on A dynamics)
        next_C = current_C + (0.1 * diffused_shock)

        return next_A, next_C
