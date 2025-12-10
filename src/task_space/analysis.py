import numpy as np
import pandas as pd


class Nowcaster:
    """
    Computes economic observables from the graph state.
    Ref: Section 3.3 (Exposures) and Section 6 (Results).
    """
    def __init__(self, n_tasks, n_occupations):
        self.n_tasks = n_tasks
        self.n_occupations = n_occupations

        # Matrix of occupation weights (n_occupations, n_tasks)
        # Each row is rho_j, summing to 1.
        self.occupation_weights = None

    def generate_synthetic_occupations(self, task_clusters):
        """
        Creates dummy occupation profiles for testing.
        Assumes each occupation specializes in one 'blob' of tasks.
        """
        print("Generating synthetic occupation weights...")
        weights = np.zeros((self.n_occupations, self.n_tasks))

        # If we have cluster labels from Phase 1, assign occupations to them
        # For now, we create random sparse weights to mimic specialization
        rng = np.random.default_rng(42)

        for i in range(self.n_occupations):
            # Pick a random center and assign mass using a softmax (concentration)
            center = rng.integers(0, self.n_tasks)
            # Create distance-based weights (Gaussian-ish)
            dists = np.abs(np.arange(self.n_tasks) - center)
            # Wrap around for simplicity in 1D array view, though graph is high-dim
            raw_weights = np.exp(-dists / 50.0)
            weights[i] = raw_weights / raw_weights.sum()

        self.occupation_weights = weights

    def compute_exposures(self, A_t, C_t):
        """
        Calculates D_j (Displacement) and U_j (Augmentation) for all occupations.
        Definition 3.6: D_j(t) = <A_t, rho_j>
        """
        if self.occupation_weights is None:
            raise ValueError("Occupation weights not initialized.")

        # Dot product: (n_occs, n_tasks) @ (n_tasks,) -> (n_occs,)
        # This is the discrete integration over the measure.
        D_j = self.occupation_weights @ A_t

        # U_j uses log C_t per Definition 3.6
        U_j = self.occupation_weights @ np.log(C_t)

        return pd.DataFrame({
            "Displacement_D": D_j,
            "Augmentation_U": U_j
        })

    def measure_frontier(self, A_t, threshold=0.5):
        """
        Calculates the share of tasks inside the automation frontier F(t).
        Ref: Definition 3.10
        """
        # Count tasks where displacement intensity exceeds threshold
        automated_count = np.sum(A_t >= threshold)
        share = automated_count / self.n_tasks
        return share

    def track_churn(self, current_exposures, prev_exposures):
        """
        Calculates the delta in exposure (Top-10 changes plot).
        Ref: Section 6 (Results).
        """
        if prev_exposures is None:
            return None

        delta = current_exposures - prev_exposures
        return delta.sort_values("Displacement_D", ascending=False)
