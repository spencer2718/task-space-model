import numpy as np
from scipy import sparse


class DynamicsEngine:
    """
    Implements the differential inclusions as discrete time-steps.
    Ref: Section 3.4 of the source text.
    """
    def __init__(self, decay_rate=0.1):
        self.decay_rate = decay_rate

    def create_shock_vector(self, n_tasks, target_indices, magnitude=1.0):
        """
        Creates an exogenous investment input I_t (Definition 3.7).
        In the synthetic case, this targets a specific 'cluster' of tasks.
        """
        shock = np.zeros(n_tasks)
        shock[target_indices] = magnitude
        return shock

    def apply_operator(self, adj_matrix, shock_vector):
        """
        Models the tool T_z as a local operator (Definition 3.4).
        Physically, this diffuses the 'capability shock' to neighboring tasks
        defined by the metric space d.
        """
        # 1. Normalize adjacency to behave like a diffusion operator
        # (Row-normalize: influence spreads to neighbors)
        row_sums = np.array(adj_matrix.sum(axis=1)).flatten()
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        D_inv = sparse.diags(1.0 / row_sums)
        normalized_adj = D_inv.dot(adj_matrix)

        # 2. Apply the operator (Spread the shock)
        # Direct impact + 1-hop neighbor impact
        # The 'Locality' property from Def 3.4
        diffused_impact = shock_vector + (self.decay_rate * normalized_adj.dot(shock_vector))

        return diffused_impact

    def evolve(self, current_A, current_C, adj_matrix, shock_vector):
        """
        Updates the state fields A_t and C_t based on the shock.
        Maps to Aggregation Classes (Definition 3.5).
        """
        # Calculate the local impact of the new tool
        impact_field = self.apply_operator(adj_matrix, shock_vector)

        # Update A (Displacement): Aggregator is typically MAX.
        # Logic: If a new tool automates a task better than the old one,
        # the 'automation level' rises. It acts as a frontier.
        # Ref: Theorem 3.2 (Monotone frontier expansion)
        next_A = np.maximum(current_A, impact_field)

        # Update C (Complementarity): Aggregator is typically MULTIPLICATIVE or ADDITIVE.
        # Logic: Tools stack. A new tool adds to the worker's leverage.
        # We use a simple additive model here for stability, or (1 + impact) multiplier.
        next_C = current_C * (1.0 + (0.1 * impact_field))

        return next_A, next_C
