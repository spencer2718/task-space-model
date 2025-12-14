import numpy as np
import pytest
from task_space.shocks.propagation import propagate_shock


class TestPropagate:
    def test_shapes(self):
        """Output shapes are correct."""
        n_act, n_occ = 100, 50
        I_t = np.random.rand(n_act)
        K = np.random.rand(n_act, n_act)
        rho = np.random.rand(n_occ, n_act)
        rho = rho / rho.sum(axis=1, keepdims=True)

        result = propagate_shock(I_t, K, rho)

        assert result.I_t.shape == (n_act,)
        assert result.A_t.shape == (n_act,)
        assert result.E.shape == (n_occ,)

    def test_uniform_shock(self):
        """Uniform shock gives exposure proportional to measure mass."""
        n_act, n_occ = 10, 3
        I_t = np.ones(n_act)  # Uniform shock
        K = np.eye(n_act)     # Identity kernel (no propagation)
        rho = np.array([
            [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ])

        result = propagate_shock(I_t, K, rho)

        # With identity kernel and uniform shock, E_j = sum_a rho_j(a) = 1 for all j
        np.testing.assert_array_almost_equal(result.E, [1.0, 1.0, 1.0])

    def test_localized_shock(self):
        """Localized shock affects nearby occupations more."""
        n_act = 5
        I_t = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Shock only on first activity
        K = np.eye(n_act)  # No propagation

        rho = np.array([
            [1.0, 0, 0, 0, 0],    # Occ 0: only uses activity 0
            [0, 1.0, 0, 0, 0],    # Occ 1: only uses activity 1
        ])

        result = propagate_shock(I_t, K, rho)

        assert result.E[0] == pytest.approx(1.0)  # Fully exposed
        assert result.E[1] == pytest.approx(0.0)  # Not exposed

    def test_kernel_propagation(self):
        """Kernel spreads shock to neighboring activities."""
        n_act = 3
        I_t = np.array([1.0, 0.0, 0.0])  # Shock on activity 0

        # Kernel with spillover
        K = np.array([
            [1.0, 0.5, 0.1],
            [0.5, 1.0, 0.5],
            [0.1, 0.5, 1.0],
        ])

        rho = np.array([
            [1.0, 0.0, 0.0],  # Occ 0: only activity 0
            [0.0, 1.0, 0.0],  # Occ 1: only activity 1
            [0.0, 0.0, 1.0],  # Occ 2: only activity 2
        ])

        result = propagate_shock(I_t, K, rho)

        # A_t = K @ I_t = [1.0, 0.5, 0.1]
        np.testing.assert_array_almost_equal(result.A_t, [1.0, 0.5, 0.1])

        # E = rho @ A_t
        assert result.E[0] == pytest.approx(1.0)
        assert result.E[1] == pytest.approx(0.5)
        assert result.E[2] == pytest.approx(0.1)

    def test_metadata_preserved(self):
        """Metadata is correctly stored in result."""
        I_t = np.ones(5)
        K = np.eye(5)
        rho = np.ones((3, 5)) / 5

        result = propagate_shock(I_t, K, rho, shock_name="test_shock")

        assert result.shock_name == "test_shock"


class TestPropagationSpec:
    """Tests from spec section 2.3."""

    def test_propagation_localized_with_kernel(self):
        """Kernel spreads localized shock to nearby activities."""
        n_act = 5
        # Shock on activity 0 only
        I_t = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        # Kernel with distance decay (activity 1 close, activity 4 far)
        K = np.array([
            [1.0, 0.8, 0.5, 0.2, 0.1],
            [0.8, 1.0, 0.6, 0.3, 0.1],
            [0.5, 0.6, 1.0, 0.5, 0.2],
            [0.2, 0.3, 0.5, 1.0, 0.6],
            [0.1, 0.1, 0.2, 0.6, 1.0],
        ])

        # Occupation concentrated on activity 1 (near shock)
        rho_near = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
        # Occupation concentrated on activity 4 (far from shock)
        rho_far = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])

        result_near = propagate_shock(I_t, K, rho_near)
        result_far = propagate_shock(I_t, K, rho_far)

        # Near occupation should have higher exposure
        assert result_near.E[0] > result_far.E[0], "Kernel should spread shock to nearby"

        # Verify exact values: E = ρ @ (K @ I)
        A_t = K @ I_t  # [1.0, 0.8, 0.5, 0.2, 0.1]
        assert result_near.E[0] == pytest.approx(0.8)  # rho_near @ A_t
        assert result_far.E[0] == pytest.approx(0.1)   # rho_far @ A_t

    def test_propagation_no_sigma_parameter(self):
        """Propagate_shock should not accept sigma parameter."""
        import inspect
        sig = inspect.signature(propagate_shock)
        param_names = list(sig.parameters.keys())

        assert 'sigma' not in param_names, "sigma should not be a parameter (K already encodes it)"
