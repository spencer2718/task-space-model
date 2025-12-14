import numpy as np
import pytest
from task_space.similarity.kernel import calibrate_sigma, build_kernel_matrix, check_kernel_discrimination


class TestCalibrateSigma:
    def test_nn_median(self):
        """sigma = median of nearest-neighbor distances."""
        dist = np.array([
            [0.0, 0.1, 0.5],
            [0.1, 0.0, 0.4],
            [0.5, 0.4, 0.0],
        ])
        # NN: [0.1, 0.1, 0.4] -> median = 0.1
        sigma = calibrate_sigma(dist)
        assert sigma == pytest.approx(0.1)

    def test_rejects_global(self):
        """Reject non-NN methods."""
        dist = np.random.rand(5, 5)
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        with pytest.raises(ValueError, match="not supported"):
            calibrate_sigma(dist, method='global_median')


class TestBuildKernelMatrix:
    def test_exponential(self):
        """K_ij = exp(-d_ij / sigma)."""
        dist = np.array([[0.0, 1.0], [1.0, 0.0]])
        K, sigma = build_kernel_matrix(dist, sigma=1.0)

        assert K[0, 0] == pytest.approx(1.0)
        assert K[0, 1] == pytest.approx(np.exp(-1))

    def test_auto_sigma(self):
        """Auto-calibrates when sigma=None."""
        dist = np.random.rand(10, 10)
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)

        K, sigma = build_kernel_matrix(dist, sigma=None)
        assert sigma > 0

    def test_row_normalize_warning(self):
        """Warns when row_normalize=True."""
        dist = np.random.rand(5, 5)
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        with pytest.warns(UserWarning, match="destroy signal"):
            build_kernel_matrix(dist, sigma=0.1, row_normalize=True)


class TestDiscrimination:
    def test_good_discrimination(self):
        """High ratio when sigma is small relative to distances."""
        np.random.seed(42)
        dist = np.random.rand(100, 100)
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)

        ratio, ok = check_kernel_discrimination(dist, sigma=0.1)
        assert ratio > 3.0
        assert ok

    def test_collapsed(self):
        """Low ratio when sigma is too large."""
        np.random.seed(42)
        dist = np.random.rand(100, 100) * 0.5 + 0.5  # distances in [0.5, 1.0]
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)

        ratio, ok = check_kernel_discrimination(dist, sigma=5.0)
        assert ratio < 2.0
        assert not ok
