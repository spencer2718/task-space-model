import numpy as np
import pytest
from task_space.validation.regression import (
    compute_clustered_se,
    simple_regression,
    RegressionResult,
)


class TestClusteredSE:
    def test_reduces_to_ols_without_clustering(self):
        """With n clusters of size 1, clustered SE ~ OLS SE."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3 * x + np.random.randn(n)

        X = np.column_stack([np.ones(n), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        # Each observation is its own cluster
        cluster_ids = np.arange(n)
        se_clustered = compute_clustered_se(X, residuals, cluster_ids)

        # OLS SE
        ss_res = np.sum(residuals**2)
        var_resid = ss_res / (n - 2)
        XtX_inv = np.linalg.inv(X.T @ X)
        se_ols = np.sqrt(var_resid * np.diag(XtX_inv))

        # Should be similar (not exact due to small-sample adjustments)
        np.testing.assert_array_almost_equal(se_clustered, se_ols, decimal=1)

    def test_clustering_increases_se(self):
        """Clustered SE >= OLS SE when observations are correlated within clusters."""
        np.random.seed(42)
        n_clusters = 20
        cluster_size = 10
        n = n_clusters * cluster_size

        # Generate correlated data within clusters
        x = []
        y = []
        for c in range(n_clusters):
            cluster_effect = np.random.randn()
            x_c = np.random.randn(cluster_size)
            y_c = 2 + 3 * x_c + cluster_effect + 0.1 * np.random.randn(cluster_size)
            x.extend(x_c)
            y.extend(y_c)

        x = np.array(x)
        y = np.array(y)
        cluster_ids = np.repeat(np.arange(n_clusters), cluster_size)

        X = np.column_stack([np.ones(n), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        se_clustered = compute_clustered_se(X, residuals, cluster_ids)

        # OLS SE (ignoring clustering)
        ss_res = np.sum(residuals**2)
        var_resid = ss_res / (n - 2)
        XtX_inv = np.linalg.inv(X.T @ X)
        se_ols = np.sqrt(var_resid * np.diag(XtX_inv))

        # Clustered SE should be larger due to within-cluster correlation
        assert se_clustered[1] >= se_ols[1] * 0.9  # Allow some tolerance


class TestSimpleRegression:
    def test_perfect_fit(self):
        """R^2 = 1 for perfectly linear relationship."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 + 3 * x  # Perfect linear

        result = simple_regression(x, y)

        assert result.r2 == pytest.approx(1.0)
        assert result.beta[0] == pytest.approx(2.0)
        assert result.beta[1] == pytest.approx(3.0)

    def test_no_relationship(self):
        """Low R^2 for random data."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        result = simple_regression(x, y)

        # R^2 should be close to 0
        assert result.r2 < 0.1

    def test_negative_relationship(self):
        """Negative beta for inverse relationship."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 10 - 2 * x

        result = simple_regression(x, y)

        assert result.beta[1] < 0
        assert result.t[1] < 0

    def test_returns_correct_shape(self):
        """Result has correct array shapes."""
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50)

        result = simple_regression(x, y)

        assert len(result.beta) == 2
        assert len(result.se) == 2
        assert len(result.t) == 2
        assert len(result.p) == 2
        assert isinstance(result.r2, float)
        assert isinstance(result.n_pairs, int)
