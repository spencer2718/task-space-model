"""Unit tests for Wasserstein distance computation."""
import numpy as np
import pytest

from task_space.similarity.wasserstein import (
    compute_wasserstein_distances,
    wasserstein_to_similarity,
)


class TestWassersteinDistance:
    """Tests for compute_wasserstein_distances."""

    @pytest.fixture
    def simple_problem(self):
        """3 occupations, 5 activities."""
        # Occupation measures (rows sum to 1)
        occ_measures = np.array([
            [0.5, 0.5, 0.0, 0.0, 0.0],  # Occ 0: activities 0,1
            [0.0, 0.0, 0.5, 0.5, 0.0],  # Occ 1: activities 2,3
            [0.25, 0.25, 0.25, 0.25, 0.0],  # Occ 2: activities 0,1,2,3
        ])
        # Ground metric (identity-like: nearby activities are close)
        ground = np.array([
            [0.0, 0.1, 0.5, 0.6, 1.0],
            [0.1, 0.0, 0.4, 0.5, 0.9],
            [0.5, 0.4, 0.0, 0.1, 0.5],
            [0.6, 0.5, 0.1, 0.0, 0.4],
            [1.0, 0.9, 0.5, 0.4, 0.0],
        ])
        return occ_measures, ground

    def test_self_distance_zero(self, simple_problem):
        """Distance from occupation to itself should be 0."""
        occ_measures, ground = simple_problem
        result = compute_wasserstein_distances(occ_measures, ground, verbose=False)
        np.testing.assert_array_almost_equal(np.diag(result.distance_matrix), 0)

    def test_symmetry(self, simple_problem):
        """Distance matrix should be symmetric."""
        occ_measures, ground = simple_problem
        result = compute_wasserstein_distances(occ_measures, ground, verbose=False)
        np.testing.assert_array_almost_equal(
            result.distance_matrix,
            result.distance_matrix.T
        )

    def test_triangle_inequality(self, simple_problem):
        """Wasserstein satisfies triangle inequality."""
        occ_measures, ground = simple_problem
        result = compute_wasserstein_distances(occ_measures, ground, verbose=False)
        D = result.distance_matrix

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    assert D[i, k] <= D[i, j] + D[j, k] + 1e-10

    def test_similar_occupations_closer(self, simple_problem):
        """Occupations with overlapping activities should be closer."""
        occ_measures, ground = simple_problem
        result = compute_wasserstein_distances(occ_measures, ground, verbose=False)
        D = result.distance_matrix

        # Occ 2 overlaps with both 0 and 1, so should be closer to each
        # than 0 is to 1 (which have disjoint support)
        assert D[0, 2] < D[0, 1]
        assert D[1, 2] < D[0, 1]

    def test_result_metadata(self, simple_problem):
        """Result contains correct metadata."""
        occ_measures, ground = simple_problem
        result = compute_wasserstein_distances(occ_measures, ground, verbose=False)

        assert result.n_occupations == 3
        assert result.n_activities == 5
        assert result.computation_time_seconds > 0
        assert len(result.assumptions) > 0

    def test_ground_metric_mismatch_raises(self):
        """Mismatched ground metric shape should raise error."""
        occ_measures = np.array([[0.5, 0.5], [0.3, 0.7]])
        ground = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])  # 3x3, not 2x2

        with pytest.raises(ValueError, match="Ground distance shape"):
            compute_wasserstein_distances(occ_measures, ground, verbose=False)

    def test_identical_distributions_zero_distance(self):
        """Identical distributions should have zero distance."""
        occ_measures = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],  # Same as first
            [0.5, 0.5, 0.0, 0.0],
        ])
        ground = np.array([
            [0.0, 0.5, 0.5, 1.0],
            [0.5, 0.0, 1.0, 0.5],
            [0.5, 1.0, 0.0, 0.5],
            [1.0, 0.5, 0.5, 0.0],
        ])
        result = compute_wasserstein_distances(occ_measures, ground, verbose=False)

        # Distance between occ 0 and occ 1 should be 0
        assert result.distance_matrix[0, 1] < 1e-10


class TestWassersteinSimilarity:
    """Tests for wasserstein_to_similarity."""

    def test_self_similarity_one(self):
        """Self-similarity should be 1."""
        D = np.array([
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 0.7],
            [1.0, 0.7, 0.0],
        ])
        sim = wasserstein_to_similarity(D)
        np.testing.assert_array_almost_equal(np.diag(sim), 1.0)

    def test_similarity_decreases_with_distance(self):
        """Higher distance should mean lower similarity."""
        D = np.array([
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 0.7],
            [1.0, 0.7, 0.0],
        ])
        sim = wasserstein_to_similarity(D)

        # sim[0,1] > sim[0,2] because D[0,1] < D[0,2]
        assert sim[0, 1] > sim[0, 2]

    def test_similarity_symmetry(self):
        """Similarity matrix should be symmetric."""
        D = np.array([
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 0.7],
            [1.0, 0.7, 0.0],
        ])
        sim = wasserstein_to_similarity(D)
        np.testing.assert_array_almost_equal(sim, sim.T)

    def test_all_zero_distances(self):
        """All identical distributions should give all-ones similarity."""
        D = np.zeros((3, 3))
        sim = wasserstein_to_similarity(D)
        np.testing.assert_array_almost_equal(sim, np.ones((3, 3)))

    def test_similarity_bounded(self):
        """Similarity values should be in (0, 1]."""
        D = np.array([
            [0.0, 0.5, 2.0],
            [0.5, 0.0, 1.5],
            [2.0, 1.5, 0.0],
        ])
        sim = wasserstein_to_similarity(D)

        assert np.all(sim > 0)
        assert np.all(sim <= 1.0)
