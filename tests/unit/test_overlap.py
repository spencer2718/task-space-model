import numpy as np
import pytest
from task_space.similarity.overlap import (
    compute_jaccard_overlap,
    compute_kernel_overlap,
    compute_normalized_overlap,
)


class TestJaccardOverlap:
    def test_identical_occupations(self):
        """Identical occupations have Jaccard = 1."""
        occ = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ])
        J = compute_jaccard_overlap(occ)
        assert J[0, 1] == pytest.approx(1.0)
        assert J[1, 0] == pytest.approx(1.0)

    def test_disjoint_occupations(self):
        """Disjoint occupations have Jaccard = 0."""
        occ = np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ])
        J = compute_jaccard_overlap(occ)
        assert J[0, 1] == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partial overlap: |A & B| / |A | B|."""
        occ = np.array([
            [1, 1, 1, 0],  # A = {0, 1, 2}
            [0, 1, 1, 1],  # B = {1, 2, 3}
        ])
        # Intersection = {1, 2} = 2, Union = {0, 1, 2, 3} = 4
        J = compute_jaccard_overlap(occ)
        assert J[0, 1] == pytest.approx(2.0 / 4.0)

    def test_symmetric(self):
        """Jaccard matrix is symmetric."""
        np.random.seed(42)
        occ = np.random.rand(10, 20) > 0.5
        J = compute_jaccard_overlap(occ.astype(float))
        np.testing.assert_array_almost_equal(J, J.T)


class TestKernelOverlap:
    def test_identity_kernel(self):
        """With identity kernel, overlap = dot product."""
        occ = np.array([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ])
        K = np.eye(3)
        overlap = compute_kernel_overlap(occ, K)

        # rho_0 @ I @ rho_1 = rho_0 . rho_1 = 0.5*0 + 0.5*0.5 + 0*0.5 = 0.25
        assert overlap[0, 1] == pytest.approx(0.25)

    def test_symmetric(self):
        """Kernel overlap is symmetric."""
        np.random.seed(42)
        occ = np.random.rand(5, 10)
        occ = occ / occ.sum(axis=1, keepdims=True)
        K = np.random.rand(10, 10)
        K = (K + K.T) / 2  # Make symmetric

        overlap = compute_kernel_overlap(occ, K)
        np.testing.assert_array_almost_equal(overlap, overlap.T)


class TestNormalizedOverlap:
    def test_self_overlap_is_one(self):
        """Normalized self-overlap is always 1."""
        np.random.seed(42)
        occ = np.random.rand(5, 10)
        occ = occ / occ.sum(axis=1, keepdims=True)
        K = np.random.rand(10, 10)
        K = (K + K.T) / 2

        norm_overlap = compute_normalized_overlap(occ, K)
        np.testing.assert_array_almost_equal(np.diag(norm_overlap), np.ones(5))

    def test_bounded_zero_one(self):
        """Normalized overlap is in [0, 1] for positive kernel."""
        np.random.seed(42)
        occ = np.random.rand(5, 10)
        occ = occ / occ.sum(axis=1, keepdims=True)

        # Create positive kernel
        dist = np.random.rand(10, 10)
        dist = (dist + dist.T) / 2
        K = np.exp(-dist)

        norm_overlap = compute_normalized_overlap(occ, K)
        assert np.all(norm_overlap >= -1e-10)
        assert np.all(norm_overlap <= 1 + 1e-10)

    def test_controls_concentration(self):
        """Normalized overlap reduces effect of concentration."""
        # Specialist: concentrated on one activity
        # Generalist: spread across all activities
        specialist = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        generalist = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        occ = np.vstack([specialist, generalist])

        K = np.eye(5)  # Identity kernel

        raw = compute_kernel_overlap(occ, K)
        norm = compute_normalized_overlap(occ, K)

        # Raw: specialist self-overlap (1.0) > generalist self-overlap (0.2)
        assert raw[0, 0] > raw[1, 1]

        # Normalized: both self-overlaps are 1.0
        assert norm[0, 0] == pytest.approx(1.0)
        assert norm[1, 1] == pytest.approx(1.0)
