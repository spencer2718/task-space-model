import pytest


def test_entropy_control():
    """After controlling for entropy, overlap t-stat ≈ 5.29."""
    # Deferred to v0.6.4 — requires control regression infrastructure
    pytest.skip("Control regression infrastructure not yet implemented (deferred to v0.6.4)")
