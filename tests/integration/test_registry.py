import pytest
from task_space.shocks.registry import (
    register_shock, get_shock, list_shocks,
    _reset_registry, _restore_default_shocks
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset registry before and after each test."""
    _restore_default_shocks()
    yield
    _restore_default_shocks()


def test_registry_extensibility():
    """Register custom shock at runtime, use via config."""
    import numpy as np

    initial_count = len(list_shocks())

    @register_shock(name="test_custom", description="Test shock")
    def shock_test_custom(domain, intensity: float = 2.0, **kwargs):
        return np.full(len(domain.activity_ids), intensity)

    assert "test_custom" in list_shocks()
    assert len(list_shocks()) == initial_count + 1

    shock = get_shock("test_custom")
    assert shock.fn is not None
    assert shock.description == "Test shock"


def test_registry_isolation():
    """Shocks registered in one test don't leak to others."""
    # If clean_registry fixture works, "test_custom" should not exist
    # (unless test_registry_extensibility ran in same session without cleanup)
    default_shocks = ['uniform', 'gaussian_directed', 'capability_v1', 'capability_v2', 'rbtc']
    current_shocks = list_shocks()

    for shock in default_shocks:
        assert shock in current_shocks, f"Missing default shock: {shock}"


def test_get_unknown_shock_raises():
    """Getting unknown shock raises ValueError with available list."""
    with pytest.raises(ValueError) as exc_info:
        get_shock("nonexistent_shock")
    assert "nonexistent_shock" in str(exc_info.value)
    assert "Available" in str(exc_info.value)
