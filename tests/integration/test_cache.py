import pytest


def test_cache_usage(mocker):
    """
    Cache hit should NOT call computation function.

    Uses mocking instead of wall-clock timing (avoids flaky CI).
    """
    from task_space.data import artifacts
    from task_space.data.artifacts import get_embeddings, clear_cache

    # Clear cache for this specific test to ensure clean state
    clear_cache("embeddings")

    # Spy on the internal computation function
    spy = mocker.spy(artifacts, '_compute_embeddings_impl')

    texts = ["test sentence for cache validation", "another unique test sentence"]

    # First call — should compute
    _ = get_embeddings(texts, force_recompute=True)
    initial_count = spy.call_count
    assert initial_count == 1, "First call should compute"

    # Second call — must be cache hit
    _ = get_embeddings(texts)
    assert spy.call_count == 1, "Second call should use cache, not compute"


def test_cache_invalidation_on_different_input():
    """Different inputs should compute separately."""
    from task_space.data.artifacts import get_embeddings

    emb1 = get_embeddings(["unique text one for test"])
    emb2 = get_embeddings(["unique text two for test"])

    # Different inputs → different embeddings
    assert emb1.shape == emb2.shape
    assert not (emb1 == emb2).all(), "Different text should give different embeddings"


def test_force_recompute_bypasses_cache(mocker):
    """force_recompute=True should always compute."""
    from task_space.data import artifacts
    from task_space.data.artifacts import get_embeddings

    spy = mocker.spy(artifacts, '_compute_embeddings_impl')

    texts = ["force recompute test sentence"]

    # First call
    _ = get_embeddings(texts, force_recompute=True)
    assert spy.call_count == 1

    # Second call with force_recompute — should compute again
    _ = get_embeddings(texts, force_recompute=True)
    assert spy.call_count == 2, "force_recompute should bypass cache"
