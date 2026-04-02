from unittest.mock import AsyncMock

import pytest

from src.adapters.outbound.vector_store.numpy_in_memory import NumpyInMemoryVectorStore
from src.domain.models import KnowledgeArticle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _article(article_id: str = "a1") -> KnowledgeArticle:
    return KnowledgeArticle(id=article_id, category="c", question="Q", answer="A")


def _kb(*article_ids: str) -> AsyncMock:
    mapping = {aid: _article(aid) for aid in article_ids}
    kb = AsyncMock()
    kb.get_by_id = AsyncMock(side_effect=lambda aid: mapping.get(aid))
    return kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kb() -> AsyncMock:
    return _kb("a1", "a2", "a3")


@pytest.fixture
def store(kb: AsyncMock) -> NumpyInMemoryVectorStore:
    return NumpyInMemoryVectorStore(kb)


# ---------------------------------------------------------------------------
# upsert / search
# ---------------------------------------------------------------------------


async def test_search_empty_store_returns_empty(store: NumpyInMemoryVectorStore) -> None:
    results = await store.search([1.0, 0.0], top_k=5)
    assert results == []


async def test_search_zero_query_vector_returns_empty(store: NumpyInMemoryVectorStore) -> None:
    await store.upsert("a1", [1.0, 0.0])
    results = await store.search([0.0, 0.0], top_k=5)
    assert results == []


async def test_search_returns_exact_match_at_top(store: NumpyInMemoryVectorStore) -> None:
    await store.upsert("a1", [1.0, 0.0])
    await store.upsert("a2", [0.0, 1.0])

    results = await store.search([1.0, 0.0], top_k=2)

    assert results[0].article.id == "a1"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


async def test_search_ranks_by_cosine_similarity(store: NumpyInMemoryVectorStore) -> None:
    await store.upsert("a1", [1.0, 0.0])  # fully aligned with query
    await store.upsert("a2", [0.0, 1.0])  # orthogonal

    results = await store.search([1.0, 0.1], top_k=2)

    ids = [r.article.id for r in results]
    assert ids[0] == "a1"
    assert ids[1] == "a2"


async def test_search_top_k_limits_results(store: NumpyInMemoryVectorStore) -> None:
    for aid in ("a1", "a2", "a3"):
        await store.upsert(aid, [1.0, 0.0])

    results = await store.search([1.0, 0.0], top_k=2)

    assert len(results) == 2


async def test_search_top_k_larger_than_store_returns_all(
    store: NumpyInMemoryVectorStore,
) -> None:
    await store.upsert("a1", [1.0, 0.0])
    await store.upsert("a2", [0.0, 1.0])

    results = await store.search([1.0, 0.0], top_k=100)

    assert len(results) == 2


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_removes_vector(store: NumpyInMemoryVectorStore) -> None:
    await store.upsert("a1", [1.0, 0.0])
    await store.delete("a1")

    results = await store.search([1.0, 0.0], top_k=5)
    assert results == []


async def test_delete_nonexistent_is_silent(store: NumpyInMemoryVectorStore) -> None:
    await store.delete("does_not_exist")


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


async def test_clear_empties_store(store: NumpyInMemoryVectorStore) -> None:
    await store.upsert("a1", [1.0, 0.0])
    await store.upsert("a2", [0.0, 1.0])
    await store.clear()

    results = await store.search([1.0, 0.0], top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# upsert normalisation
# ---------------------------------------------------------------------------


async def test_upsert_normalises_vector(store: NumpyInMemoryVectorStore) -> None:
    await store.upsert("a1", [3.0, 4.0])  # L2 norm = 5 → normalised [0.6, 0.8]

    results = await store.search([3.0, 4.0], top_k=1)

    assert results[0].score == pytest.approx(1.0, abs=1e-5)


async def test_upsert_zero_vector_is_stored_without_error(
    store: NumpyInMemoryVectorStore,
) -> None:
    await store.upsert("a1", [0.0, 0.0])
    results = await store.search([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].score == pytest.approx(0.0, abs=1e-5)
