from unittest.mock import AsyncMock, call

import pytest

from src.domain.models import Confidence, KnowledgeArticle, SearchResult
from src.domain.services.rag_service import RAGService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _article(article_id: str = "a1", category: str = "general") -> KnowledgeArticle:
    return KnowledgeArticle(
        id=article_id,
        category=category,
        question="What is the reset procedure?",
        answer="Press the reset button for 5 seconds.",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedding() -> AsyncMock:
    emb = AsyncMock()
    emb.embed = AsyncMock(return_value=[0.0, 1.0, 0.0])
    emb.embed_batch = AsyncMock(
        side_effect=lambda texts: [[float(i)] * 2 for i in range(len(texts))]
    )
    return emb


@pytest.fixture
def vector_store() -> AsyncMock:
    vs = AsyncMock()
    vs.clear = AsyncMock()
    vs.upsert = AsyncMock()
    vs.search = AsyncMock(return_value=[])
    return vs


@pytest.fixture
def llm() -> AsyncMock:
    llm_mock = AsyncMock()
    llm_mock.generate = AsyncMock(return_value="LLM synthesized answer.")
    return llm_mock


@pytest.fixture
def rag(embedding: AsyncMock, vector_store: AsyncMock, llm: AsyncMock) -> RAGService:
    return RAGService(embedding, vector_store, llm, top_k=5)


# ---------------------------------------------------------------------------
# answer() - no results path
# ---------------------------------------------------------------------------


async def test_answer_no_results_returns_fallback_without_llm(
    rag: RAGService,
    embedding: AsyncMock,
    vector_store: AsyncMock,
    llm: AsyncMock,
) -> None:
    out = await rag.answer("any question?")

    embedding.embed.assert_awaited_once_with("any question?")
    vector_store.search.assert_awaited_once()
    llm.generate.assert_not_called()

    assert out.answer == "No matching knowledge base content was found for this question."
    assert out.sources == []
    assert out.confidence is Confidence.NONE


# ---------------------------------------------------------------------------
# answer() - confidence thresholds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("best_score", "expected_confidence"),
    [
        (0.82, Confidence.HIGH),
        (0.65, Confidence.MEDIUM),
        (0.45, Confidence.LOW),
        (0.44, Confidence.NONE),
    ],
)
async def test_answer_confidence_from_best_score(
    rag: RAGService,
    vector_store: AsyncMock,
    llm: AsyncMock,
    best_score: float,
    expected_confidence: Confidence,
) -> None:
    article = _article()
    vector_store.search = AsyncMock(
        return_value=[
            SearchResult(article=article, score=0.1),
            SearchResult(article=article, score=best_score),
        ],
    )

    out = await rag.answer("q")

    assert out.confidence is expected_confidence
    assert out.sources == [article.id, article.id]
    llm.generate.assert_awaited_once()


# ---------------------------------------------------------------------------
# answer() - top_k override
# ---------------------------------------------------------------------------


async def test_answer_passes_default_top_k_to_vector_store(
    rag: RAGService,
    vector_store: AsyncMock,
) -> None:
    await rag.answer("q")

    _args, kwargs = vector_store.search.await_args
    assert kwargs["top_k"] == 5


async def test_answer_passes_top_k_override(
    rag: RAGService,
    vector_store: AsyncMock,
) -> None:
    vector_store.search = AsyncMock(return_value=[])

    await rag.answer("x", top_k=3)

    _args, kwargs = vector_store.search.await_args
    assert kwargs["top_k"] == 3


# ---------------------------------------------------------------------------
# answer() - system-prompt quality add-ons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected_fragment", "present"),
    [
        (0.85, "Retrieval quality:", False),
        (0.45, "Retrieval quality: The strongest matches are only moderately similar", True),
        (0.20, "Retrieval quality: The strongest matches are weakly related", True),
    ],
)
async def test_answer_system_prompt_retrieval_quality_addon(
    rag: RAGService,
    vector_store: AsyncMock,
    llm: AsyncMock,
    score: float,
    expected_fragment: str,
    present: bool,
) -> None:
    article = _article()
    vector_store.search = AsyncMock(
        return_value=[SearchResult(article=article, score=score)],
    )

    await rag.answer("q")

    sys_prompt, _user = llm.generate.await_args[0]
    if present:
        assert expected_fragment in sys_prompt
    else:
        assert expected_fragment not in sys_prompt


# ---------------------------------------------------------------------------
# answer() - user-prompt formatting
# ---------------------------------------------------------------------------


async def test_answer_user_prompt_contains_excerpt_fields(
    rag: RAGService,
    vector_store: AsyncMock,
    llm: AsyncMock,
) -> None:
    article = _article(article_id="id-x", category="billing")
    vector_store.search = AsyncMock(
        return_value=[SearchResult(article=article, score=0.9)],
    )

    await rag.answer("How do I pay?")

    _sys, user_prompt = llm.generate.await_args[0]
    assert "### Excerpt 1 (id=id-x, category=billing)" in user_prompt
    assert "FAQ question: What is the reset procedure?" in user_prompt
    assert "FAQ answer: Press the reset button for 5 seconds." in user_prompt
    assert "User question:\nHow do I pay?" in user_prompt


async def test_answer_sources_order_matches_search_results(
    rag: RAGService,
    vector_store: AsyncMock,
    llm: AsyncMock,
) -> None:
    a1, a2 = _article("first"), _article("second")
    vector_store.search = AsyncMock(
        return_value=[
            SearchResult(article=a1, score=0.9),
            SearchResult(article=a2, score=0.7),
        ],
    )

    out = await rag.answer("q")

    assert out.sources == ["first", "second"]


# ---------------------------------------------------------------------------
# index_knowledge_base()
# ---------------------------------------------------------------------------


async def test_index_knowledge_base_clear_embed_upsert(
    rag: RAGService,
    embedding: AsyncMock,
    vector_store: AsyncMock,
) -> None:
    kb = AsyncMock()
    kb.list_all = AsyncMock(return_value=[_article("1"), _article("2")])

    await rag.index_knowledge_base(kb)

    vector_store.clear.assert_awaited_once()
    kb.list_all.assert_awaited_once()
    expected_text = "What is the reset procedure?\nPress the reset button for 5 seconds."
    embedding.embed_batch.assert_awaited_once_with([expected_text, expected_text])
    vector_store.upsert.assert_has_calls(
        [
            call("1", [0.0, 0.0]),
            call("2", [1.0, 1.0]),
        ],
    )


async def test_index_knowledge_base_empty_skips_embed_and_upsert(
    rag: RAGService,
    embedding: AsyncMock,
    vector_store: AsyncMock,
) -> None:
    kb = AsyncMock()
    kb.list_all = AsyncMock(return_value=[])

    await rag.index_knowledge_base(kb)

    vector_store.clear.assert_awaited_once()
    embedding.embed_batch.assert_not_called()
    vector_store.upsert.assert_not_called()


async def test_index_knowledge_base_clear_first_false_skips_clear(
    rag: RAGService,
    vector_store: AsyncMock,
) -> None:
    kb = AsyncMock()
    kb.list_all = AsyncMock(return_value=[_article()])

    await rag.index_knowledge_base(kb, clear_first=False)

    vector_store.clear.assert_not_called()
