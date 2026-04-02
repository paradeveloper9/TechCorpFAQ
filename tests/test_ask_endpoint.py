from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.adapters.inbound.http.router import create_rag_router
from src.domain.models import Confidence, RAGResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(rag: AsyncMock) -> FastAPI:
    app = FastAPI()
    app.include_router(create_rag_router(rag_service=lambda: rag), prefix="/api")
    return app


@pytest.fixture
def rag() -> AsyncMock:
    mock = AsyncMock()
    mock.answer = AsyncMock(
        return_value=RAGResponse(
            answer="Default answer.",
            sources=["kb_001"],
            confidence=Confidence.HIGH,
        )
    )
    return mock


@pytest.fixture
async def client(rag: AsyncMock) -> AsyncGenerator[AsyncClient, None]:
    app = _make_app(rag)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_ask_returns_200_with_answer(client: AsyncClient, rag: AsyncMock) -> None:
    response = await client.post("/api/ask", json={"question": "How do I reset my password?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Default answer."
    assert body["sources"] == ["kb_001"]
    assert body["confidence"] == "high"


async def test_ask_passes_question_to_rag_service(client: AsyncClient, rag: AsyncMock) -> None:
    await client.post("/api/ask", json={"question": "What are the office hours?"})

    rag.answer.assert_awaited_once()
    assert rag.answer.await_args.args[0] == "What are the office hours?"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {},
    ],
)
async def test_ask_rejects_invalid_payload(client: AsyncClient, payload: dict[str, object]) -> None:
    response = await client.post("/api/ask", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Confidence values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "confidence", [Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW, Confidence.NONE]
)
async def test_ask_serialises_all_confidence_values(
    rag: AsyncMock,
    confidence: Confidence,
) -> None:
    rag.answer = AsyncMock(return_value=RAGResponse(answer="a", sources=[], confidence=confidence))
    app = _make_app(rag)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.post("/api/ask", json={"question": "q"})

    assert response.status_code == 200
    assert response.json()["confidence"] == confidence.value
