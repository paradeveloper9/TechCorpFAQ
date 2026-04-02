from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends

from src.adapters.inbound.http.schemas import AskRequest, AskResponse
from src.domain.services.rag_service import RAGService


def create_rag_router(
    *,
    rag_service: Callable[[], RAGService],
) -> APIRouter:
    async def _rag_dep() -> RAGService:
        return rag_service()

    router = APIRouter(tags=["rag"])

    @router.post("/ask", response_model=AskResponse)
    async def answer(
        body: AskRequest,
        rag: Annotated[RAGService, Depends(_rag_dep)],
    ) -> AskResponse:
        result = await rag.answer(body.question, top_k=body.top_k)
        return AskResponse.model_validate(result)

    return router
