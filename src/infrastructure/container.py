from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.adapters.inbound.http.router import create_rag_router
from src.adapters.outbound.embedding.openrouter import OpenRouterEmbeddingAdapter
from src.adapters.outbound.knowledge_base.json_file import JsonFileKnowledgeBaseAdapter
from src.adapters.outbound.llm.openai import OpenAILLMAdapter
from src.adapters.outbound.vector_store.numpy_in_memory import NumpyInMemoryVectorStore
from src.domain.services.rag_service import RAGService
from src.infrastructure.config import Settings, get_settings


class Container:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._knowledge_base: JsonFileKnowledgeBaseAdapter | None = None
        self._vector_store: NumpyInMemoryVectorStore | None = None
        self._embedding: OpenRouterEmbeddingAdapter | None = None
        self._llm: OpenAILLMAdapter | None = None
        self._rag: RAGService | None = None

    @property
    def settings(self) -> Settings:
        return self._settings

    def knowledge_base(self) -> JsonFileKnowledgeBaseAdapter:
        if self._knowledge_base is None:
            self._knowledge_base = JsonFileKnowledgeBaseAdapter(
                self._settings.knowledge_base_path,
            )
        return self._knowledge_base

    def vector_store(self) -> NumpyInMemoryVectorStore:
        if self._vector_store is None:
            self._vector_store = NumpyInMemoryVectorStore(self.knowledge_base())
        return self._vector_store

    def embedding(self) -> OpenRouterEmbeddingAdapter:
        if self._embedding is None:
            self._embedding = OpenRouterEmbeddingAdapter(
                api_key=self._settings.openrouter_api_key,
                model=self._settings.openrouter_embedding_model,
                http_referer=self._settings.openrouter_http_referer,
                x_title=self._settings.openrouter_x_title,
                timeout=self._settings.openrouter_timeout_seconds,
            )
        return self._embedding

    def llm(self) -> OpenAILLMAdapter:
        if self._llm is None:
            self._llm = OpenAILLMAdapter(
                api_key=self._settings.openai_api_key,
                model=self._settings.openai_model,
                base_url=self._settings.openai_base_url,
            )
        return self._llm

    def rag_service(self) -> RAGService:
        if self._rag is None:
            self._rag = RAGService(
                self.embedding(),
                self.vector_store(),
                self.llm(),
                top_k=self._settings.rag_default_top_k,
            )
        return self._rag

    def rag_service_factory(self) -> Callable[[], RAGService]:
        return self.rag_service

    async def startup(self) -> None:
        await self.rag_service().index_knowledge_base(self.knowledge_base())


def create_app(container: Container | None = None) -> FastAPI:
    c = container or Container()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await c.startup()
        yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(create_rag_router(rag_service=c.rag_service_factory()), prefix="/api")
    return app
