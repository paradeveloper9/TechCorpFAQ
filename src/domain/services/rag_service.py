from src.domain.models import Confidence, RAGContext, RAGResponse, SearchResult
from src.domain.ports.embedding import EmbeddingPort
from src.domain.ports.knowledge_base import KnowledgeBasePort
from src.domain.ports.llm import LLMPort
from src.domain.ports.vector_store import VectorStorePort

_DEFAULT_SYSTEM_PROMPT = (
    "You are a support assistant. Answer using only the knowledge excerpts below. "
    "If they do not contain enough information, say so clearly and avoid guessing. "
    "Keep answers concise and actionable."
)

_DEFAULT_RETRIEVAL_QUALITY_LOW_ADDON = (
    "\n\nRetrieval quality: The strongest matches are only moderately similar to the "
    "user's question. Treat excerpts as tentative. If they do not clearly speak to "
    "the question, say so; avoid confident claims. Suggest verifying important details."
)

_DEFAULT_RETRIEVAL_QUALITY_NONE_ADDON = (
    "\n\nRetrieval quality: The strongest matches are weakly related to the user's question; "
    "they may be irrelevant or misleading. Do not imply the knowledge base clearly answers "
    "this question unless an excerpt obviously does. Prefer stating uncertainty or that no "
    "reliable match was found, and avoid inventing facts."
)


def _confidence_from_results(results: list[SearchResult]) -> Confidence:
    if not results:
        return Confidence.NONE
    best = max(r.score for r in results)
    if best >= 0.82:
        return Confidence.HIGH
    if best >= 0.65:
        return Confidence.MEDIUM
    if best >= 0.45:
        return Confidence.LOW
    return Confidence.NONE


def _format_excerpts(results: list[SearchResult]) -> str:
    blocks: list[str] = []
    for i, sr in enumerate(results, start=1):
        a = sr.article
        blocks.append(
            f"### Excerpt {i} (id={a.id}, category={a.category})\n"
            f"FAQ question: {a.question}\n"
            f"FAQ answer: {a.answer}\n"
        )
    return "\n".join(blocks)


class RAGService:
    def __init__(
        self,
        embedding: EmbeddingPort,
        vector_store: VectorStorePort,
        llm: LLMPort,
        *,
        top_k: int = 5,
        system_prompt: str | None = None,
        retrieval_quality_low_addon: str | None = None,
        retrieval_quality_none_addon: str | None = None,
    ) -> None:
        self._embedding = embedding
        self._vector_store = vector_store
        self._llm = llm
        self._top_k = top_k
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._retrieval_quality_low_addon = (
            retrieval_quality_low_addon or _DEFAULT_RETRIEVAL_QUALITY_LOW_ADDON
        )
        self._retrieval_quality_none_addon = (
            retrieval_quality_none_addon or _DEFAULT_RETRIEVAL_QUALITY_NONE_ADDON
        )

    def _retrieval_quality_system_addon(self, confidence: Confidence) -> str:
        if confidence in (Confidence.HIGH, Confidence.MEDIUM):
            return ""
        if confidence is Confidence.LOW:
            return self._retrieval_quality_low_addon
        return self._retrieval_quality_none_addon

    async def index_knowledge_base(
        self,
        knowledge_base: KnowledgeBasePort,
        *,
        clear_first: bool = True,
    ) -> None:
        if clear_first:
            await self._vector_store.clear()
        articles = await knowledge_base.list_all()
        if not articles:
            return
        texts = [f"{a.question}\n{a.answer}" for a in articles]
        embeddings = await self._embedding.embed_batch(texts)
        for article, vector in zip(articles, embeddings, strict=True):
            await self._vector_store.upsert(article.id, vector)

    async def answer(self, question: str, *, top_k: int | None = None) -> RAGResponse:
        k = self._top_k if top_k is None else top_k
        query_vector = await self._embedding.embed(question)
        results = await self._vector_store.search(query_vector, top_k=k)
        if not results:
            return RAGResponse(
                answer="No matching knowledge base content was found for this question.",
                sources=[],
                confidence=Confidence.NONE,
            )

        context = RAGContext(question=question, results=results)
        confidence = _confidence_from_results(results)
        system_prompt = self._system_prompt + self._retrieval_quality_system_addon(confidence)
        user_prompt = (
            f"{_format_excerpts(context.results)}\n---\nUser question:\n{context.question}"
        )
        answer_text = await self._llm.generate(system_prompt, user_prompt)
        sources = [r.article.id for r in results]
        return RAGResponse(answer=answer_text, sources=sources, confidence=confidence)
