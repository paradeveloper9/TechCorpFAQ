import asyncio
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.domain.models import KnowledgeArticle, SearchResult
from src.domain.ports.knowledge_base import KnowledgeBasePort
from src.domain.ports.vector_store import VectorStorePort


class NumpyInMemoryVectorStore(VectorStorePort):
    def __init__(self, kb_port: KnowledgeBasePort) -> None:
        self._kb_port = kb_port
        self._vectors: dict[str, NDArray[np.float32]] = {}

    async def upsert(self, _article_id: str, embedding: list[float]) -> None:
        vector = np.array(embedding, dtype=np.float32)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        self._vectors[_article_id] = vector

    async def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        if not self._vectors:
            return []

        query_vector = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        query_vector = query_vector / query_norm

        ids = list(self._vectors.keys())
        matrix = np.stack([self._vectors[id_] for id_ in ids], axis=0)

        scores = matrix @ query_vector

        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            partition_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = partition_indices[np.argsort(scores[partition_indices])[::-1]]

        results: list[SearchResult] = []
        fetch_tasks = []
        selected_ids = []

        for idx in top_indices:
            _article_id = ids[idx]
            score = float(scores[idx])
            selected_ids.append(_article_id)
            fetch_tasks.append(self._kb_port.get_by_id(_article_id))

        articles: list[KnowledgeArticle | None] = await asyncio.gather(*fetch_tasks)

        for _article_id, article, score in zip(
            selected_ids, articles, scores[top_indices], strict=True
        ):
            if article is None:
                continue

            results.append(SearchResult(article=article, score=float(score)))

            if len(results) == top_k:
                break

        return results

    async def delete(self, _article_id: str) -> None:
        self._vectors.pop(_article_id, None)

    async def clear(self) -> None:
        self._vectors.clear()
