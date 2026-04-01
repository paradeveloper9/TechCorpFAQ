from typing import Protocol

from src.domain.models import KnowledgeArticle


class KnowledgeBasePort(Protocol):
    async def list_all(self) -> list[KnowledgeArticle]: ...

    async def get_by_id(self, article_id: str) -> KnowledgeArticle | None: ...
