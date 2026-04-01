import json
from pathlib import Path

from pydantic import ValidationError

from src.domain.models import KnowledgeArticle
from src.domain.ports.knowledge_base import KnowledgeBasePort


class KnowledgeBaseLoadError(Exception):
    """Raised when knowledge base file cannot be loaded."""


class KnowledgeBaseFileNotFoundError(KnowledgeBaseLoadError):
    """Raised when knowledge base file does not exist."""


class KnowledgeBaseValidationError(KnowledgeBaseLoadError):
    """Raised when knowledge base data fails validation."""

    def __init__(self, message: str, errors: list[dict[str, str]]) -> None:
        super().__init__(message)
        self.errors = errors


class JsonFileKnowledgeBaseAdapter(KnowledgeBasePort):
    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path
        self._articles: list[KnowledgeArticle] | None = None

    async def list_all(self) -> list[KnowledgeArticle]:
        if self._articles is None:
            self._articles = self._load_articles()
        return self._articles

    async def get_by_id(self, article_id: str) -> KnowledgeArticle | None:
        articles = await self.list_all()
        for article in articles:
            if article.id == article_id:
                return article
        return None

    def _load_articles(self) -> list[KnowledgeArticle]:
        if not self._file_path.exists():
            raise KnowledgeBaseFileNotFoundError(
                f"Knowledge base file not found: {self._file_path}"
            )

        try:
            with self._file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise KnowledgeBaseLoadError(f"Invalid JSON in knowledge base file: {e}") from e

        if not isinstance(data, list):
            raise KnowledgeBaseLoadError("Knowledge base must be a JSON array of articles")

        articles: list[KnowledgeArticle] = []
        errors: list[dict[str, str]] = []

        for idx, item in enumerate(data):
            try:
                article = KnowledgeArticle.model_validate(item)
                articles.append(article)
            except ValidationError as e:
                errors.append(
                    {
                        "index": str(idx),
                        "id": item.get("id", "unknown"),
                        "error": str(e),
                    }
                )

        if errors:
            raise KnowledgeBaseValidationError(
                f"Failed to validate {len(errors)} article(s)",
                errors=errors,
            )

        return articles
