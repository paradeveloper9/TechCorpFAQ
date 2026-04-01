from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class KnowledgeArticle(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    category: str
    question: str
    answer: str


class SearchResult(BaseModel):
    article: KnowledgeArticle
    score: float = Field(ge=0.0, le=1.0)


class RAGContext(BaseModel):
    question: str
    results: list[SearchResult]


class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: Confidence
