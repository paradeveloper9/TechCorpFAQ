from pydantic import BaseModel, Field, field_validator

from src.domain.models import Confidence


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = None

    @field_validator("top_k")
    @classmethod
    def _top_k_positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("top_k must be at least 1")
        return v


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: Confidence
