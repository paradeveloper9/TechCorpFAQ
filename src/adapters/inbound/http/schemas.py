from pydantic import BaseModel, Field

from src.domain.models import Confidence


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: Confidence
