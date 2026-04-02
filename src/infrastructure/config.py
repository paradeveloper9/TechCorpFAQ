from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str
    openai_model: str = "openai/gpt-5.4-nano"
    openai_base_url: str = "https://openrouter.ai/api/v1"

    openrouter_api_key: str
    openrouter_embedding_model: str = "openai/text-embedding-3-small"
    openrouter_http_referer: str = ""
    openrouter_x_title: str = Field(default="RAGAssistance")
    openrouter_timeout_seconds: float = 30.0

    knowledge_base_path: Path = Path("data/knowledge_base.json")
    rag_default_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
