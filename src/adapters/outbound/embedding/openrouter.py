from typing import TYPE_CHECKING

import httpx
from openai import AsyncOpenAI

if TYPE_CHECKING:
    from openai.types import CreateEmbeddingResponse

from src.domain.ports.embedding import EmbeddingPort


class OpenRouterEmbeddingAdapter(EmbeddingPort):
    def __init__(
        self,
        api_key: str,
        model: str,
        http_referer: str = "",
        x_title: str = "",
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": http_referer,
                "X-Title": x_title,
            },
            http_client=httpx.AsyncClient(timeout=timeout),
        )

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response: CreateEmbeddingResponse = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            encoding_format="float",
        )
        sorted_data = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in sorted_data]
