from openai import AsyncOpenAI

from src.adapters.outbound.llm.observability import log_llm_generate
from src.domain.ports.llm import LLMPort


class OpenAILLMAdapter(LLMPort):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @log_llm_generate
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty response")
        if not isinstance(content, str):
            raise TypeError("LLM returned non-string content")

        return content
