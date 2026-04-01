from typing import Protocol


class LLMPort(Protocol):
    async def generate(self, system_prompt: str, user_prompt: str) -> str: ...
