from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from ..router import Session


class ChatHandler:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def stream(self, session: Session, user_input: str) -> AsyncIterator[str]:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=session.messages,
            stream=True,
        )
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content

    async def handle(self, session: Session, user_input: str) -> str:
        chunks: list[str] = []
        async for chunk in self.stream(session, user_input):
            chunks.append(chunk)
        return "".join(chunks)
