from __future__ import annotations

from collections.abc import AsyncIterator

import litellm

from ..router import Session


class ChatHandler:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base

    async def stream(self, session: Session, user_input: str) -> AsyncIterator[str]:
        session.messages.append({"role": "user", "content": user_input})
        response = await litellm.acompletion(
            model=self._model,
            messages=session.messages,
            api_key=self._api_key,
            api_base=self._api_base,
            stream=True,
        )
        chunks: list[str] = []
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                chunks.append(content)
                yield content
        reply = "".join(chunks)
        session.messages.append({"role": "assistant", "content": reply})

    async def handle(self, session: Session, user_input: str) -> str:
        chunks: list[str] = []
        async for chunk in self.stream(session, user_input):
            chunks.append(chunk)
        return "".join(chunks)
