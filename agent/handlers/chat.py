from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai import AsyncOpenAI

from ..permissions import PermissionCallback
from ..router import Session


@dataclass(frozen=True)
class UsageInfo:
    prompt_tokens: int
    completion_tokens: int


class ChatHandler:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_params: dict | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self._extra_params = extra_params or {}

    async def stream(
        self,
        session: Session,
        user_input: str,
        permission_callback: PermissionCallback | None = None,
    ) -> AsyncIterator[str | UsageInfo]:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=session.messages,
            stream=True,
            stream_options={"include_usage": True},
            **self._extra_params,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
            if chunk.usage is not None:
                yield UsageInfo(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                )

    async def handle(self, session: Session, user_input: str) -> str:
        chunks: list[str] = []
        async for item in self.stream(session, user_input):
            if isinstance(item, str):
                chunks.append(item)
        return "".join(chunks)
