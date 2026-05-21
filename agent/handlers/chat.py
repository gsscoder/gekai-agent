from __future__ import annotations

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

    async def handle(self, session: Session, user_input: str) -> str:
        session.messages.append({"role": "user", "content": user_input})
        response = await litellm.acompletion(
            model=self._model,
            messages=session.messages,
            api_key=self._api_key,
            api_base=self._api_base,
        )
        reply: str = response.choices[0].message.content
        session.messages.append({"role": "assistant", "content": reply})
        return reply
