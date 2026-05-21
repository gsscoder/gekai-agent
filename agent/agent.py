from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator

from dotenv import load_dotenv

from .handlers.action import ActionHandler
from .handlers.base import Handler
from .handlers.chat import ChatHandler
from .handlers.query import QueryHandler
from .router import Intent, IntentClassifier, Session

load_dotenv()


class GekaiAgent:
    def __init__(self, *, debug: bool = False) -> None:
        self.debug = debug
        if debug:
            logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        self.model: str = os.environ["GEKAI_DEFAULT_MODEL"]
        self._api_key: str | None = os.environ.get("GEKAI_API_KEY")
        self._api_base: str | None = os.environ.get("GEKAI_BASE_URL")
        self._classifier = IntentClassifier(
            model=self.model,
            api_key=self._api_key,
            api_base=self._api_base,
        )
        self._handlers: dict[Intent, Handler] = {
            Intent.CHAT: ChatHandler(
                model=self.model,
                api_key=self._api_key,
                api_base=self._api_base,
            ),
            Intent.QUERY: QueryHandler(),
            Intent.ACTION: ActionHandler(),
        }

    def start_session(self) -> Session:
        return Session()

    async def process(self, session: Session, user_input: str) -> str:
        intent = await self._classifier.classify(user_input)

        if self.debug:
            from rich.console import Console
            Console(stderr=True).print(
                f"[grey50]debug: intent={intent.value}[/grey50]"
            )

        handler = self._handlers[intent]
        return await handler.handle(session, user_input)

    async def process_stream(
        self, session: Session, user_input: str
    ) -> AsyncIterator[str]:
        intent = await self._classifier.classify(user_input)

        if self.debug:
            from rich.console import Console
            Console(stderr=True).print(
                f"[grey50]debug: intent={intent.value}[/grey50]"
            )

        handler = self._handlers[intent]
        if hasattr(handler, "stream"):
            async for chunk in handler.stream(session, user_input):
                yield chunk
        else:
            result = await handler.handle(session, user_input)
            yield result
