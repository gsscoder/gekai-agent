from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path

import litellm
from dotenv import load_dotenv

from .handlers.action import ActionHandler
from .handlers.base import Handler
from .handlers.chat import ChatHandler
from .handlers.query import QueryHandler
from .router import Intent, IntentClassifier, Session
from .settings import Permissions
from .ui import console

litellm.suppress_debug_info = True
litellm.success_callback = []
litellm._async_success_callback = []
litellm.callbacks = []

load_dotenv()


def _validate_config() -> None:
    errors: list[str] = []
    if not os.environ.get("GEKAI_DEFAULT_MODEL", ""):
        errors.append("GEKAI_DEFAULT_MODEL is not set")
    if not os.environ.get("GEKAI_API_KEY", ""):
        errors.append("GEKAI_API_KEY is not set")
    if errors:
        raise RuntimeError("missing configuration:\n" + "\n".join(f"  - {e}" for e in errors))


class GekaiAgent:
    def __init__(self, *, working_dir: Path, permissions: Permissions, debug: bool = False) -> None:
        self.working_dir = working_dir
        self.permissions = permissions
        self.debug = debug
        _validate_config()
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
        return Session(working_dir=self.working_dir, permissions=self.permissions)

    async def classify(self, user_input: str) -> list[tuple[Intent, str]]:
        return await self._classifier.classify(user_input)

    async def process(self, session: Session, segments: list[tuple[Intent, str]]) -> str:
        clarifications = [sub for intent, sub in segments if intent == Intent.CLARIFY]
        if clarifications:
            questions = "\n".join(f"- {q}" for q in clarifications)
            return f"before proceeding, I need some clarification:\n{questions}"

        parts: list[str] = []
        for intent, sub_prompt in segments:
            handler = self._handlers[intent]
            parts.append(await handler.handle(session, sub_prompt))
        return "\n\n".join(parts)

    async def process_stream(
        self, session: Session, segments: list[tuple[Intent, str]]
    ) -> AsyncIterator[str]:
        clarifications = [sub for intent, sub in segments if intent == Intent.CLARIFY]
        if clarifications:
            questions = "\n".join(f"- {q}" for q in clarifications)
            yield f"before proceeding, I need some clarification:\n{questions}"
            return

        first = True
        for intent, sub_prompt in segments:
            if not first:
                yield "\n\n"
            first = False
            handler = self._handlers[intent]
            if hasattr(handler, "stream"):
                async for chunk in handler.stream(session, sub_prompt):
                    yield chunk
            else:
                result = await handler.handle(session, sub_prompt)
                yield result
