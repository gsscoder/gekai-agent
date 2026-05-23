from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .handlers.action import ActionHandler
from .handlers.base import Handler
from .handlers.chat import ChatHandler, UsageInfo
from .handlers.display import DisplayHandler
from .handlers.query import QueryHandler
from .router import Intent, IntentClassifier, Session
from .settings import Permissions
from toon import encode as toon_encode

from .workspace import scan_workspace
from .enrichment import enrich_workspace
from .ui import render_enrichment_header, render_enrichment_file, render_enrichment_done
from .persistence import append_message, append_debug

load_dotenv()


def _format_workspace_context(workspace: dict) -> str:
    subset: dict = {
        "workspace_name": workspace.get("workspace_name", "unknown"),
        "workspace_type": workspace.get("workspace_type", "files"),
        "branch": workspace.get("branch"),
        "primary_languages": workspace.get("primary_languages", []),
        "projects": workspace.get("projects", []),
    }
    if not subset["projects"]:
        subset["extensions"] = workspace.get("extensions", {})
    return f"<workspace>\n{toon_encode(subset)}\n</workspace>"


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
        self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._api_base)
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
            Intent.QUERY: QueryHandler(
                model=self.model,
                api_key=self._api_key,
                api_base=self._api_base,
            ),
            Intent.DISPLAY: DisplayHandler(working_dir=working_dir),
            Intent.ACTION: ActionHandler(),
        }

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def start_session(
        self,
        workspace: dict,
        restored_messages: list[dict] | None = None,
        session_id: str | None = None,
    ) -> Session:
        session = Session(working_dir=self.working_dir, permissions=self.permissions)
        if session_id:
            session.id = session_id
        session.messages.append({"role": "system", "content": _format_workspace_context(workspace)})
        if self.debug:
            append_debug(session, session.messages[0])
            append_debug(session, session.messages[-1])
        if restored_messages:
            session.messages.extend(restored_messages)
        return session

    async def classify(self, user_input: str) -> list[tuple[Intent, str, bool]]:
        return await self._classifier.classify(user_input)

    async def _enrich_if_needed(self, session: Session, intent: Intent, plan: bool) -> None:
        if intent == Intent.QUERY and plan:
            render_enrichment_header()
            result = await enrich_workspace(
                session.working_dir,
                self._client,
                self.model,
                on_file=render_enrichment_file,
            )
            render_enrichment_done(result.prompt_tokens, result.completion_tokens)

    async def process_stream(
        self, session: Session, user_input: str, segments: list[tuple[Intent, str, bool]]
    ) -> AsyncIterator[str | UsageInfo]:
        session.messages.append({"role": "user", "content": user_input})
        append_message(session, session.messages[-1])
        all_chunks: list[str] = []
        first = True

        for intent, sub_prompt, plan in segments:
            if not first:
                sep = "\n\n"
                all_chunks.append(sep)
                yield sep
            first = False

            if intent == Intent.MEMORIZE:
                session.messages.append({"role": "system", "content": f"[preference] {sub_prompt}"})
                append_message(session, session.messages[-1])
                ack = "noted."
                all_chunks.append(ack)
                yield ack

            elif intent == Intent.CLARIFY:
                handler = self._handlers[Intent.CHAT]
                if hasattr(handler, "stream"):
                    async for item in handler.stream(session, user_input):
                        if isinstance(item, str):
                            all_chunks.append(item)
                        yield item
                else:
                    result = await handler.handle(session, user_input)
                    all_chunks.append(result)
                    yield result

            else:
                await self._enrich_if_needed(session, intent, plan)
                handler = self._handlers[intent]
                if hasattr(handler, "stream"):
                    async for item in handler.stream(session, sub_prompt):
                        if isinstance(item, str):
                            all_chunks.append(item)
                        yield item
                else:
                    result = await handler.handle(session, sub_prompt)
                    all_chunks.append(result)
                    yield result

        session.messages.append({"role": "assistant", "content": "".join(all_chunks)})
        append_message(session, session.messages[-1])
