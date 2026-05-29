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
from .handlers.query import Artifact, QueryHandler
from .normalizer import PromptNormalizer
from .router import Intent, IntentClassifier, Session
from .settings import Permissions
from toon import encode as toon_encode

from .ws_explorer import WsExplorer
from .subagent import SubAgentEvent
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
    if workspace.get("domain_map"):
        subset["domain_map"] = workspace["domain_map"]
    return f"<workspace>\nverified repository metadata — treat as authoritative for high-level questions:\n{toon_encode(subset)}\n</workspace>"


def _validate_config() -> None:
    errors: list[str] = []
    if not os.environ.get("GEKAI_CORE_MODEL_NAME", ""):
        errors.append("GEKAI_CORE_MODEL_NAME is not set")
    if not os.environ.get("GEKAI_CORE_MODEL_KEY", ""):
        errors.append("GEKAI_CORE_MODEL_KEY is not set")
    if not os.environ.get("GEKAI_SUPPORT_MODEL_NAME", ""):
        errors.append("GEKAI_SUPPORT_MODEL_NAME is not set")
    if not os.environ.get("GEKAI_SUPPORT_MODEL_KEY", ""):
        errors.append("GEKAI_SUPPORT_MODEL_KEY is not set")
    if errors:
        raise RuntimeError("missing configuration:\n" + "\n".join(f"  - {e}" for e in errors))


class GekaiAgent:
    def __init__(self, *, working_dir: Path, permissions: Permissions, debug: bool = False) -> None:
        self.working_dir = working_dir
        self.permissions = permissions
        self.debug = debug
        _validate_config()
        self.model: str = os.environ["GEKAI_CORE_MODEL_NAME"]
        self._api_key: str | None = os.environ.get("GEKAI_CORE_MODEL_KEY")
        self._api_base: str | None = os.environ.get("GEKAI_CORE_MODEL_URL")
        self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._api_base)
        self._supp_model: str = os.environ["GEKAI_SUPPORT_MODEL_NAME"]
        self._supp_api_key: str | None = os.environ.get("GEKAI_SUPPORT_MODEL_KEY")
        self._supp_api_base: str | None = os.environ.get("GEKAI_SUPPORT_MODEL_URL")
        self._supp_client = AsyncOpenAI(api_key=self._supp_api_key, base_url=self._supp_api_base)
        self._classifier = IntentClassifier(
            model=self._supp_model,
            api_key=self._supp_api_key,
            api_base=self._supp_api_base,
        )
        self._normalizer = PromptNormalizer(
            model=self._supp_model,
            api_key=self._supp_api_key,
            api_base=self._supp_api_base,
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
            Intent.ACTION: ActionHandler(),
        }

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def create_ws_explorer(self, working_dir: Path) -> WsExplorer:
        return WsExplorer(
            working_dir=working_dir,
            client=self._supp_client,
            model=self._supp_model,
        )

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

    async def normalize(self, user_input: str) -> tuple[str, str | None]:
        return await self._normalizer.normalize(user_input)

    def update_workspace_context(self, session: "Session", workspace: dict) -> None:
        session.messages[1] = {"role": "system", "content": _format_workspace_context(workspace)}

    async def process_stream(
        self,
        session: Session,
        user_input: str,
        segments: list[tuple[Intent, str, bool]],
        original_input: str | None = None,
    ) -> AsyncIterator[str | UsageInfo | SubAgentEvent]:
        session.messages.append({"role": "user", "content": original_input if original_input is not None else user_input})
        append_message(session, session.messages[-1])
        all_chunks: list[str] = []
        first = True
        artifact_content: str | None = None

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
                        elif isinstance(item, Artifact):
                            artifact_content = item.content
                        yield item
                else:
                    result = await handler.handle(session, user_input)
                    all_chunks.append(result)
                    yield result

            else:
                handler = self._handlers[intent]
                if hasattr(handler, "stream"):
                    async for item in handler.stream(session, sub_prompt):
                        if isinstance(item, str):
                            all_chunks.append(item)
                        elif isinstance(item, Artifact):
                            artifact_content = item.content
                        yield item
                else:
                    result = await handler.handle(session, sub_prompt)
                    all_chunks.append(result)
                    yield result

        if artifact_content:
            _append_bounded_artifact(session, artifact_content)

        session.messages.append({"role": "assistant", "content": "".join(all_chunks)})
        append_message(session, session.messages[-1])


def _append_bounded_artifact(session: Session, content: str) -> None:
    """Keep at most the last 8 [artifact] system messages in the in-memory transcript."""
    # Collect non-artifact messages + the most recent 7 artifacts, then append the new one
    non_artifacts = []
    recent_artifacts = []
    for m in session.messages:
        if m.get("role") == "system" and (m.get("content") or "").startswith("[artifact]"):
            recent_artifacts.append(m)
        else:
            non_artifacts.append(m)
    # Keep only the last 7 existing artifacts
    recent_artifacts = recent_artifacts[-7:]
    session.messages[:] = non_artifacts + recent_artifacts + [{"role": "system", "content": content}]
    append_message(session, session.messages[-1])
