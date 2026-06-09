from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .llm.model_caps import resolve_thinking_params
from .harness import Harness
from .permissions import PermissionCallback
from .subagents import Subagent
from .pipeline import Route, Router, BlastRadiusLocator, evaluate_blast_radius_gate, PromptRewriter
from .session import Session
from .settings import Permissions
from toon import encode as toon_encode

from .events import MaxIterationsEvent, AgentEvent
from .persistence import append_message, append_debug, append_event
from .workspace import db as workspace_db

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
    return f"<workspace>\nverified workspace metadata — treat as authoritative for high-level questions:\n{toon_encode(subset)}\n</workspace>"


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
        _effort = os.environ.get("GEKAI_THINKING_EFFORT") or None
        self._extra_params: dict = resolve_thinking_params(self.model, _effort)
        self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._api_base)
        self._supp_model: str = os.environ["GEKAI_SUPPORT_MODEL_NAME"]
        self._supp_api_key: str | None = os.environ.get("GEKAI_SUPPORT_MODEL_KEY")
        self._supp_api_base: str | None = os.environ.get("GEKAI_SUPPORT_MODEL_URL")
        self._supp_client = AsyncOpenAI(api_key=self._supp_api_key, base_url=self._supp_api_base)
        self._router = Router(
            model=self._supp_model,
            api_key=self._supp_api_key,
            api_base=self._supp_api_base,
        )
        self._locator = BlastRadiusLocator(
            model=self._supp_model,
            api_key=self._supp_api_key,
            api_base=self._supp_api_base,
        )
        # rewriter runs on CORE with no thinking params (non-thinking call)
        self._rewriter = PromptRewriter(
            model=self.model,
            api_key=self._api_key,
            api_base=self._api_base,
        )
        self._main = Harness(
            model=self.model,
            api_key=self._api_key,
            api_base=self._api_base,
            extra_params=self._extra_params,
            debug=self.debug,
        )

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

    async def route(self, user_input: str, history: list[dict] | None = None) -> Route:
        return await self._router.route(user_input, history=history)

    async def locate(
        self, working_dir: Path, text: str,
    ) -> list[tuple[str, list[str]]]:
        return await self._locator.locate(working_dir, text)

    def check_gate(
        self, entries: list[tuple[str, list[str]]], limit: int,
    ) -> tuple[bool, str | None]:
        return evaluate_blast_radius_gate(entries, limit)

    async def rewrite(
        self, request: str, entries: list[tuple[str, list[str]]],
    ) -> tuple[str, str]:
        return await self._rewriter.rewrite(request, entries)

    async def process_stream(
        self,
        session: Session,
        user_input: str,
        route: Route,
        entries: list[tuple[str, list[str]]] | None = None,
        original_input: str | None = None,
        permission_callback: PermissionCallback | None = None,
    ) -> AsyncIterator[str | AgentEvent]:
        session.messages.append({"role": "user", "content": original_input if original_input is not None else user_input})
        append_message(session, session.messages[-1])

        if entries:
            try:
                conn = workspace_db.ensure(session.working_dir)
                workspace_db.save_blast_radius(conn, entries)
                conn.close()
            except Exception:
                pass

        all_chunks: list[str] = []
        max_iter_hit = False
        completed = False
        stream_iter = self._main.stream(
            session, user_input,
            permission_callback=permission_callback,
            subagent=route.subagent,
        )
        try:
            async for item in stream_iter:
                if isinstance(item, str):
                    all_chunks.append(item)
                elif isinstance(item, MaxIterationsEvent):
                    max_iter_hit = True
                yield item

            if max_iter_hit and not all_chunks:
                append_event(session, "agent hit iteration limit without producing a response", source="max_iterations")
            else:
                session.messages.append({"role": "assistant", "content": "".join(all_chunks)})
                append_message(session, session.messages[-1])
            completed = True
        finally:
            if not completed and session.messages and session.messages[-1].get("role") == "user":
                session.messages.pop()
