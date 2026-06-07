from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

_log = logging.getLogger(__name__)

from .persona import SYSTEM_PROMPT
from .settings import Permissions
from .subagents import Subagent, SUBAGENTS


@dataclass
class Route:
    subagent: Subagent | None = None
    rejected: bool = False


@dataclass
class Session:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict] = field(
        default_factory=lambda: [{"role": "system", "content": SYSTEM_PROMPT}]
    )
    working_dir: Path = field(default_factory=Path.cwd)
    permissions: Permissions = field(default_factory=lambda: Permissions(read=True, write=False, exec=False))
    scope_gate: bool = True
    blast_radius_limit: int = 5


_ROUTER_PROMPT_BASE = (
    "you guard a user message for a coding agent on a local repository\n"
    "output exactly one token — no prose, no punctuation\n"
    "choices:\n"
    "  main             — the default: chat, inspection, workspace questions, general code changes, light edits — anything the main agent handles directly\n"
    "  REJECTED         — user input not in English\n"
    "  <subagent-name>  — one of the subagents below; ONLY when the request clearly and specifically matches that subagent's specialty\n"
    "bias toward main unless a specialist clearly fits\n"
    "<subagents>\n"
    "{menu}"
)


class Router:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self._subagents = list(SUBAGENTS)
        menu = "\n".join(f"  {p.name} — {p.description}" for p in self._subagents)
        self._prompt = _ROUTER_PROMPT_BASE.replace("{menu}", menu)

    async def route(
        self, user_input: str, history: list[dict] | None = None,
    ) -> Route:
        context_msgs: list[dict] = []
        if history:
            context_msgs = [m for m in history if m["role"] in ("user", "assistant")][-6:]
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": self._prompt},
                *context_msgs,
                {"role": "user", "content": user_input},
            ],
        )
        raw: str = response.choices[0].message.content.strip()
        first = raw.split()[0] if raw.split() else ""
        first_lower = first.lower()

        if first_lower == "rejected":
            return Route(rejected=True)
        if first_lower == "main":
            return Route()
        for p in self._subagents:
            if first_lower == p.name.lower():
                return Route(subagent=p)
        _log.warning("router parse failure — unknown token %r; raw: %r", first, raw)
        return Route()
