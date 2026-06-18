from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

_log = logging.getLogger(__name__)

from ..subagents import Subagent, SUBAGENTS
from ._directives import PIPELINE_DIRECTIVES


@dataclass
class Route:
    subagent: Subagent | None = None
    rejected: bool = False
    trivial: bool = False
    explore: bool = False


_ROUTER_PROMPT_BASE = (
    "you route a user message for a coding agent on a local workspace\n"
    "output exactly one token — no prose, no punctuation\n"
    "choices:\n"
    "  REJECTED         — the message is not in English\n"
    "  TRIVIAL          — answerable with no codebase access: greetings, identity/capability "
    "questions, acknowledgments, general knowledge unrelated to this workspace; "
    "when unsure, do NOT choose this\n"
    "  EXPLORE          — a read-only investigation that ends in an answer about files or "
    "structure: \"list files\", \"where is X defined/located\", \"what files exist under Y\", "
    "\"show project structure\"; NEVER choose this if the request also asks for an edit, fix, "
    "or any change — prefer main in that case; when unsure, prefer main\n"
    "  <subagent-name>  — the request fits one subagent's specialty (see below), or explicitly asks to use or delegate the task to it by name\n"
    "  main             — anything else; handled directly by the coding agent\n"
    "<subagents>\n"
    "{subagents-meta}"
)


class Router:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=30.0),
        )
        self._subagents = [p for p in SUBAGENTS if p.user_invocable]
        menu = "\n".join(
            f"  {p.name} — {p.description}" + (
                f" — also pick this for any other {p.namespace}-type request that doesn't match a "
                "more specific subagent above"
                if p.is_fallback else ""
            )
            for p in self._subagents
        )
        self._prompt = PIPELINE_DIRECTIVES + _ROUTER_PROMPT_BASE.replace("{subagents-meta}", menu)

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
        if first_lower == "trivial":
            return Route(trivial=True)
        if first_lower == "explore":
            return Route(explore=True)
        for p in self._subagents:
            if first_lower == p.name.lower():
                return Route(subagent=p)
        _log.warning("router parse failure — unknown token %r; raw: %r", first, raw)
        return Route()
