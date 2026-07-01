from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

from ..subagents import Subagent, SUBAGENTS
from ._directives import PIPELINE_DIRECTIVES

_log = logging.getLogger(__name__)


@dataclass
class Route:
    subagent: Subagent | None = None
    trivial: bool = False
    rejected: bool = False
    reason: str = ""


_GATE_PROMPT = (
    "you classify the intent of a user message for a coding agent on a local workspace\n"
    "output exactly one of:\n"
    "  TRIVIAL         — answerable with no codebase access: greetings, identity/capability "
    "questions, acknowledgments, general knowledge unrelated to this workspace; "
    "when unsure, do NOT choose this\n"
    "  REJECTED <name> — the user explicitly names/asks for a specific agent by name and that "
    "name is not in the roster below (typo, unknown name); do NOT substitute the closest match, "
    "do NOT route to main — output exactly REJECTED followed by the literal name the user wrote\n"
    "  ACT             — everything else: reading, analysing, creating, editing, or deleting in "
    "the workspace; when unsure, choose this\n"
    "<subagents>\n"
    "{subagents-meta}\n"
)


class Gate:
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
        menu = "\n".join(f"  {p.name} — {p.description}" for p in self._subagents)
        self._prompt = PIPELINE_DIRECTIVES + _GATE_PROMPT.replace("{subagents-meta}", menu)
        non_invocable = [p for p in SUBAGENTS if not p.user_invocable]
        if non_invocable:
            system_only = "\n".join(f"  {p.name}" for p in non_invocable)
            self._prompt += (
                "\n<system-only — if the user explicitly asks for one of these by name, "
                "REJECTED, never route to it>\n" + system_only
            )

    async def gate(self, user_input: str, history: list[dict] | None = None) -> Route:
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

        if first_lower == "trivial":
            return Route(trivial=True)
        if first_lower == "rejected":
            rest = raw.split(maxsplit=1)
            name = rest[1].strip() if len(rest) > 1 else ""
            return Route(rejected=True, reason=name)
        if first_lower == "act":
            return Route()

        _log.warning("gate parse failure — unexpected output %r; falling back to ACT", raw)
        return Route()
