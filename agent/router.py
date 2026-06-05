from __future__ import annotations

import enum
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

_log = logging.getLogger(__name__)

from .profiles import AgentProfile, PROFILES
from .settings import Permissions


class Intent(enum.Enum):
    CHAT = "chat"
    ACTION = "action"
    REJECTED = "rejected"


@dataclass
class Route:
    intent: Intent
    profile: AgentProfile | None = None

    @property
    def namespace(self) -> str | None:
        if self.intent is not Intent.ACTION:
            return None
        return self.profile.namespace if self.profile else "generic"


SYSTEM_PROMPT = (
    "you are Gekai, a coding agent operating on a local repository\n"
    "you can read, search, and modify files in the repository through tool calls\n"
    "follow user instructions literally — do exactly what is asked; never substitute with what you think is more helpful\n"
    "<behavior>\n"
    "stay focused on the codebase and its domain\n"
    "when asked general questions, answer briefly and steer back to the task\n"
    "when modifying code, be precise and minimal — change only what is requested\n"
    "never fabricate file contents or paths — use tools to read them; when contents are already in context, present them directly\n"
    "<file_handling>\n"
    "when the user asks to show, print, or display a file, output exactly this format: first a line `Display(filename)` where filename is the basename only, then the full file contents in a fenced code block — never summarize, paraphrase, or editorialize\n"
    "<response_style>\n"
    "IMPORTANT: be terse — no filler, no hedging, no disclaimers. If you can say it in one sentence, don't use three.\n"
    "prefer short sentences and fragments over verbose explanations\n"
    "answer in 1-3 sentences unless complexity demands more\n"
    "state facts and decisions directly; never open with 'I think' or 'it seems'\n"
    "<output_format>\n"
    "IMPORTANT: never use consecutive blank lines; never place a blank line after an intro line (a line ending with a colon or that introduces what follows); no blank lines before, after, or between items in code blocks, file trees, or diagrams; never start a response with a blank line\n"
    "no bullet lists unless the user asks or the content is naturally a list\n"
    "never output horizontal separators of any kind: not ---, not ───, not ===, not ***, not any sequence of repeated characters forming a line"
)


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
    "you route a user message for a coding agent on a local repository\n"
    "output exactly one token — no prose, no punctuation\n"
    "choices:\n"
    "  chat           — general coding question answered from knowledge; no repo access needed\n"
    "  action/generic — inspect repo, answer workspace questions, light prose/doc edits\n"
    "  REJECTED       — user input not in English\n"
    "  <profile-name> — one of the profiles below; for changes that create or modify code/structure\n"
    "prefer chat over action when unsure; prefer action/generic over a profile when the change scope is unclear\n"
    "<profiles>\n"
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
        self._profiles = list(PROFILES)
        menu = "\n".join(f"  {p.name} — {p.description}" for p in self._profiles)
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
            return Route(intent=Intent.REJECTED)
        if first_lower == "chat":
            return Route(intent=Intent.CHAT)
        if first_lower == "action/generic":
            return Route(intent=Intent.ACTION)
        for p in self._profiles:
            if first_lower == p.name.lower():
                return Route(intent=Intent.ACTION, profile=p)
        _log.warning("router parse failure — unknown token %r; raw: %r", first, raw)
        return Route(intent=Intent.ACTION)
