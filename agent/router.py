from __future__ import annotations

import enum
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

_log = logging.getLogger(__name__)

from .profiles import NAMESPACES
from .settings import Permissions


class Intent(enum.Enum):
    CHAT = "chat"
    ACTION = "action"
    REJECTED = "rejected"


@dataclass
class Segment:
    intent: Intent
    text: str
    namespace: str | None = None  # only meaningful for Intent.ACTION


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


CLASSIFIER_PROMPT = (
    "you route messages for a coding agent working on a local code repository\n"
    "decompose the user message into one or more labeled tasks\n"
    "output format: each line must be exactly `label: text` where label is one of chat, action/coding, action/management, action/generic\n"
    "no preamble, no explanation, no markdown, no numbering — labeled lines only\n"
    "<labels>\n"
    " chat                — general coding question, explanation, or conversation; answer from knowledge\n"
    " action/coding       — create, edit, refactor, or simplify code or tests\n"
    " action/management   — repository/file organization: scaffolding, moving/renaming files, restructuring layout\n"
    " action/generic      — any other repo action: reading/inspecting code, prose/markdown/doc text edits, anything not clearly code or repo-structure\n"
    "<rules>\n"
    " assume all requests relate to the current codebase unless clearly otherwise\n"
    " when a message could fit multiple labels, prefer chat over action\n"
    " be cagey: when unsure between a code namespace and generic, prefer action/generic\n"
    "<examples>\n"
    " input: refactor auth error handling and tell me if GET /users returns JSON\n"
    "  action/coding: refactor auth error handling\n"
    "  action/generic: does GET /users return JSON\n"
    " input: describe the project\n"
    "  action/generic: describe the project\n"
    " input: how does the auth system work\n"
    "  action/generic: explain how the auth system works\n"
    " input: what's on line 10 of main.py\n"
    "  action/generic: what is on line 10 of main.py\n"
    " input: refactor error handling across all modules\n"
    "  action/coding: refactor error handling across all modules\n"
    " input: rename the variable on line 5 of utils.py\n"
    "  action/coding: rename the variable on line 5 of utils.py\n"
    " input: move the parsers into a parsers/ package\n"
    "  action/management: move the parsers into a parsers/ package\n"
    " input: fix the typo in the README\n"
    "  action/generic: fix the typo in the README\n"
    "<human_language>\n"
    "if the request is not in English, do not process it and respond exactly: REJECTED\n"
)


def evaluate_single_order_gate(
    segments: list[Segment],
) -> tuple[bool, str | None]:
    """One order per turn; >1 actionable segment is refused."""
    if len(segments) > 1:
        return (True, f"One order per turn — split into {len(segments)} separate messages")
    return (False, None)


class IntentClassifier:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def classify(
        self, user_input: str, history: list[dict] | None = None,
    ) -> list[Segment]:
        context_msgs: list[dict] = []
        if history:
            turns = [m for m in history if m["role"] in ("user", "assistant")][-6:]
            context_msgs = turns
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                *context_msgs,
                {"role": "user", "content": user_input},
            ],
        )
        raw: str = response.choices[0].message.content.strip()
        if raw.strip() == "REJECTED":
            return [Segment(Intent.REJECTED, "User input must be in English")]
        segments: list[Segment] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or ": " not in line:
                continue
            label, _, sub_prompt = line.partition(": ")
            label = label.strip().lower()
            kind, _, ns = label.partition("/")
            try:
                intent = Intent(kind)
            except ValueError:
                intent = Intent.CHAT
            if intent is Intent.ACTION:
                namespace = ns if ns in NAMESPACES else "generic"
                segments.append(Segment(Intent.ACTION, sub_prompt.strip(), namespace=namespace))
            else:
                segments.append(Segment(intent, sub_prompt.strip()))
        if not segments:
            _log.warning("classifier parse failure — no valid segments; raw output: %r", raw)
            return [Segment(Intent.CHAT, user_input)]
        return segments
