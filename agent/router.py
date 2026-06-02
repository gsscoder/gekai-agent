from __future__ import annotations

import enum
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

_log = logging.getLogger(__name__)

from .settings import Permissions


class Intent(enum.Enum):
    CHAT = "chat"
    QUERY = "query"
    ACTION = "action"
    MEMORIZE = "memorize"
    CLARIFY = "clarify"


SYSTEM_PROMPT = (
    "you are Gekai, a coding agent operating on a local repository\n"
    "you work within a session — conversation history persists across turns\n"
    "you can read, search, and modify files in the repository through tool calls\n"
    "follow user instructions literally — do exactly what is asked; never substitute with what you think is more helpful\n"
    "<behavior>\n"
    "stay focused on the codebase and its domain\n"
    "when asked general questions, answer briefly and steer back to the task\n"
    "when modifying code, be precise and minimal — change only what is requested\n"
    "never fabricate file contents or paths — use tools to read them; when contents are already in context, present them directly\n"
    "<response_style>\n"
    "when explaining or answering questions: be terse — no filler, no hedging, no disclaimers\n"
    "prefer short sentences and fragments over verbose explanations\n"
    "answer in 1-3 sentences unless complexity demands more\n"
    "no bullet lists unless the user asks or the content is naturally a list\n"
    "never output horizontal separators of any kind: not ---, not ───, not ===, not ***, not any sequence of repeated characters forming a line\n"
    "never output more than one blank line in a row; never place blank lines before or after headings; never start a response with a blank line\n"
    "state facts and decisions directly; never open with 'I think' or 'it seems'\n"
    "<file_handling>\n"
    "when the user asks to show, print, or display a file, output its full contents in a fenced code block — never summarize, paraphrase, or editorialize"
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
    "output format: each line must be exactly `label: text` where label is one of chat, query, action, memorize, clarify\n"
    "no preamble, no explanation, no markdown, no numbering — labeled lines only\n"
    "<labels>\n"
    " chat      — general coding question, explanation, or conversation; answer from knowledge\n"
    " query     — needs to inspect the repository: read files, search code, understand structure\n"
    " action    — modifies repository files (create, edit, delete, refactor)\n"
    " memorize  — any rule, constraint, or preference that should persist across future turns: coding style,\n"
    "             project conventions, off-limits files or directories, tool preferences, or any instruction\n"
    "             that applies beyond the current request\n"
    "             (e.g. 'from now on use spaces instead of tabs', 'this project follows Google style guide',\n"
    "             'don't touch the migrations folder')\n"
    " clarify   — only use clarify if you cannot determine which files, feature area, or domain the request relates to\n"
    "<rules>\n"
    " assume all requests relate to the current codebase unless clearly otherwise\n"
    " when a message could fit multiple labels, prefer the least destructive: chat over query, query over action\n"
    " prefer chat or query over clarify — only clarify if truly blocked\n"
    "<examples>\n"
    " input: refactor auth error handling and tell me if GET /users returns JSON\n"
    "  action: refactor auth error handling\n"
    "  query: does GET /users return JSON\n"
    " input: describe the project\n"
    "  query: describe the project\n"
    " input: how does the auth system work\n"
    "  query: explain how the auth system works\n"
    " input: what's on line 10 of main.py\n"
    "  query: what is on line 10 of main.py\n"
    " input: refactor error handling across all modules\n"
    "  action: refactor error handling across all modules\n"
    " input: rename the variable on line 5 of utils.py\n"
    "  action: rename the variable on line 5 of utils.py"
)

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")


def _word_count(text: str) -> int:
    stripped = _CODE_FENCE_RE.sub("", text)
    return len(re.findall(r"\w+", stripped))


def evaluate_structural_gate(
    segments: list[tuple[Intent, str]],
    max_segments: int = 3,
    big_prompt_min_words: int = 50,
) -> tuple[bool, str | None]:
    """Pure-Python structural gate. Returns (rejected, reason)."""
    if len(segments) > max_segments:
        return (True, f"Too many tasks ({len(segments)})")
    if len(segments) == 1:
        return (False, None)
    if len(segments) > 2:
        big_count = sum(1 for _, text in segments if _word_count(text) > big_prompt_min_words)
        if big_count > 1:
            return (True, "Multiple complex tasks")
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
    ) -> list[tuple[Intent, str]]:
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
        segments: list[tuple[Intent, str]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or ": " not in line:
                continue
            label, _, sub_prompt = line.partition(": ")
            label = label.strip().lower()
            try:
                intent = Intent(label)
            except ValueError:
                intent = Intent.CHAT
            segments.append((intent, sub_prompt.strip()))
        if not segments:
            _log.warning("classifier parse failure — no valid segments; raw output: %r", raw)
            return [(Intent.CHAT, user_input)]
        return segments
