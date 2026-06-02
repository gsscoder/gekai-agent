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
    "output format: each line must be exactly `label: weight: text` where label is one of chat, query, action, memorize, clarify\n"
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
    "<weight>\n"
    " weight is a decimal scoring how broad, vague, or multi-faceted the task remains after decomposition\n"
    " 0.0-0.3 — precise and well-scoped: single file, named symbol, exact line\n"
    " 0.4-0.7 — moderate: one feature area, a few files, a clear goal\n"
    " 0.8-1.0 — broad or vague: cross-cutting, whole-codebase, or underspecified (will be rejected)\n"
    "<rejection>\n"
    " when ANY segment receives weight >= 0.8, also emit a final line: reject: <reason>\n"
    " reason must state only what is wrong — do not suggest alternatives or corrections\n"
    " keep it under 10 words\n"
    "<rules>\n"
    " assume all requests relate to the current codebase unless clearly otherwise\n"
    " when a message could fit multiple labels, prefer the least destructive: chat over query, query over action\n"
    " prefer chat or query over clarify — only clarify if truly blocked\n"
    "<examples>\n"
    " input: refactor auth error handling and tell me if GET /users returns JSON\n"
    "  action: 0.5: refactor auth error handling\n"
    "  query: 0.15: does GET /users return JSON\n"
    " input: describe the project\n"
    "  query: 0.4: describe the project\n"
    " input: how does the auth system work\n"
    "  query: 0.45: explain how the auth system works\n"
    " input: what's on line 10 of main.py\n"
    "  query: 0.1: what is on line 10 of main.py\n"
    " input: refactor error handling across all modules\n"
    "  action: 0.9: refactor error handling across all modules\n"
    "  reject: refactoring error handling across all modules is too broad — specify which modules or error paths\n"
    " input: rename the variable on line 5 of utils.py\n"
    "  action: 0.1: rename the variable on line 5 of utils.py"
)


# scope gate thresholds — pure-Python guard over the weighted classifier output;
# a single over-weight segment (> _MAX_SEGMENT_WEIGHT) trips it just like a count or cumulative overflow
_MAX_SEGMENTS = 3
_MAX_TOTAL_WEIGHT = 1.5
_MAX_SEGMENT_WEIGHT = 0.8
_MIN_WORDS_FOR_WEIGHT = 12
# fallback when a segment's weight field is missing or unparseable
_DEFAULT_WEIGHT = 0.5
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")


@dataclass(frozen=True)
class Classification:
    segments: list[tuple[Intent, str]]
    weights: list[float]
    total_weight: float
    gated: bool
    reason: str | None = None


def _word_count(text: str) -> int:
    stripped = _CODE_FENCE_RE.sub("", text)
    return len(stripped.split())


def _evaluate_gate(
    segments: list[tuple[Intent, str]], weights: list[float],
) -> tuple[float, bool]:
    total = sum(weights)
    if len(weights) == 1:
        gated = (
            weights[0] >= _MAX_SEGMENT_WEIGHT
            and _word_count(segments[0][1]) >= _MIN_WORDS_FOR_WEIGHT
        )
    else:
        gated = (
            len(weights) > _MAX_SEGMENTS
            or total > _MAX_TOTAL_WEIGHT
            or any(w >= _MAX_SEGMENT_WEIGHT for w in weights)
        )
    return total, gated


class IntentClassifier:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def classify(self, user_input: str, history: list[dict] | None = None) -> Classification:
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
        weights: list[float] = []
        reject_reason: str | None = None
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            label, _, rest = line.partition(": ")
            label_lower = label.strip().lower()
            if label_lower == "reject":
                raw_reason = rest.strip()[:120]
                reject_reason = (raw_reason[:1].upper() + raw_reason[1:]) if raw_reason else None
                continue
            weight_str, sep, sub_prompt = rest.partition(": ")
            if not sep:
                continue
            try:
                intent = Intent(label_lower)
            except ValueError:
                intent = Intent.CHAT
            try:
                weight = float(weight_str.strip())
            except ValueError:
                weight = _DEFAULT_WEIGHT
            segments.append((intent, sub_prompt.strip()))
            weights.append(weight)
        if not segments:
            _log.warning("classifier parse failure — no valid segments; raw output: %r", raw)
            segments = [(Intent.CHAT, user_input)]
            weights = [_DEFAULT_WEIGHT]
        total_weight, gated = _evaluate_gate(segments, weights)
        if gated:
            _log.info(
                "classifier scope gate tripped — segments=%d total_weight=%.2f",
                len(segments),
                total_weight,
            )
            if not reject_reason:
                reject_reason = f"request contains {len(segments)} tasks with cumulative complexity {total_weight:.1f}"
        return Classification(
            segments=segments, weights=weights, total_weight=total_weight,
            gated=gated, reason=reject_reason,
        )
