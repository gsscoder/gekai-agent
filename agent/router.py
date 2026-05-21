from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

from .settings import Permissions


class Intent(enum.Enum):
    CHAT = "chat"
    QUERY = "query"
    ACTION = "action"
    CLARIFY = "clarify"


SYSTEM_PROMPT = (
    "you are Gekai, a coding agent operating on a local repository\n"
    "you work within a session — conversation history persists across turns\n"
    "you can read, search, and modify files in the repository through tool calls\n"
    "stay focused on the codebase and its domain\n"
    "when asked general questions, answer briefly and steer back to the task\n"
    "when modifying code, be precise and minimal — change only what is requested\n"
    "do not hallucinate file contents or paths; if unsure, ask or use tools to verify"
)


@dataclass
class Session:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict] = field(
        default_factory=lambda: [{"role": "system", "content": SYSTEM_PROMPT}]
    )
    working_dir: Path = field(default_factory=Path.cwd)
    permissions: Permissions = field(default_factory=lambda: Permissions(read=True, write=False))


CLASSIFIER_PROMPT = (
    "decompose the user message into one or more labeled tasks\n"
    "output one line per task in the format: {label}: {sub-prompt}\n"
    "labels:\n"
    "  chat    — answer from model knowledge, no external data needed\n"
    "  query   — needs external data: web search, docs, repo file reads\n"
    "  action  — modifies repository files (create, edit, delete)\n"
    "  clarify — the request is too ambiguous to act on; state what is unclear\n\n"
    "rules:\n"
    "  output only the labeled lines, no extra text\n"
    "  use clarify for any part that cannot be confidently categorized\n"
    "  when in doubt between chat and query, use query\n"
    "  when in doubt between query and action, use query\n\n"
    "example input: refactor auth error handling and tell me if GET /users returns JSON or YAML\n"
    "example output:\n"
    "action: refactor auth error handling\n"
    "query: does GET /users return JSON or YAML"
)


class IntentClassifier:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def classify(self, user_input: str) -> list[tuple[Intent, str]]:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
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
            try:
                intent = Intent(label.strip().lower())
            except ValueError:
                intent = Intent.CHAT
            segments.append((intent, sub_prompt.strip()))
        return segments if segments else [(Intent.CHAT, user_input)]
