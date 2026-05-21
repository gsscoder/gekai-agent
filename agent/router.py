from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field

import litellm


class Intent(enum.Enum):
    CHAT = "chat"
    QUERY = "query"
    ACTION = "action"


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


CLASSIFIER_PROMPT = (
    "classify the user message into exactly one category\n"
    "reply with a single word: chat, query, or action\n"
    "chat: the user is having a conversation, asking for explanations, or discussing "
    "topics that can be answered from your own knowledge without external lookups\n"
    "query: the user needs information that requires fetching external data — "
    "searching the web, reading documentation, scanning repository files, "
    "looking up APIs, checking current state of anything outside this conversation\n"
    "action: the user wants to modify files, write code, create or delete resources "
    "in the repository\n\n"
    "when in doubt between chat and query, prefer query\n"
    "when in doubt between query and action, prefer query"
)


class IntentClassifier:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base

    async def classify(self, user_input: str) -> Intent:
        response = await litellm.acompletion(
            model=self._model,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": user_input},
            ],
            api_key=self._api_key,
            api_base=self._api_base,
        )
        raw: str = response.choices[0].message.content.strip().lower()
        try:
            return Intent(raw)
        except ValueError:
            return Intent.CHAT
