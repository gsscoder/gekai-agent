from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field

import litellm


class Intent(enum.Enum):
    CHAT = "chat"
    QUERY = "query"
    ACTION = "action"


@dataclass
class Session:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict] = field(default_factory=list)


CLASSIFIER_PROMPT = (
    "classify the user message into exactly one category\n"
    "Reply with a single word: chat, query, or action\n"
    "chat: general conversation, explanations, questions not about a specific codebase\n"
    "query: questions requiring reading files or searching the repository (read-only)\n"
    "action: requests that require modifying files in the repository (read + write)"
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
