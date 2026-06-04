from __future__ import annotations

import logging

from openai import AsyncOpenAI

_log = logging.getLogger(__name__)

from .profiles import AgentProfile, fallback_for, profiles_for


SELECTOR_PROMPT = (
    "you select the behavioral profile for a coding-agent action within one namespace\n"
    "choose exactly one profile by name from the menu, or output `none` if none fits\n"
    "prefer the most specific match\n"
    "output the name only — no prose, no punctuation\n"
)


class ProfileSelector:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def select(
        self, namespace: str, text: str, history: list[dict] | None = None,
    ) -> AgentProfile:
        candidates = profiles_for(namespace)
        if not candidates:
            # caller must skip selection for namespaces without profiles (e.g. "generic" or unknown)
            raise ValueError(
                f"no profiles for namespace {namespace!r}; caller must skip selection"
            )
        # one menu line per candidate
        menu = "\n".join(f" {p.name} — {p.description}" for p in candidates)
        context_msgs: list[dict] = []
        if history:
            turns = [m for m in history if m["role"] in ("user", "assistant")][-6:]
            context_msgs = turns
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": SELECTOR_PROMPT + "\n<profiles>\n" + menu},
                *context_msgs,
                {"role": "user", "content": text},
            ],
        )
        raw: str = response.choices[0].message.content.strip()
        # first whitespace-delimited token, lowercased
        first = raw.split()[0].lower() if raw.split() else ""
        for p in candidates:
            if first == p.name.lower():
                return p
        # none / empty / unparseable / unknown → correct fallback beats wrong specialist
        if first != "none":
            _log.warning(
                "selector parse failure — no candidate match; raw output: %r", raw
            )
        return fallback_for(namespace)
