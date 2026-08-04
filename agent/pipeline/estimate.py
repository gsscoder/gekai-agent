from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

_log = logging.getLogger(__name__)


@dataclass
class ScopeEstimate:
    scope: str = "solo"  # "chat" (root solo, no codebase access) | "solo" (root solo, codebase available) | "mutate" (sequencer + interpreter)


_ESTIMATE_PROMPT = (
    "you estimate the scope of a user request for a coding agent on a local workspace\n"
    "output exactly one of:\n"
    "  CHAT      — answerable with no codebase access: greetings, identity/capability questions, "
    "acknowledgments, general knowledge unrelated to this workspace; when unsure, do NOT choose "
    "this — choose SOLO instead\n"
    "  SOLO      — a single small file, a few small edits, or a read/query; no specialist unit of "
    "work is implied; when unsure, choose this\n"
    "  MUTATE    — implementation-sized: multiple files/modules, or a distinct unit of work such "
    "as a full module or a test suite\n"
    "when unsure between CHAT and SOLO, choose SOLO — an over-estimate just answers with the "
    "codebase available (harmless); an under-estimate wrongly skips needed codebase access\n"
    "when unsure between SOLO and MUTATE, choose SOLO — an over-estimate just lets the agent "
    "proceed solo (harmless); an under-estimate wrongly skips planning\n"
)


class Estimator:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_params: dict | None = None,
    ) -> None:
        self._model = model
        self._extra_params = extra_params or {}
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=30.0),
        )

    async def estimate(self, user_input: str, history: list[dict] | None = None) -> ScopeEstimate:
        context_msgs: list[dict] = []
        if history:
            context_msgs = [m for m in history if m["role"] in ("user", "assistant")][-6:]
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                messages=[
                    {"role": "system", "content": _ESTIMATE_PROMPT},
                    *context_msgs,
                    {"role": "user", "content": user_input},
                ],
                **self._extra_params,
            )
            raw: str = response.choices[0].message.content.strip()
        except Exception:
            _log.warning("estimate call failed; falling back to solo", exc_info=True)
            return ScopeEstimate(scope="solo")

        first = raw.split()[0] if raw.split() else ""
        first_lower = first.lower()

        if first_lower == "chat":
            return ScopeEstimate(scope="chat")
        if first_lower == "solo":
            return ScopeEstimate(scope="solo")
        if first_lower == "mutate":
            return ScopeEstimate(scope="mutate")

        _log.warning("estimate parse failure — unexpected output %r; falling back to solo", raw)
        return ScopeEstimate(scope="solo")
