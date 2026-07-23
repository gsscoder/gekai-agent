from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

_log = logging.getLogger(__name__)


@dataclass
class ScopeEstimate:
    mutate: bool = False  # False = trivial (main solo); True = mutate (sequencer + interpreter)


_ESTIMATE_PROMPT = (
    "you estimate the scope of a user request for a coding agent on a local workspace\n"
    "output exactly one of:\n"
    "  TRIVIAL   — a single small file, a few small edits, or a read/query; no specialist unit "
    "of work is implied; when unsure, choose this\n"
    "  MUTATE    — implementation-sized: multiple files/modules, or a distinct unit of work such "
    "as a full module or a test suite\n"
    "when unsure, choose TRIVIAL — an over-estimate just lets the agent proceed solo (harmless); "
    "an under-estimate wrongly skips planning\n"
)


class Estimator:
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
        self._prompt = _ESTIMATE_PROMPT

    async def estimate(self, user_input: str) -> ScopeEstimate:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                messages=[
                    {"role": "system", "content": self._prompt},
                    {"role": "user", "content": user_input},
                ],
            )
            raw: str = response.choices[0].message.content.strip()
        except Exception:
            _log.warning("estimate call failed; falling back to trivial", exc_info=True)
            return ScopeEstimate()

        first = raw.split()[0] if raw.split() else ""
        first_lower = first.lower()

        if first_lower == "trivial":
            return ScopeEstimate()
        if first_lower == "mutate":
            return ScopeEstimate(mutate=True)

        _log.warning("estimate parse failure — unexpected output %r; falling back to trivial", raw)
        return ScopeEstimate()
