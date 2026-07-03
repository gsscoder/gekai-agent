from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx
from openai import AsyncOpenAI

from ..subagents import SUBAGENTS

_log = logging.getLogger(__name__)


@dataclass
class ScopeEstimate:
    implementation_sized: bool = False
    specialists: list[str] = field(default_factory=list)


_ESTIMATE_PROMPT = (
    "you estimate the scope of a user request for a coding agent on a local workspace\n"
    "output exactly one of:\n"
    "  TRIVIAL                  — a single small file or a few small edits; no specialist unit "
    "of work is implied; when unsure, choose this\n"
    "  IMPLEMENTATION <names>   — implementation-sized: multiple files/modules, or a distinct "
    "unit of work such as a full module or a test suite; <names> is a comma-separated list of "
    "specialist(s) from the roster below whose work the request implies\n"
    "when unsure, choose TRIVIAL — an over-estimate just lets the agent proceed solo (harmless); "
    "an under-estimate wrongly blocks solo work\n"
    "<subagents>\n"
    "{subagents-meta}\n"
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
        self._subagents = [p for p in SUBAGENTS if p.user_invocable]
        menu = "\n".join(f"  {p.name} — {p.description}" for p in self._subagents)
        self._prompt = _ESTIMATE_PROMPT.replace("{subagents-meta}", menu)
        self._roster_names = {p.name for p in self._subagents}

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
        if first_lower == "implementation":
            rest = raw.split(maxsplit=1)
            names_str = rest[1].strip() if len(rest) > 1 else ""
            names = [n.strip() for n in names_str.split(",") if n.strip()]
            specialists = [n for n in names if n in self._roster_names]
            return ScopeEstimate(implementation_sized=True, specialists=specialists)

        _log.warning("estimate parse failure — unexpected output %r; falling back to trivial", raw)
        return ScopeEstimate()
