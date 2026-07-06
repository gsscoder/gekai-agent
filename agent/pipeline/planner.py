"""Planner stage (plan 27, improvement 3): decomposition + measurement.

CORE thinking, one call per mutation turn. Phase 1 (decomposition) assigns
each unit of work to `main` or an auto-assignable subagent, in dependency
order. Phase 2 (measurement) is folded into the same call: the model marks
`verify` on any step whose complexity warrants a preventive check. The
complexity metric itself is open point 1 (plan 27) — until it is designed,
the model's own in-prompt judgment is the mechanical placeholder, exactly as
validated by tests/test_planner_probe.py.
"""

from __future__ import annotations

import json
import re

import httpx
from openai import AsyncOpenAI

from ..subagents import SUBAGENTS, Subagent
from .plan import MAIN_AGENT, Plan, parse_plan

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)

_PLANNER_PROMPT = (
    "you are the planning stage of a coding harness. decompose the user's request into a "
    "JSON array of steps, each an object with keys:\n"
    '  "agent": "{main}" or one of the auto-assignable specialists below\n'
    '  "task": a self-contained instruction string for that agent\n'
    '  "verify": null, or the string "mechanical" if this step\'s change is complex enough to '
    "warrant a preventive check before moving on (no dedicated verify agent exists yet — "
    '"mechanical" is the placeholder check)\n'
    "order steps by dependency (scaffolding/logic before tests), regardless of the order "
    "mentioned in the request. never fragment one artifact across steps. "
    "respond with ONLY the JSON array — no prose, no markdown fences.\n"
    "<auto-assignable-specialists>\n"
    "{roster}\n"
    "</auto-assignable-specialists>"
)


def _build_prompt(roster: list[Subagent]) -> str:
    menu = "\n".join(f"  {p.name} — {p.description}" for p in roster)
    return _PLANNER_PROMPT.format(main=MAIN_AGENT, roster=menu)


class Planner:
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
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=30.0),
        )
        self._roster = [s for s in SUBAGENTS if s.auto_assignable]
        self._full_roster = list(SUBAGENTS)
        self._prompt = _build_prompt(self._roster)

    async def plan(self, user_input: str, *, seed: str | None = None) -> Plan:
        """Returns a validated Plan. Raises ValueError if the model's output
        fails schema/roster/phase validation (fail loud — plan 27 decision 6).
        """
        prompt = user_input if seed is None else f"primary specialist: {seed}\n\n{user_input}"
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": prompt},
            ],
            **self._extra_params,
        )
        raw_text: str = response.choices[0].message.content or ""
        match = _JSON_ARRAY.search(raw_text)
        if not match:
            raise ValueError(f"planner returned no JSON array: {raw_text!r}")
        raw = json.loads(match.group(0))
        return parse_plan(raw, self._full_roster)


__all__ = ["Planner"]
