"""Sequencer stage (plan 27, improvement 3; renamed under plan 28): decomposition + measurement.

CORE thinking, one call per mutation turn. Phase 1 (decomposition) assigns
each unit of work to `root` or an auto-assignable subagent, in dependency
order. Phase 2 (measurement) is folded into the same call: the model marks
`verify` on any step whose complexity warrants a preventive check. The
complexity metric itself is open point 1 (plan 27) — until it is designed,
the model's own in-prompt judgment is the mechanical placeholder, exactly as
validated by tests/test_sequencer_probe.py.
"""

from __future__ import annotations

import json
import re

import httpx
from openai import AsyncOpenAI

from ..subagents import SUBAGENTS, Subagent
from .plan import TaskGraph, parse_task_graph

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_SEQUENCER_PROMPT = (
    "you are the planning stage of a coding harness. decompose the user's request into a "
    "JSON object with keys:\n"
    '  "summary": a short 1-2 sentence gist of what the user wants, in your own words\n'
    '  "steps": a JSON array of steps, each an object with keys:\n'
    '    "agent": one of the auto-assignable specialists below — every step goes to a specialist, '
    "never to root\n"
    '    "instruction": a self-contained instruction string for that agent\n'
    '    "mission": a short human-readable phrase (~8-10 words) naming this step\'s job, '
    "distinct from the full instruction\n"
    '    "verify": null, or the string "mechanical" if this step\'s change is complex enough to '
    "warrant a preventive check before moving on (no dedicated verify agent exists yet — "
    '"mechanical" is the placeholder check)\n'
    '    "scope": optional, one of "read" (pure investigation/no file changes), "edit" (only '
    'edits existing files), "fs" (also creates/moves/deletes files or needs shell) — omit '
    "entirely if the step's tool breadth is unclear or doesn't matter; this narrows the step's "
    "tool grant, it never widens beyond the agent's normal set\n"
    "order steps by dependency (scaffolding/logic before tests), regardless of the order "
    "mentioned in the request. never fragment one artifact across steps. "
    "when the request builds something new from scratch (greenfield, not editing an existing "
    "tree), the task graph MUST structure the workspace by the established conventions of every "
    "ecosystem it touches — the canonical directory layout, entry points, and project/manifest "
    "files a practitioner of that stack expects to find. this is mandatory, not stylistic: a "
    "serious project is never a loose pile of files at the workspace root. a repository may span "
    "several ecosystems at once (e.g. a backend and a frontend in different languages); each "
    "component is laid out by its own stack's conventions independently. do not infer the stack "
    "narrowly — honor whatever conventions the requested languages, frameworks, and project kind "
    "imply. "
    "respond with ONLY the JSON object — no prose, no markdown fences\n"
    "<auto-assignable-specialists>\n"
    "{roster}\n"
    "</auto-assignable-specialists>"
)


def _build_prompt(roster: list[Subagent]) -> str:
    menu = "\n".join(f"  {p.name} — {p.description}" for p in roster)
    return _SEQUENCER_PROMPT.format(roster=menu)


class Sequencer:
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
        roster = [s for s in SUBAGENTS if s.auto_assignable]
        self._full_roster = list(SUBAGENTS)
        self._prompt = _build_prompt(roster)

    async def sequence(self, user_input: str, *, seed: str | None = None) -> TaskGraph:
        """Returns a validated TaskGraph. Raises ValueError if the model's output
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
        match = _JSON_OBJECT.search(raw_text)
        if not match:
            raise ValueError(f"sequencer returned no JSON object: {raw_text!r}")
        raw = json.loads(match.group(0))
        return parse_task_graph(raw, self._full_roster)


__all__ = ["Sequencer"]
