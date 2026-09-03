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

from ..oneshot import complete
from ..tiers.resolve import ResolvedTier
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
    "insert a discovery step — an auto-assignable specialist offering read-only workspace "
    "exploration — ahead of a step whose own investigation would otherwise take more than ~5 "
    "read/grep/search calls, because the request's radius is wide, spans multiple layers or "
    "modules, or names no specific target; when unsure, omit it — omitting a needed discovery "
    "step is recoverable (the assigned specialist can still search on its own), inserting an "
    "unneeded one is pure waste (an extra cold-context run, extra latency, extra tokens). when "
    "the request is a pure read-only question or investigation — no change requested yet — a "
    "single discovery step is the whole graph: its report becomes the final answer directly. "
    "never place one when the request already names the specific file(s), symbol(s), or "
    "directory to touch. never insert a discovery step when the file set it would investigate "
    "is the same file set the following implementation step will edit — a cold-spawned "
    "implementing agent MUST re-read a file's exact current bytes before it can safely edit it, "
    "so a discovery report over that same set saves it nothing; that agent discovers inline by "
    "opening the files it already knows it needs. reserve a genuine discovery step for a pure "
    "investigation/question (the single-step case above) or for narrowing a wider candidate set "
    "down to the few files that actually need edits — surveying many files to find which ones "
    "matter is real work an implementing step should not have to redo, unlike re-confirming a "
    "set the caller already handed it. "
    'a step\'s "instruction" may contain {{{{step_1}}}}, {{{{step_2}}}}, etc. (1-based) to receive that '
    "prior step's full output verbatim, substituted in before the referencing step runs; the "
    "referenced index must be an earlier step in the same graph, never itself or a later one. a "
    "discovery step's report always reaches the final answer regardless, but that does not make "
    "{{{{step_k}}}} optional: when the graph has a discovery step, every later step whose work "
    "depends on files that step already found MUST reference {{{{step_k}}}} with that discovery "
    "step's output, so it never re-discovers from scratch what was already found. "
    "when the request builds something new from scratch (greenfield, not editing an existing "
    "tree), the task graph MUST structure the workspace by the established conventions of every "
    "ecosystem it touches — the canonical directory layout, entry points, and project/manifest "
    "files a practitioner of that stack expects to find. this is mandatory, not stylistic: a "
    "serious project is never a loose pile of files at the workspace root. a repository may span "
    "several ecosystems at once (e.g. a backend and a frontend in different languages); each "
    "component is laid out by its own stack's conventions independently. do not infer the stack "
    "narrowly — honor whatever conventions the requested languages, frameworks, and project kind "
    "imply. "
    "when the request context shows the workspace already has a test suite or framework in "
    "place — test files mentioned in context, a testing dependency in a manifest, or a stack "
    "whose ecosystem conventionally ships one — and the graph makes a non-trivial code change, "
    "the graph MUST include a dedicated verification step (a test-writing/test-running "
    'specialist, or "verify" set on the relevant step) rather than leaving test coverage to '
    "chance; never fabricate a test-suite signal that isn't actually there. skip this for "
    "trivial one-line changes — the same don't-insert-steps-you-don't-need judgment governs "
    "here as everywhere else in this prompt. "
    "respond with ONLY the JSON object — no prose, no markdown fences\n"
    "<auto-assignable-specialists>\n"
    "{roster}\n"
    "</auto-assignable-specialists>"
)


def _build_prompt(roster: list[Subagent]) -> str:
    menu = "\n".join(f"  {p.name} — {p.description}" for p in roster)
    return _SEQUENCER_PROMPT.format(roster=menu)


class Sequencer:
    def __init__(self, tier: ResolvedTier) -> None:
        self._tier = tier
        roster = [s for s in SUBAGENTS if s.auto_assignable]
        self._full_roster = list(SUBAGENTS)
        self._prompt = _build_prompt(roster)

    async def sequence(self, user_input: str) -> TaskGraph:
        """Returns a validated TaskGraph. Raises ValueError if the model's output
        fails schema/roster/phase validation (fail loud — plan 27 decision 6).
        """
        raw_text = await complete(
            self._tier, system=self._prompt, user=user_input, read_timeout=120.0,
        )
        match = _JSON_OBJECT.search(raw_text)
        if not match:
            raise ValueError(f"sequencer returned no JSON object: {raw_text!r}")
        raw = json.loads(match.group(0))
        return parse_task_graph(raw, self._full_roster)


__all__ = ["Sequencer"]
