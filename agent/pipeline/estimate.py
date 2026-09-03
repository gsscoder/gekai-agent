from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from ..oneshot import complete
from ..tiers.resolve import ResolvedTier

_log = logging.getLogger(__name__)


@dataclass
class ScopeEstimate:
    scope: Literal["chat", "solo", "mutate"] = "solo"  # "chat" (root solo, no codebase access) | "solo" (root solo, codebase available) | "mutate" (sequencer + interpreter)


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
    "a query or investigation that must scan many files or modules across the codebase is MUTATE, "
    "not SOLO, even when no file will be changed — breadth of scope is the MUTATE signal here, not "
    "whether anything gets edited\n"
    "a request naming more than one deliverable, or that spans more than one layer of the stack "
    "(e.g. a backend endpoint plus its frontend UI, or a schema change plus the route that reads "
    "it), is MUTATE even when each individual piece looks small — this is a decisive signal, not "
    "a case of being unsure\n"
    "a request about this workspace/repo/app/code is never CHAT, no matter how briefly it is "
    "phrased or how short an answer it invites — e.g. \"what does this app do\" or \"briefly "
    "explain this repo\" both need a look at the codebase to answer correctly, so both are SOLO "
    "at least; words like \"briefly\", \"in short\", \"quick summary\" describe the desired answer "
    "length, not the scope; only a turn that is genuinely codebase-independent (greetings, "
    "identity questions, general knowledge unrelated to this workspace) is CHAT\n"
    "when unsure between CHAT and SOLO, choose SOLO — an over-estimate just answers with the "
    "codebase available (harmless); an under-estimate wrongly skips needed codebase access\n"
    "when unsure between SOLO and MUTATE, choose SOLO — an over-estimate just lets the agent "
    "proceed solo (harmless); an under-estimate wrongly skips planning\n"
)


class Estimator:
    def __init__(self, tier: ResolvedTier) -> None:
        self._tier = tier

    async def estimate(self, user_input: str, history: list[dict] | None = None) -> ScopeEstimate:
        context_msgs: list[dict] = []
        if history:
            context_msgs = [m for m in history if m["role"] in ("user", "assistant")][-6:]
        try:
            raw = (await complete(
                self._tier,
                system=_ESTIMATE_PROMPT,
                user=user_input,
                context=context_msgs,
                temperature=0,
            )).strip()
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
