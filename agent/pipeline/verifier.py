"""Verifier stage: a single, tool-less, cold LLM call that checks one coding
step's actual diff against the user's original request and the step's own
instruction, structurally identical to `Sequencer` (client construction,
prompt-building, JSON parsing).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..oneshot import complete
from ..tiers.resolve import ResolvedTier

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_DIFF_CHAR_LIMIT = 8000


@dataclass(frozen=True)
class Verdict:
    """Result of one verifier call. `ok=True` on pass, on a fail-open
    (unparseable response, API error) — the verifier must never be the
    reason a good turn halts — and on a gate-skipped step (no LLM call
    made). `violations` is only ever non-empty on a real, parsed FAIL."""
    ok: bool
    violations: list[str] = field(default_factory=list)


_VERIFIER_PROMPT = (
    "you are the verification stage of a coding harness. you receive one completed step "
    "from a larger plan: the user's original request, the specific instruction this step "
    "was given, the diff it actually produced, and the step's own text output/self-report. "
    "the step may be one of several in the plan — judge it ONLY against its own instruction "
    "and the parts of the user request relevant to that instruction, never demand the whole "
    "request be satisfied by this one step. "
    "judge from the diff itself, not the self-report — the self-report can be wrong or "
    "incomplete. check literal correctness against the instruction: naming and contract "
    "consistency across files touched in the same diff (e.g. a field renamed on one side of "
    "an API boundary must match the other side), whether the change plausibly does what the "
    "instruction asked, and whether it violates an explicit MUST-NOT or scope-limiting "
    "constraint stated in the user's request or the step instruction (e.g. \"only for X, not "
    "for Y\"). also check for a lifecycle-invalidation gap: when the diff introduces new "
    "persistent client- or server-side state (a storage key, a cache entry, a timestamp/marker "
    "used to skip repeated work like re-validation), check whether the diff also handles that "
    "state's invalidation/clearing on a lifecycle event (e.g. logout, session end, expiry) that "
    "is visible in the diff's own context or implied by the user's request or step instruction — "
    "using only what the diff, instruction, and request show, never by inventing app behavior "
    "you can't see. flag a gap as a violation, naming the specific state introduced and the "
    "specific event it is not cleared on, only when that gap is visible from the given "
    "diff/instruction/request alone; never guess or hallucinate that a gap exists when they give "
    "no evidence the relevant lifecycle event exists at all. never flag a pre-existing issue "
    "visible only in unchanged/context lines of the "
    "diff that the step's instruction never asked it to fix. "
    "respond with ONLY a JSON object — no prose, no markdown fences:\n"
    '{"verdict": "pass" | "fail", "violations": ["...", "..."]}\n'
    '"violations" is a short list of concrete, specific problems — empty when verdict is '
    '"pass".'
)


class Verifier:
    def __init__(self, tier: ResolvedTier) -> None:
        self._tier = tier

    async def verify(
        self,
        *,
        user_input: str,
        step_instruction: str,
        diff_text: str,
        output: str,
    ) -> Verdict:
        """Never raises. Any exception (network, API, parse) is caught and
        treated as fail-open: Verdict(ok=True, violations=[]) — a broken
        verifier must never be the reason a working turn halts."""
        try:
            truncated_diff = diff_text
            if len(truncated_diff) > _DIFF_CHAR_LIMIT:
                truncated_diff = truncated_diff[:_DIFF_CHAR_LIMIT] + "... (truncated)"
            message = (
                f"<user_request>\n{user_input}\n</user_request>\n"
                f"<step_instruction>\n{step_instruction}\n</step_instruction>\n"
                f"<diff>\n{truncated_diff}\n</diff>\n"
                f"<step_output>\n{output}\n</step_output>"
            )
            raw_text = await complete(
                self._tier, system=_VERIFIER_PROMPT, user=message, read_timeout=60.0,
            )
            match = _JSON_OBJECT.search(raw_text)
            if not match:
                return Verdict(ok=True, violations=[])
            verdict = json.loads(match.group(0))
            if verdict.get("verdict") == "fail":
                violations = verdict.get("violations", [])
                if not isinstance(violations, list):
                    violations = []
                return Verdict(ok=False, violations=violations)
            return Verdict(ok=True, violations=[])
        except Exception:
            return Verdict(ok=True, violations=[])


__all__ = ["Verdict", "Verifier"]
