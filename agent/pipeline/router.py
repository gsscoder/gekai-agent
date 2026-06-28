from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

_log = logging.getLogger(__name__)

from ..subagents import Subagent, SUBAGENTS
from ._directives import PIPELINE_DIRECTIVES

_AGENT_LINE_RE = re.compile(r"^([\w-]+):\s*$")


@dataclass
class PlanStep:
    subagent: Subagent | None  # None => this step runs on main
    raw: str  # verbatim slice/extraction of the original prompt; NOT rewritten


@dataclass
class Route:
    subagent: Subagent | None = None
    trivial: bool = False
    explore: bool = False
    plan: list[PlanStep] | None = None
    rejected: bool = False
    reason: str = ""


def _parse_plan(raw: str, subagents: list[Subagent]) -> list[PlanStep] | None:
    """Parse a `<plan>` block into ordered `PlanStep`s. Returns None on any parse failure."""
    lines = raw.splitlines()[1:]  # drop the `<plan>` tag line
    by_name = {p.name.lower(): p for p in subagents}

    blocks: list[tuple[str, list[str]]] = []
    for line in lines:
        match = _AGENT_LINE_RE.match(line)
        if match:
            blocks.append((match.group(1), []))
        elif blocks:
            blocks[-1][1].append(line)

    if not blocks:
        return None

    steps: list[PlanStep] = []
    for name, body_lines in blocks:
        name_lower = name.lower()
        body = "\n".join(body_lines).strip("\n")
        body = body.strip()
        if not body:
            return None
        if name_lower == "main":
            steps.append(PlanStep(subagent=None, raw=body))
        elif name_lower in by_name:
            steps.append(PlanStep(subagent=by_name[name_lower], raw=body))
        else:
            return None

    return steps


_ROUTER_PROMPT_BASE = (
    "you route a user message for a coding agent on a local workspace\n"
    "output exactly one token — no prose, no punctuation\n"
    "choices:\n"
    "  TRIVIAL          — answerable with no codebase access: greetings, identity/capability "
    "questions, acknowledgments, general knowledge unrelated to this workspace; "
    "when unsure, do NOT choose this\n"
    "  EXPLORE          — a read-only investigation that ends in an answer about files or "
    "structure: \"list files\", \"where is X defined/located\", \"what files exist under Y\", "
    "\"show project structure\"; NEVER choose this if the request also asks for an edit, fix, "
    "or any change — prefer main in that case; when unsure, prefer main\n"
    "  <subagent-name>  — the request fits one subagent's specialty (see below), or explicitly asks to use or delegate the task to it by name\n"
    "  REJECTED <name>  — the user explicitly names/asks for a specific subagent by name and that "
    "name is not in <subagents> below (typo, unknown name, or a system-only agent never offered to "
    "users); do NOT substitute the closest specialty, do NOT choose main, do NOT guess — output "
    "exactly REJECTED followed by the literal name the user wrote\n"
    "  main             — the generalist default: general or simple requests, reading/explaining/"
    "running code, and building a small or simple app whole — main scaffolds the structure and "
    "writes the code itself. pick a subagent below only when the request clearly fits its "
    "specialty; that is the exception, not the default — when unsure, choose main\n"
    "  <plan>           — rare: the request holds two or more substantial deliverables that need "
    "different specialists in sequence (e.g. a non-trivial implementation plus a separate test "
    "suite); a single app or library is one deliverable, not a plan; see plan format below\n"
    "<subagents>\n"
    "{subagents-meta}\n"
    "<plan format>\n"
    "default to a single token — a plan is the rare exception, only when the request holds two or "
    "more substantial deliverables that need different specialists run in sequence\n"
    "split across distinct deliverables, never across phases of one — a single app or library is "
    "built whole by one agent, structure and code together, never as scaffold-then-code\n"
    "single token: \"create a console python app structured conventionally that sums two numbers\" "
    "— one small deliverable, main builds it whole, NOT a plan\n"
    "plan: \"build a CSV parsing library, then a comprehensive pytest suite for it\" — two "
    "substantial deliverables needing different specialists → plan: code-expert implements the "
    "library, then test-expert writes the suite\n"
    "when a plan is warranted, output exactly:\n"
    "<plan>\n"
    "<agent-name>:\n"
    "  <raw extraction of the part of the request this step covers; may span lines>\n"
    "<agent-name>:\n"
    "  <raw extraction of the part of the request this step covers; may span lines>\n"
    "where each <agent-name> is one of the subagent names above or main"
)


class Router:
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
        self._prompt = PIPELINE_DIRECTIVES + _ROUTER_PROMPT_BASE.replace("{subagents-meta}", menu)
        non_invocable = [p for p in SUBAGENTS if not p.user_invocable]
        if non_invocable:
            system_only = "\n".join(f"  {p.name}" for p in non_invocable)
            self._prompt += (
                "\n<system-only — if the user explicitly asks for one of these by name, "
                "REJECTED, never route to it>\n" + system_only
            )

    async def route(
        self, user_input: str, history: list[dict] | None = None,
    ) -> Route:
        context_msgs: list[dict] = []
        if history:
            context_msgs = [m for m in history if m["role"] in ("user", "assistant")][-6:]
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": self._prompt},
                *context_msgs,
                {"role": "user", "content": user_input},
            ],
        )
        raw: str = response.choices[0].message.content.strip()

        if raw.lower().startswith("<plan>"):
            steps = _parse_plan(raw, self._subagents)
            if not steps:
                _log.warning("router plan parse failure — raw: %r", raw)
                return Route()
            if len(steps) == 1:
                return Route(subagent=steps[0].subagent)
            return Route(plan=steps)

        first = raw.split()[0] if raw.split() else ""
        first_lower = first.lower()

        if first_lower == "main":
            return Route()
        if first_lower == "trivial":
            return Route(trivial=True)
        if first_lower == "explore":
            return Route(explore=True)
        if first_lower == "rejected":
            rest = raw.split(maxsplit=1)
            name = rest[1].strip() if len(rest) > 1 else ""
            return Route(rejected=True, reason=name)
        for p in self._subagents:
            if first_lower == p.name.lower():
                return Route(subagent=p)
        _log.warning("router parse failure — unknown token %r; raw: %r", first, raw)
        return Route()
