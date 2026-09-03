from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass

from ..persona import _IDENTITY_SUB, _SHARED_BODY
from ..permissions import Permissions
from ..tools.catalog import RUNGS

# action namespaces and their TUI badge colors — co-located so a namespace
# cannot be declared without a color (no fallback color at render time);
# namespaces with no user_invocable members (e.g. "generic") are innate —
# no routable subagents, selector skipped
NAMESPACE_COLORS: dict[str, str] = {
    "coding": "#FFD700",
    "testing": "red",
    "generic": "#7FDBCA",
}
NAMESPACES = tuple(NAMESPACE_COLORS)


@dataclass(frozen=True)
class ToolPolicy:
    """A unit's tool ceiling for assignment-time scoping (plan 31 Phase 1):
    an index into `agent.tools.catalog.RUNGS`, the highest rung it may ever
    be granted. Tighten-only — `harness/tool_scope.py` narrows a unit's
    grant per task-graph step but never widens it past this ceiling."""
    ceiling: int

    def __post_init__(self) -> None:
        if not (0 <= self.ceiling < len(RUNGS)):
            raise ValueError(
                f"tool policy ceiling {self.ceiling!r} out of range: must be 0 <= ceiling < {len(RUNGS)}"
            )


@dataclass(frozen=True)
class Subagent:
    name: str  # unique subagent id, e.g. "code-refactorer"
    namespace: str  # one of NAMESPACES
    description: str  # one-line LLM selection menu entry
    short_description: str = ""  # concise one-liner shown in the slash-command palette
    mandate: str = ""  # 1-2 line activation hook: "your specialization is…"
    directives: str = ""  # system-prompt fragment injected after the mandate
    tools: list[str] | None = None  # tool-name allowlist; None = all tools
    tool_policy: ToolPolicy | None = None  # assignment-time tool ceiling (plan 31 Phase 1); None = no ceiling declared, unit runs full
    permissions: Permissions | None = None  # permission overlay; None = inherit session
    user_invocable: bool = True  # router menu + prompt-quoting eligibility; False = system-managed worker
    alias: str = ""  # if set and user_invocable, shown/typed in slash palette instead of `name`
    params: str = "<subagent prompt>"  # declared parameter signature shown as a dimmed hint, e.g. `/<alias-or-name> <params>`
    auto_assignable: bool = False  # phase-1 decomposition may assign it; False = delegate-only, never auto-assigned by the sequencer
    # ADDITIONAL namespaces (beyond this subagent's own, which is always
    # auto-inherited unconditionally — never needs listing itself here) whose
    # directives should also be composed into this subagent's directives.
    # () = own namespace only (today's behavior, unchanged). ("*",) = own
    # namespace + every other registered domain, ranked by
    # NAMESPACE_DIRECTIVE_RANK then name. A domain with no directives
    # (e.g. "generic") silently contributes nothing.
    directive_domains: tuple[str, ...] = ()
    # additional roster names this unit may hand a task to via the `delegate`
    # tool (hidden-bound in `_build_agent`). () = feature dormant for this
    # unit — no `delegate` tool is registered at all, and this is the
    # default for every subagent shipped today. Depth is capped at 1: a
    # subagent built via delegation is built with `can_delegate=False`, so
    # cycles are structurally impossible, not merely disallowed by policy.
    delegates_to: tuple[str, ...] = ()
    # marks a unit as a discovery precursor: read-only investigation whose
    # report may feed a later step via `{{step_k}}`, or stand as the whole
    # graph for a pure investigation request — either way its output reaches
    # the user via `_respond`. Structurally, not name-specifically, caps a
    # graph at one such step (see `plan.py`'s `parse_task_graph`); generic so
    # any future read-only precursor unit gets the same guardrail for free.
    # False = an ordinary step, the default for every subagent shipped today.
    discovery_stage: bool = False
    # per-unit override of the global iteration ceiling (`harness/core.py`'s
    # `_MAX_ITERATIONS`) — bounds the cost of a step whose whole job is
    # cheap-and-bounded (e.g. a discovery precursor) even if it goes astray.
    # None = inherit the default.
    max_iterations: int | None = None

    def build_system_base(self) -> str:
        """Subagent identity (member, not the whole) + assigned role + the body
        shared verbatim with root + directives — the <tools> block is
        appended by the harness once the effective tool set is known."""
        system = _IDENTITY_SUB
        if self.mandate:
            system += f"\n{self.mandate}"
        system += f"\n{_SHARED_BODY}"
        if self.directives:
            system += f"\n<directives>\n{self.directives}"
        return system


def _compose_directives(p: Subagent, ns_directives: dict[str, str], ns_rank: dict[str, int]) -> str:
    """Compose a subagent's effective directives: own namespace, then any
    additional domains from `p.directive_domains` (in resolved order), then
    the subagent's own mandate-level directives. Empty parts are skipped;
    parts are joined with a single "\n" (no double newlines)."""
    own = ns_directives.get(p.namespace, "")

    if "*" in p.directive_domains:
        extra_domains = sorted(
            (d for d in ns_directives if d != p.namespace),
            key=lambda d: (ns_rank.get(d, 100), d),
        )
    else:
        seen: set[str] = set()
        extra_domains = []
        for d in p.directive_domains:
            if d == p.namespace or d in seen:
                continue
            seen.add(d)
            extra_domains.append(d)

    parts = [own] + [ns_directives.get(d, "") for d in extra_domains] + [p.directives]
    return "\n".join(part for part in parts if part)


def _discover() -> tuple[list[Subagent], dict[str, str], dict[str, int]]:
    import dataclasses

    ns_directives: dict[str, str] = {}
    ns_directive_rank: dict[str, int] = {}
    raw_subagents: list[Subagent] = []
    package = __name__
    for ns_info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        if not ns_info.ispkg:
            continue
        ns_pkg = importlib.import_module(f"{package}.{ns_info.name}")
        ns = getattr(ns_pkg, "namespace", None)
        nd = getattr(ns_pkg, "namespace_directives", None)
        if isinstance(ns, str) and isinstance(nd, str):
            ns_directives[ns] = nd
            ns_directive_rank[ns] = getattr(ns_pkg, "namespace_directive_rank", 100)
        for info in pkgutil.iter_modules(ns_pkg.__path__):
            mod = importlib.import_module(f"{package}.{ns_info.name}.{info.name}")
            p = getattr(mod, "subagent", None)
            if isinstance(p, Subagent):
                raw_subagents.append(p)

    result: list[Subagent] = []
    for p in raw_subagents:
        composed = _compose_directives(p, ns_directives, ns_directive_rank)
        if composed:
            result.append(dataclasses.replace(p, directives=composed))
        else:
            result.append(p)
    return result, ns_directives, ns_directive_rank


SUBAGENTS: list[Subagent]
NAMESPACE_DIRECTIVES: dict[str, str]
NAMESPACE_DIRECTIVE_RANK: dict[str, int]
SUBAGENTS, NAMESPACE_DIRECTIVES, NAMESPACE_DIRECTIVE_RANK = _discover()


def validate_registry() -> None:
    """Startup validation; raises ValueError on any violation."""
    for ns in NAMESPACES:
        if not NAMESPACE_COLORS.get(ns):
            raise ValueError(f"namespace {ns!r} has no badge color defined")
    # single namespace of "slash strings" — both `name` and `alias` live here,
    # since either can be typed in the slash palette; value tracks the owning
    # subagent's real name and whether it claimed the string via name/alias
    seen: dict[str, tuple[str, str]] = {}
    names = {p.name for p in SUBAGENTS}
    for p in SUBAGENTS:
        if p.namespace not in NAMESPACES:
            raise ValueError(f"subagent {p.name!r} has unknown namespace: {p.namespace!r}")
        if p.name in seen:
            other_name, other_kind = seen[p.name]
            raise ValueError(
                f"duplicate subagent name: {p.name!r} (already claimed as {other_kind} by {other_name!r})"
            )
        seen[p.name] = (p.name, "name")
        if p.alias:
            if p.alias in seen:
                other_name, other_kind = seen[p.alias]
                raise ValueError(
                    f"subagent {p.name!r} alias {p.alias!r} collides with {other_kind} of subagent {other_name!r}"
                )
            seen[p.alias] = (p.name, "alias")
        if p.tool_policy is not None and not (0 <= p.tool_policy.ceiling < len(RUNGS)):
            raise ValueError(f"subagent {p.name!r} has out-of-range tool policy ceiling: {p.tool_policy.ceiling!r}")
        if "*" in p.directive_domains and len(p.directive_domains) > 1:
            raise ValueError(
                f"subagent {p.name!r} mixes wildcard '*' with explicit entries in directive_domains: "
                f"{p.directive_domains!r} — use '*' alone or list explicit domains, not both"
            )
        for d in p.directive_domains:
            if d == "*" or d == p.namespace:
                continue
            if d not in NAMESPACE_DIRECTIVES:
                raise ValueError(
                    f"subagent {p.name!r} lists unknown/directive-less domain {d!r} in directive_domains "
                    f"(not its own namespace and not a key in NAMESPACE_DIRECTIVES)"
                )
        for target in p.delegates_to:
            if target == p.name:
                raise ValueError(f"subagent {p.name!r} lists itself in delegates_to")
            if target not in names:
                raise ValueError(f"subagent {p.name!r} lists unknown delegate target {target!r} in delegates_to")
