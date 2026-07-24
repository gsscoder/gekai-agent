from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass

from ..persona import _IDENTITY_SUB, _SHARED_BODY
from ..settings import Permissions
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
    auto_assignable: bool = False  # phase-1 decomposition may assign it; False = post-planning-only (verify/repair)

    def build_system_base(self) -> str:
        """Subagent identity (member, not the whole) + assigned role + the body
        shared verbatim with the main agent + directives — the <tools> block is
        appended by the harness once the effective tool set is known."""
        system = _IDENTITY_SUB
        if self.mandate:
            system += f"\n{self.mandate}"
        system += f"\n{_SHARED_BODY}"
        if self.directives:
            system += f"\n<directives>\n{self.directives}"
        return system


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
        nd = ns_directives.get(p.namespace, "")
        if nd:
            composed = nd + ("\n" + p.directives if p.directives else "")
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
    seen: set[str] = set()
    for p in SUBAGENTS:
        if p.namespace not in NAMESPACES:
            raise ValueError(f"subagent {p.name!r} has unknown namespace: {p.namespace!r}")
        if p.name in seen:
            raise ValueError(f"duplicate subagent name: {p.name!r}")
        seen.add(p.name)
        if p.tool_policy is not None and not (0 <= p.tool_policy.ceiling < len(RUNGS)):
            raise ValueError(f"subagent {p.name!r} has out-of-range tool policy ceiling: {p.tool_policy.ceiling!r}")
