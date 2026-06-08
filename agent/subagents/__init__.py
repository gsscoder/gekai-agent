from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass

from ..persona import _IDENTITY_SUB, _SHARED_BODY
from ..settings import Permissions

# action namespaces and their TUI badge colors — co-located so a namespace
# cannot be declared without a color (no fallback color at render time);
# "generic" is innate — no subagents, selector skipped
NAMESPACE_COLORS: dict[str, str] = {
    "coding": "#FFD700",
    "generic": "#7FDBCA",
}
NAMESPACES = tuple(NAMESPACE_COLORS)


@dataclass(frozen=True)
class Subagent:
    name: str  # unique subagent id, e.g. "code-refactorer"
    namespace: str  # one of NAMESPACES
    description: str  # one-line LLM selection menu entry
    mandate: str = ""  # 1-2 line activation hook: "your specialization is…"
    directives: str = ""  # system-prompt fragment injected after the mandate
    tools: list[str] | None = None  # tool-name allowlist; None = all tools
    permissions: Permissions | None = None  # permission overlay; None = inherit session
    is_fallback: bool = False  # marks the per-namespace residual fallback

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


def _discover() -> list[Subagent]:
    import dataclasses

    ns_directives: dict[str, str] = {}
    raw_subagents: list[Subagent] = []
    package = __name__
    for info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        mod = importlib.import_module(f"{package}.{info.name}")
        if info.name.startswith("_"):
            ns = getattr(mod, "namespace", None)
            nd = getattr(mod, "namespace_directives", None)
            if isinstance(ns, str) and isinstance(nd, str):
                ns_directives[ns] = nd
        else:
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
    return result


SUBAGENTS: list[Subagent] = _discover()

_by_namespace: dict[str, list[Subagent]] = {}
for _p in SUBAGENTS:
    _by_namespace.setdefault(_p.namespace, []).append(_p)


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
    for ns in NAMESPACES:
        if ns == "generic":
            continue
        members = _by_namespace.get(ns, [])
        if not members:
            raise ValueError(f"namespace {ns!r} has no subagents")
        fallbacks = [p for p in members if p.is_fallback]
        if len(fallbacks) != 1:
            raise ValueError(
                f"namespace {ns!r} must have exactly one fallback subagent (got {len(fallbacks)})"
            )
