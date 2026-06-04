from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass

from ..settings import Permissions

# action namespaces; "generic" is innate — no profiles, selector skipped
NAMESPACES = ("coding", "management", "generic")


@dataclass(frozen=True)
class AgentProfile:
    name: str  # unique profile id, e.g. "code-refactorer"
    namespace: str  # one of NAMESPACES
    description: str  # one-line LLM selection menu entry
    directives: str  # system-prompt fragment injected after SYSTEM_PROMPT
    tools: list[str] | None = None  # tool-name allowlist; None = all tools
    permissions: Permissions | None = None  # permission overlay; None = inherit session
    is_fallback: bool = False  # marks the per-namespace residual fallback


def _discover() -> list[AgentProfile]:
    import dataclasses

    ns_directives: dict[str, str] = {}
    raw_profiles: list[AgentProfile] = []
    package = __name__
    for info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        mod = importlib.import_module(f"{package}.{info.name}")
        if info.name.startswith("_"):
            ns = getattr(mod, "namespace", None)
            nd = getattr(mod, "namespace_directives", None)
            if isinstance(ns, str) and isinstance(nd, str):
                ns_directives[ns] = nd
        else:
            p = getattr(mod, "profile", None)
            if isinstance(p, AgentProfile):
                raw_profiles.append(p)

    result: list[AgentProfile] = []
    for p in raw_profiles:
        nd = ns_directives.get(p.namespace, "")
        if nd:
            composed = nd + ("\n" + p.directives if p.directives else "")
            result.append(dataclasses.replace(p, directives=composed))
        else:
            result.append(p)
    return result


PROFILES: list[AgentProfile] = _discover()

_by_namespace: dict[str, list[AgentProfile]] = {}
for _p in PROFILES:
    _by_namespace.setdefault(_p.namespace, []).append(_p)


def profiles_for(namespace: str) -> list[AgentProfile]:
    """Members of a namespace; empty for "generic" or unknown."""
    return list(_by_namespace.get(namespace, []))


def fallback_for(namespace: str) -> AgentProfile:
    """The is_fallback profile of a namespace.

    "generic" is innate (no profile) and handled by the caller — calling this
    for "generic" raises like any namespace with no fallback.
    """
    for p in _by_namespace.get(namespace, []):
        if p.is_fallback:
            return p
    raise ValueError(f"no fallback profile for namespace: {namespace!r}")


def validate_registry() -> None:
    """Startup validation; raises ValueError on any violation."""
    seen: set[str] = set()
    for p in PROFILES:
        if p.namespace not in NAMESPACES:
            raise ValueError(f"profile {p.name!r} has unknown namespace: {p.namespace!r}")
        if p.name in seen:
            raise ValueError(f"duplicate profile name: {p.name!r}")
        seen.add(p.name)
    for ns in NAMESPACES:
        if ns == "generic":
            continue
        members = _by_namespace.get(ns, [])
        if not members:
            raise ValueError(f"namespace {ns!r} has no profiles")
        fallbacks = [p for p in members if p.is_fallback]
        if len(fallbacks) != 1:
            raise ValueError(
                f"namespace {ns!r} must have exactly one fallback profile (got {len(fallbacks)})"
            )
