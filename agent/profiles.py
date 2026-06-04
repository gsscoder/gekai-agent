from __future__ import annotations

from dataclasses import dataclass

from .settings import Permissions

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


# registry — seed profiles per namespace; "generic" intentionally has none
PROFILES: list[AgentProfile] = [
    AgentProfile(
        name="code-refactorer",
        namespace="coding",
        description="restructure existing code without changing behavior: renames, extractions, dedup",
        directives=(
            "do not invent features\n"
            "do not alter behavior beyond the request\n"
            "preserve public signatures unless asked\n"
            "merge into existing style, introduce no new patterns"
        ),
    ),
    AgentProfile(
        name="code-simplifier",
        namespace="coding",
        description="reduce complexity and over-engineering in existing code",
        directives=(
            "remove unnecessary abstraction\n"
            "do not change observable behavior\n"
            "keep edits minimal and local"
        ),
    ),
    AgentProfile(
        name="code-expert",
        namespace="coding",
        description="general code changes when no specialized profile fits: features, fixes, tests",
        directives=(
            "merge into existing code style\n"
            "do not over-engineer\n"
            "change only what is requested\n"
            "add no speculative abstraction"
        ),
        is_fallback=True,
    ),
    AgentProfile(
        name="ws-manager",
        namespace="management",
        description="repository and file organization: scaffold, move/rename files, restructure layout when no specialized profile fits",
        directives=(
            "do not modify file contents beyond what reorganization requires\n"
            "preserve behavior\n"
            "confirm structural intent matches the request"
        ),
        is_fallback=True,
    ),
]


# internal lookup by namespace
_by_namespace: dict[str, list[AgentProfile]] = {}
for _p in PROFILES:
    _by_namespace.setdefault(_p.namespace, []).append(_p)


def profiles_for(namespace: str) -> list[AgentProfile]:
    """members of a namespace; empty for "generic" or unknown."""
    return list(_by_namespace.get(namespace, []))


def fallback_for(namespace: str) -> AgentProfile:
    """the is_fallback profile of a namespace.

    "generic" is innate (no profile) and is handled by the caller, not via
    fallback — calling this for "generic" raises like any namespace with no
    fallback.
    """
    for p in _by_namespace.get(namespace, []):
        if p.is_fallback:
            return p
    raise ValueError(f"no fallback profile for namespace: {namespace!r}")


def validate_registry() -> None:
    """startup validation; raises ValueError on any violation."""
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
