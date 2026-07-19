"""Dynamic directive pump (plan 28 Phase 3): main borrows domain *expertise*
without a cold spawn borrowing the *role* (decision 13). Only the shallow,
mission-free `namespace_directives` groups (`agent/subagents/*/__init__.py`)
ever reach here — a `Subagent.directives` (deep, mission-presupposing) is
never pumped, by construction: it is only assembled inside
`Subagent.build_system_base()`, which this module never calls.

Open point 2 (escape-rank representation) is settled as: rank is a small
int per namespace group (`namespace_directive_rank`, lower = higher
priority), authored next to that namespace's `namespace_directives`. The
budget is a cap on *domains* pumped, not individual directive lines — each
namespace's directives are already a short, cohesive craft block, so
ranking at that granularity is sufficient and keeps authoring in the same
plain-string convention the codebase already uses. No conflict-resolution
step is needed beyond the budget cutoff: escaping directives are mission-
free by construction (decision 11), so they cannot goal-conflict, only
style-conflict, which is low-stakes and left for the model to reconcile.

Open point 3 (domain detection signals) is settled as: file extensions of
backtick-quoted paths in the prompt (the shape `PromptRewriter` produces
when wired) plus a small keyword lexicon — no filesystem access, no
located-files list threaded in, since nothing upstream of the harness
currently populates one. A turn with no located files and no domain
keywords detects zero domains and pumps nothing (the safe default).
"""

from __future__ import annotations

import re

from .subagents import NAMESPACE_DIRECTIVES, NAMESPACE_DIRECTIVE_RANK

_EXTENSION_DOMAINS: dict[str, str] = {
    ".py": "coding", ".pyi": "coding",
    ".js": "coding", ".jsx": "coding", ".ts": "coding", ".tsx": "coding",
    ".go": "coding", ".rs": "coding", ".java": "coding",
    ".c": "coding", ".cpp": "coding", ".h": "coding", ".hpp": "coding",
    ".cs": "coding", ".rb": "coding", ".php": "coding",
}

_TEST_PATH_RE = re.compile(r"(^|[/\\])test_|_test\.|\.test\.|\.spec\.|[/\\]tests?[/\\]", re.IGNORECASE)
_PATH_RE = re.compile(r"`([^`]+)`")

_KEYWORD_DOMAINS: dict[str, tuple[str, ...]] = {
    "testing": ("test", "tests", "pytest", "unittest", "assert", "coverage", "spec", "fixture"),
}

PUMP_BUDGET = 2  # cap on domains pumped into main per turn (hard problem 3)


def detect_domains(prompt: str) -> set[str]:
    """Mechanical (non-LLM) domain detection over the raw prompt text."""
    domains: set[str] = set()
    for path in _PATH_RE.findall(prompt):
        dot = path.rfind(".")
        domain = _EXTENSION_DOMAINS.get(path[dot:].lower()) if dot != -1 else None
        if domain:
            domains.add(domain)
            if _TEST_PATH_RE.search(path):
                domains.add("testing")
    lowered = prompt.lower()
    for domain, keywords in _KEYWORD_DOMAINS.items():
        if any(kw in lowered for kw in keywords):
            domains.add(domain)
    return domains


def pump(prompt: str) -> tuple[str, list[str]]:
    """Assemble the escaping directives for this turn's detected domains,
    budgeted top-k by rank. Returns (directive_text, domains_used) —
    domains_used is "" / [] when nothing was pumped (no domain detected, or
    a detected domain has no escaping directives)."""
    domains = detect_domains(prompt) & NAMESPACE_DIRECTIVES.keys()
    if not domains:
        return "", []
    chosen = sorted(domains, key=lambda d: (NAMESPACE_DIRECTIVE_RANK.get(d, 100), d))[:PUMP_BUDGET]
    return "\n".join(NAMESPACE_DIRECTIVES[d] for d in chosen), chosen
