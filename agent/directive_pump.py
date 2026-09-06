"""Dynamic directive pump (plan 28 Phase 3): root borrows domain *expertise*
without a cold spawn borrowing the *role* (decision 13). Only the shallow,
mission-free `namespace_directives` groups (`agent/subagents/*/__init__.py`)
ever reach here — a `Subagent.directives` (deep, mission-presupposing) is
never pumped, by construction: it is only assembled inside
`Subagent.build_system_base()`, which this module never calls.

Directives are pumped unconditionally into every eligible turn, rather than
being detected from the prompt's text shape. The original design detected
domains from backtick-quoted file paths and a keyword lexicon in the raw
prompt — but the backtick shape was only ever produced by `PromptRewriter`,
which is never constructed anywhere in the codebase (dead code). That made
detection a silent no-op on any plain-English request: zero domains
detected, zero directives pumped, even though root was about to write code.
Eligibility is now decided by the caller instead: root gets the pump on
every turn where it has tool access to write code, and skips it only on the
"chat" rung (pure greeting/chit-chat, no code involved -- the rung still
carries root's normal tool schemas, it just never needs domain expertise).

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

Plan 36 Phase 1 adds a second, independent axis alongside the domain pump
above: language craft, resolved deterministically from a task's file
extensions and manifests rather than pumped unconditionally. It is a
genuinely separate registry (`LANGUAGE_DIRECTIVES`) rather than folded into
`NAMESPACE_DIRECTIVES`, because a language has no TUI namespace or badge to
register under — there is nothing for `validate_registry()` to check a
colour against. `LANGUAGE_BUDGET` is kept as its own pool, deliberately not
sharing `PUMP_BUDGET`'s cap: a Python testing turn must not have to choose
between a domain slot and a language slot, since the two axes are
orthogonal and neither should starve the other.
"""

from __future__ import annotations

from pathlib import Path

from .subagents import NAMESPACE_DIRECTIVES, NAMESPACE_DIRECTIVE_RANK

PUMP_BUDGET = 2  # cap on domains pumped into root per turn (hard problem 3)

# Answers "which directives", not "which tree-sitter parser" — deliberately
# not shared with `agent/workspace/symbols.py::_EXT_TO_LANG`, which answers
# the parser question and will diverge once a language gets directives with
# no parser installed.
_EXT_TO_LANG: dict[str, str] = {".py": "python", ".ts": "typescript"}

LANGUAGE_BUDGET = 2  # cap on languages detected per turn, its own pool (see module docstring)

# Manifest filename(s) per language, used only when no extension hit exists.
_LANG_MANIFESTS: dict[str, tuple[str, ...]] = {
    "python": ("pyproject.toml",),
    "typescript": ("tsconfig.json",),
}


def detect_languages(task: str, working_dir: Path) -> list[str]:
    """Deterministic, non-LLM detection of the languages a task touches.
    Scans `task` for substrings carrying a known extension, not immediately
    followed by another word character (so `.tsx` never registers as a
    `.ts` hit), first; falls back to a manifest file's presence directly
    under `working_dir` only when no extension hit exists at all. Returns
    the top `LANGUAGE_BUDGET` languages ranked by hit count descending,
    then name ascending — `[]` when neither signal fires.

    Scans via `str.find`, not a regex, deliberately: an unanchored regex
    whose lead-in is a broadly-matching character class (`[\\w./\\\\-]+`)
    degrades to O(n^2) on `task` text with no extension anywhere — every
    one of the n starting positions the engine tries costs O(n) to fail,
    since nothing rules a position out early. `task` is user-controlled
    (it flows in verbatim from the turn's own instruction) and unbounded,
    so this path must stay linear regardless of what it's fed."""
    lower_task = task.lower()
    hits: dict[str, int] = {}
    for ext, lang in _EXT_TO_LANG.items():
        start = 0
        while (idx := lower_task.find(ext, start)) != -1:
            end = idx + len(ext)
            if end == len(task) or not (task[end].isalnum() or task[end] == "_"):
                hits[lang] = hits.get(lang, 0) + 1
            start = idx + 1

    if not hits:
        for lang, manifests in _LANG_MANIFESTS.items():
            if any((working_dir / manifest).exists() for manifest in manifests):
                hits[lang] = 1

    if not hits:
        return []

    ranked = sorted(hits, key=lambda lang: (-hits[lang], lang))
    return ranked[:LANGUAGE_BUDGET]


def pump() -> tuple[str, list[str]]:
    """Assemble the escaping directives for the top-`PUMP_BUDGET` registered
    namespaces, ranked by `namespace_directive_rank`. Returns
    (directive_text, domains_used) — domains_used is "" / [] only when no
    namespace directives are registered at all."""
    if not NAMESPACE_DIRECTIVES:
        return "", []
    chosen = sorted(
        NAMESPACE_DIRECTIVES.keys(),
        key=lambda d: (NAMESPACE_DIRECTIVE_RANK.get(d, 100), d),
    )[:PUMP_BUDGET]
    return "\n".join(NAMESPACE_DIRECTIVES[d] for d in chosen), chosen
