"""Hardcoded registry of language-specific craft directives (plan 36 Phase 1).

`LANGUAGE_DIRECTIVES` holds string literals only — no `.md` files, no
`read_text()`, no filesystem source of any kind. A language's directive
text ships inside this module exactly as `namespace_directives` ships
inside each `agent/subagents/*/__init__.py`.

Every entry MUST be mission-free: style and idiom rules only, never "your
job is...", never a process or scope instruction. This text is stacked onto
an existing subagent's role prompt without replacing that role, so a
mission sentence here would compete with the subagent's own mandate and
reintroduce the exact role hierarchy this design avoids. Hold this
constraint when adding the next language key.
"""

from __future__ import annotations

LANGUAGE_DIRECTIVES: dict[str, str] = {
    "python": (
        "write for python 3.12+ — use the newest 3.x syntax you have reliable knowledge of, "
        "favoring clarity and expressive constructs over legacy patterns\n"
        "prefer explicit named parameters; avoid **kwargs except for true pass-through cases such as "
        "decorators and adapters — when used, document every consumed key\n"
        "never use a mutable default (list, dict, set); take None and initialize inside the function\n"
        "do not mutate an input argument unless the function name or its documentation says so; "
        "otherwise return a new object\n"
        "signal errors with specific exceptions, never with a sentinel return value (None, False, -1) "
        "unless the signature explicitly types it\n"
        "annotate every function fully; use Optional[T] only when None carries real semantic meaning, "
        "never as a generic default"
    ),
    "typescript": (
        "write for node.js esm with typescript 6+, using the newest syntax you have reliable knowledge "
        "of, favoring clarity and expressive constructs over legacy patterns\n"
        "avoid any; use unknown for uncertain types and narrow before use\n"
        "do not use the non-null assertion operator (!); handle null and undefined explicitly\n"
        "avoid object, {}, and Record<string, any>; use specific interfaces or type aliases\n"
        "avoid wide unions; use discriminated unions with a type or kind field for explicit handling\n"
        "avoid as casting; use type guards (is) for safe narrowing"
    ),
}
