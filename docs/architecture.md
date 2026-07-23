# Architecture

## Overview

**Plan 27 supersedes plans 25/26.** `delegate`-in-main (a tool main could elect to call) and NL
agent-quoting ("use code-expert to…") are retired outright — no hybrid, no flag gate. Control
flow for any workspace mutation now lives in engineered harness code, not in the core model's
turn-by-turn judgement: **Gate (guard) → Estimator (guard) → Sequencer (CORE thinking, mutation
path only) → fixed interpreter**. The only explicit way to summon a specific agent is `/agent-x`,
which *seeds* the sequencer — it does not dispatch the whole turn to that agent directly.

```
user input
    │
    ▼
Gate                   [supp model, non-thinking] — one call per turn; chit-chat/act guard only
    │                  TRIVIAL | ACT   (REJECTED <name> retired — rejection is now
    │                                   capability-based, at the /agent-x command layer)
    │
    ├── TRIVIAL ─────────────────────────────────────────────────────────────────┐
    │   greeting / identity / general knowledge — answerable with no codebase    │
    │   access                                                                   │
    │                                                                            │
    └── ACT (unknown token also falls to ACT — never silent)                    │
        any request that requires codebase access or workspace action            │
                                                                                 │
    ┌────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
Estimator               [supp model, non-thinking] — trivial-vs-mutate guard (subagent=None only)
    │                    TRIVIAL | MUTATE
    │
    ├── TRIVIAL ──────────────────────────────────────────┐  no ceremony — main solo,
    │   a single small file, a few edits, a read/query     │  no sequencer, deliberately loose
    │                                                       │
    └── MUTATE ─────────────┐                              │
        or an explicit      │                              │
        /agent-x seed       │                              │
                             ▼                              │
                        Sequencer          [core model, CORE thinking — one call]
                        decomposition (main/auto-assignable steps, dependency order)
                        + measurement (preventive verify/repair placement by complexity)
                             │
                             ▼                              │
                        Plan               data — flat list of {agent, task, verify, repair}
                             │                              │
                             ▼                              │
                        Interpreter         fixed, engineered, knows no agent by name
                        execute → verify → repair → re-verify → halt, per step;
                        empty dispatch output is always a failure
                             │                              │
              ┌──────────────┴──────────────┐               │
              │                              │               │
        step.agent == "main"          step.agent == <subagent>
        direct instruction,           run_subagent(...) — cold, fire-and-forget,
        no spawn                      traced on main's session bus
                                                                                 │
    ┌────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
main's session — every spawn and its outcome recorded; a halt reports which step
failed and keeps completed work (no rollback); a mechanical recap is appended to
session history so the next turn is not answered blind
```

---

## Session Message Structure

`Session.messages` is the live context passed to the core model each turn.

```
index  role      content
─────────────────────────────────────────────────────────
  0    system    SYSTEM_PROMPT
  1    system    <workspace> block  (TOON-encoded, injected fresh on startup)
  …    user      prior turns
  …    assistant prior turns
  N    user      current input      (appended before dispatch)
  N+1  assistant response           (appended after handler completes)
```

**System messages** (`[0]`, `[1]`) are excluded from persistence;
they are re-injected fresh on every startup or resume.

**Workspace context injection:** `_init_session()` always passes an empty `workspace` dict to
`agent.start_session()`, which builds the `<workspace>` block via `_format_workspace_context()`
and appends it as `Session.messages[1]`. The block is therefore always injected with
default/unknown values (`workspace_type: "files"`, empty `primary_languages`/`projects`, etc.).

Injected fields — **always:** `workspace_name`, `workspace_type`, `branch`, `primary_languages`,
`projects`; **conditional:** `extensions` (when `projects` is empty), `domain_map` (when present).
Preamble: `"verified repository metadata — treat as authoritative for high-level questions:"`.

---

## System Prompt Assembly

`agent/persona.py` is the neutral module shared by `subagents` and `harness` — extracted to
break an import cycle (`subagents` needs the persona pieces to build its own prompts; `harness`
needs `Subagent`, which lives in `subagents`).

It splits identity from body so a subagent never stacks two competing "you are" assertions:

```
_IDENTITY_MAIN   "you are Gekai, a coding agent…" + capability statement   — main agent only
_IDENTITY_SUB    "you are part of Gekai…"          + tool-neutral capability — subagent only
_SHARED_BODY     meta-rule + <behavior> + <file_handling> + <response_style> + <output_format>
                 — specialization-independent, reused verbatim by both

SYSTEM_PROMPT = _IDENTITY_MAIN + _SHARED_BODY    ← byte-identical to the pre-split constant
```

A subagent is a scoped role *played within* Gekai, not Gekai itself — `_IDENTITY_SUB` keeps
membership ("you're part of the system") without the false claim of being the whole agent, and
states capability tool-neutrally (no hardcoded "you can modify files" that would contradict a
read-only subagent's actual `<tools>` block).

The `<tools>` block is **not** a static constant. `render_tool_instruction(assigned)` in
`persona.py` generates it deterministically (no LLM call) from a fragment table
(`_TOOL_GUIDANCE`) keyed on tool-name groups from `agent/tools/catalog.py`
(`READ_TOOLS`/`EDIT_TOOLS`/`FS_TOOLS`/`SHELL_TOOLS`/`ALL_TOOLS`). Each fragment fires when the
assigned set intersects ("any") or fully contains ("all") its trigger group — so the prompt
never references a tool the agent doesn't actually have. `render_tool_instruction(ALL_TOOLS)`
reproduces the original static instruction string verbatim.

In **direct mode** the `Harness` composes: `SYSTEM_PROMPT + "\n<tools>\n" + render_tool_instruction(<full registered set>)`.

In **spawn mode** the selected `Subagent` builds its own *base* via `build_system_base()`:

```
_IDENTITY_SUB
\n{subagent.mandate}                       ← plain prose role line, e.g. "you act as a …" — only when non-empty
\n{_SHARED_BODY}
\n<directives>\n{subagent.directives}      ← only when subagent.directives is non-empty
```

…and the `Harness` appends the `<tools>` block afterward (see [Harness — Tool Loop](#harness--tool-loop)),
once the *effective* tool set is known. `mandate` is now a bare role-identity sentence
("you act as a code-change specialist…") concatenated directly into the prose — there is no
`<core_mandate>` wrapper tag; `<directives>` (the operational *how*) remains the only
subagent-specific tag. Tags throughout are non-closing (no `</tag>`).

### Dynamic directive pump (plan 28 Phase 3)

`agent/directive_pump.py` generalizes `render_tool_instruction`'s pattern — deterministic,
non-LLM assembly keyed on a detected set — from "tools you have" to "domains this turn needs".
Each namespace package under `agent/subagents/` may declare `namespace_directives` (a shallow,
mission-free craft-text block, e.g. `coding/__init__.py`, `testing/__init__.py`) alongside a
`namespace_directive_rank` int (lower = higher priority). `agent/subagents/__init__.py._discover()`
collects these into two module-level exports, `NAMESPACE_DIRECTIVES` and
`NAMESPACE_DIRECTIVE_RANK`, disjoint from `Subagent.directives` (the deep, mission-presupposing
per-role text a specialist gets from `build_system_base()` — this never escapes to main).

`detect_domains(prompt)` reads backtick-quoted file-path extensions and a small keyword lexicon
out of the raw prompt (no filesystem access, no located-files list — nothing upstream of the
harness populates one yet); `pump(prompt)` intersects the detected domains against
`NAMESPACE_DIRECTIVES`, sorts by rank, and takes the top `PUMP_BUDGET` (2), returning the
concatenated directive text plus the domain list for telemetry. `Harness.stream`/`_stream_graph`
(`agent/harness/core.py`) call this only when `subagent is None`, appending the result as a
`<domain_directives>` block onto `system_base` before `_enrich_system_base`, and emit a
`DirectivePumpEvent` (logged to `events-*.jsonl` as `directive_pump`) whenever a domain was
actually pumped.

---

## Sandbox / Isolation

Tool execution is **not** OS-sandboxed. The current boundary is:

- **Path jail** — `_resolve_in_ws` rejects any path that resolves outside the workspace root (symlinks included via `.resolve()`), and hard-denies `.aiignore`-forbidden paths (see [Workspace Ignore Rules & Hidden-Path Grants](#workspace-ignore-rules--hidden-path-grants)) before any grant logic runs.
- **No shell for file ops** — `read_file`/`write_file`/etc. use `shutil` / `pathlib` directly; no subprocess or shell interpolation surface. `run_command` (the one exec tool) does use a real subprocess, gated by `required_permission="exec"` and a best-effort forbidden-path scan (`_forbidden_token`).
- **Permission gate** — `PermissionGate` enforces `read` / `write` / `exec` per tool call; `exec` permission is not granted by default.

This is proportionate for a single-user local prototype. Full isolation is **deferred** under one explicit assumption: **no exec or network tool exists yet**. The moment either lands, OS-level sandboxing becomes blocking — the path jail is meaningless once arbitrary code runs with user privileges.

---

## Workspace Ignore Rules & Hidden-Path Grants

`agent/workspace/ignore.py` defines a two-tier ignore model, used everywhere the agent walks, lists, searches, or touches files.

```python
class IgnoreRules:
    def __init__(self, working_dir: Path) -> None: ...
    def is_hidden(self, rel_path: str) -> bool: ...
    def is_forbidden(self, rel_path: str) -> bool: ...

def load(working_dir: Path) -> IgnoreRules: ...
```

`IgnoreRules.load(working_dir)` builds two `pathspec.PathSpec` matchers (gitwildmatch syntax) from the workspace's `.gitignore` and `.aiignore`:

| Tier        | Patterns                                                                 | Meaning                                                                 |
|-------------|---------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `hidden`    | builtin floor (`.*`, `node_modules/`, `__pycache__/`, `bin/`, `obj/`) + `.gitignore` + `.aiignore` | Discovery skips these — scanner walk, `list_files`, `grep` without an explicit path, the workspace indexer. |
| `forbidden` | `.aiignore` only                                                           | The "red zone" — every file tool hard-denies these, even via an explicit path. No override possible. |

`load()` is cheap (two small file reads) and is called fresh per operation rather than cached — no invalidation logic exists yet (alpha).

Callers pass POSIX-style relative paths (`/` separators, no leading `/`); directories get a trailing `/` so gitwildmatch directory-only patterns (e.g. `build/`) match correctly.

### Enforcement points

- **Scanner** (`agent/workspace/scanner.py`, `_walk`) — filters `dirnames`/`filenames` in-place via `rules.is_hidden(...)`, replacing the old hardcoded skip-dir set. `list_files`/`list_dirs` and `agent/tools/files.py`'s `_walk_files` (used by `grep`) all go through this.
- **`_list_files`** (`agent/tools/files.py`) — globs the working dir directly and drops any match where `rules.is_hidden(rel)`.
- **File tools** (`agent/tools/files.py`) — every tool resolves its path(s) via `_authorize` (below); `_resolve_in_ws` hard-denies `forbidden` paths regardless of grants.
- **Shell tool** (`agent/tools/shell.py`, `_run_command`) — `_forbidden_token(command, working_dir, rules)` tokenizes the command (`shlex.split`, falling back to `.split()`), strips quotes, also checks the RHS of `key=value` tokens, and resolves each candidate token as a path under `working_dir`. If any resolved token is `forbidden`, `_run_command` returns `"error: command references a restricted path: {rel}"` without running anything. This is best-effort — it does not catch encoded paths, env-var expansion, or other obfuscation — but stops the common case of catting/grepping a red-zone file via `run_command` instead of the file tools.

### `_authorize` — hidden-path grant flow

```python
async def _authorize(
    path: str,
    working_dir: Path,
    allow_hidden: set[str] | None,
    grant_cb: HiddenGrantCallback | None,
    pending: set[str] | None = None,
    *,
    mode: str,
) -> Path | str
```

Every file-tool helper (`_read_file`, `_file_info`, `_grep`, `_edit_file`, `_write_file`, `_symbols`, `_move_file`, `_copy_file`, `_delete_file`, `_make_dir`) routes its path argument(s) through `_authorize`, with `mode="read"` or `mode="write"` depending on the operation. Returns the resolved `Path` on success, or an `"error: ..."` string on failure — callers check `isinstance(result, str)`.

Flow:

1. `_resolve_in_ws(path, working_dir)` — path-jail + `forbidden`-tier hard-deny. Returns `None` (→ `"error: path outside working directory"`) if the path escapes the workspace or matches `.aiignore`. **Forbidden paths are never prompted, with or without a grant.**
2. If the resolved path is `hidden` (and not forbidden):
   - If `rel` is already in `allow_hidden`, proceed — no prompt.
   - Else if `rel` is already in `pending` (a concurrent grant is in flight for the same path), deny immediately: `"error: access to hidden path denied: {rel}"`.
   - Else if `grant_cb is None`, deny: `"error: access to hidden path denied: {rel}"`.
   - Else add `rel` to `pending`, `await grant_cb(rel, mode)`:
     - `False` → deny (same error string); only this tool call fails, no session/worker cancellation.
     - `True` → add `rel` to `allow_hidden` (in-memory) and persist via `save_allow_hidden(working_dir, rel)`.
   - `rel` is removed from `pending` in a `finally` block regardless of outcome.
3. Return the resolved `Path`.

Granted hidden paths remain invisible to discovery tools (`list_files`, `grep` without an explicit `path`) — a grant covers only the explicitly-named path, not directory listings.

### Concurrency: `pending`

`make_file_tools` seeds one shared `pending: set[str] = set()` (mirroring `PermissionGate._pending`) and threads it into every tool closure alongside `allow_hidden` and `grant_cb`. If two concurrent tool calls (e.g. via `asyncio.gather`) target the same un-granted hidden path, the first adds `rel` to `pending` and awaits `grant_cb`; the second observes `rel` already in `pending` and is denied immediately — no double-prompt, no hang.

### Persistence

```python
def load_allow_hidden(working_dir: Path) -> set[str]: ...
def save_allow_hidden(working_dir: Path, rel: str) -> None: ...
```

(`agent/settings.py`) read/write `permissions.allow_hidden` — a JSON array of relative paths — in `.gekai/settings.local.json`, merging with any other existing top-level keys (`permissions.workspace`, `permissions.external`, etc.) in that file. `make_file_tools(working_dir, grant_cb=None)` calls `load_allow_hidden(working_dir)` once at construction to seed the in-memory `allow_hidden` set; subsequent grants within the session update both the set and the file.

### `hidden_grant_callback` wiring

```python
HiddenGrantCallback = Callable[[str, str], Awaitable[bool]]   # (rel_path, mode) -> grant?
```

Defined in `agent/tools/files.py`, re-exported via `agent/tools/__init__.py` and `agent/harness/__init__.py`. Threaded end to end:

```
make_tools(working_dir, grant_cb)              agent/tools/__init__.py
  └─ make_file_tools(working_dir, grant_cb)    agent/tools/files.py

Harness.stream(..., hidden_grant_callback)     agent/harness/core.py   (direct + spawn modes)
  └─ _build_agent(..., hidden_grant_callback)
       └─ make_tools(working_dir, grant_cb=hidden_grant_callback)

GekaiAgent.process_stream(..., hidden_grant_callback, seed)  agent/agent.py
  └─ self._main.stream(..., hidden_grant_callback=hidden_grant_callback, seed=seed)

TUI._hidden_grant_callback(self, rel, mode) -> bool   agent/tui/app.py
  └─ passed as hidden_grant_callback into process_stream(...)
```

The TUI implementation pauses the status timer, asks `"Grant {mode} access to hidden path '{rel}' (excluded by .gitignore)?"` via `_ask_choice` (yes/no), and returns `choice == "y"`. If the worker was already cancelled (`self._worker_cancelled`), it short-circuits to `False` without prompting.

---

## Gate

`Gate` is a **pure chit-chat/act guard** — plan 27 improvement 4 dropped the `REJECTED <name>`
branch entirely. `Gate.gate(user_input, history=None)` makes **one LLM call** on the **support
model (non-thinking)**, temperature 0, and returns a single `Route`.

```python
@dataclass
class Route:
    trivial: bool = False
```

An unrecognized token falls to `Route()` (ACT) rather than silently downgrading — the gate never
suppresses action.

**History context:** last 6 user/assistant turns prepended before the user message.

The gate prompt produces exactly one of two tokens:

```
GATE_PROMPT
├── TRIVIAL          answerable with no codebase access — greetings, identity/capability
│                    questions, acknowledgments, general knowledge unrelated to this
│                    workspace; when unsure, NOT this
└── ACT              any request requiring codebase access or workspace action;
                     unknown/malformed token also falls here — fail to action, not silence
```

`TRIVIAL` is deliberately conservative — a false `ACT` costs only the main agent's time, while a
false `TRIVIAL` would deny a real codebase question any tool access.

**No agent-quoting in prose.** "use code-expert to do X" is not a routing directive — the model
reads it for intent (`do X`) and routes normally; the "use code-expert" mention is ignored, never
name-checked. `detect_subagent_mentions` and the `<subagents_request>` injection (plan 26) are
deleted outright. The only explicit way to summon a specific agent is `/agent-x` (a TUI command,
not a Gate output) — an unknown `/agent-x` is rejected at the command layer, never by the model
naming an agent that doesn't exist.

### Gate table

| Output token         | Route                | Notes                                          |
|-----------------------|----------------------|------------------------------------------------|
| `TRIVIAL`             | `Route(trivial=True)`| Harness non-thinking (`extra_params={}`), no Estimator/Sequencer |
| `ACT`                 | `Route()`             | falls through to the Estimator guard            |
| unexpected/malformed  | `Route()`             | warning logged; falls to ACT                    |

---

## Estimator

`Estimator` is the second guard — a **trivial-vs-mutate binary**, run only when `Gate` returned
`ACT` and the turn is not itself a subagent's own nested run. One LLM call, support model,
non-thinking, temperature 0.

```python
@dataclass
class ScopeEstimate:
    mutate: bool = False
```

| Output   | Meaning                                                        | Next stage         |
|----------|-----------------------------------------------------------------|---------------------|
| `TRIVIAL`| a single small file, a few small edits, or a read/query          | main solo, no sequencer (case 2 — deliberately loose) |
| `MUTATE` | implementation-sized: multiple files/modules, a distinct unit    | Sequencer + interpreter (case 4) |

An explicit `/agent-x` seed (or a single-duty match — case 3) **bypasses the Estimator entirely**
and always goes to the Sequencer, seeded with that agent's duty — one mutation path for both cases
3 and 4 (plan 27 decision 4), no shortcut that skips verification.

---

## Sequencer + Interpreter

Plan 27 replaces `delegate`-in-main with two engineered pieces:

- **Sequencer** (`agent/pipeline/sequencer.py`) — CORE thinking, one call per mutation turn. Given the
  request (and an optional `/agent-x` seed), returns a raw plan text the model itself decomposes
  into ordered steps assigned to `main` or an **auto-assignable** subagent (`code-expert`,
  `test-expert`), each carrying `verify`/`repair` when the model judges the step's complexity
  warrants a preventive check (the complexity *metric* itself remains an open design point — see
  `_agentfiles/27_plan_gate_and_data_plan.md`; today the model's own in-prompt judgment is the
  mechanical placeholder). The raw output is parsed and validated by `agent/pipeline/plan.py`'s
  `parse_plan` — schema, roster, and phase-eligibility checked, fail loud on any violation.
- **Interpreter** (`agent/harness/interpreter.py`) — a **fixed**, engineered step-runner. Walks the
  validated `Plan` with `execute → verify → repair → re-verify → halt`. It knows no agent by name
  or role; `dispatch(agent, task)` is the only seam, supplied by the caller. An empty dispatch
  output is always treated as a failure. A step failing verify once is repaired (the named
  `repair` agent, or a re-dispatch of `step.agent` if none is named) and re-verified; a second
  failure halts the whole plan in place — completed steps' work is kept, nothing rolls back.

`agent/harness/core.py`'s `Harness._stream_plan` wires the two together for production: `dispatch`
runs `main` steps as a direct instruction (no spawn) via the same `_build_agent` used elsewhere,
and subagent steps via `run_subagent` (`agent/tools/delegate.py`) — a cold, fire-and-forget nested
run whose start/outcome are traced on the same `EventBus` the single-agent path uses, so
diff/log/delegation rendering is shared, not reimplemented. A plan run ends with a **mechanical**
(no narrator LLM call) recap of every step's outcome, yielded as the turn's assistant text so it
persists into session history for the next turn's continuity.

---

## Subagent Routing

### Subagent

Frozen dataclass defined in `agent/subagents/` package — not user-definable. It is a real entity
that owns the assembly of its own system prompt:

```python
@dataclass(frozen=True)
class Subagent:
    name: str
    namespace: str
    description: str       # sequencer's selection signal — positive scope + "Not for…" boundary
    mandate: str = ""      # 1-2 line role-identity sentence: "you act as a …"
    directives: str = ""   # the *how* — operational specifics
    tools: list[str] | None = None        # allowlist; None = all tools
    permissions: Permissions | None = None # overlay, ANDed with session permissions
    user_invocable: bool = True    # router menu + /agent-x eligibility; False = system-managed worker
    auto_assignable: bool = False  # plan 27: phase-1 decomposition may assign it;
                                    # False = post-planning-only (verify/repair agents, e.g. the
                                    # coming fact-checker) — never assigned by prompt decomposition

    def build_system_base(self) -> str: ...  # _IDENTITY_SUB + mandate + _SHARED_BODY + <directives>
                                              # <tools> appended later by the Harness
```

`description` appears in the sequencer's roster string (`name: description`) and carries a "Not for…"
boundary clause to sharpen decomposition decisions. `mandate` is the role-identity sentence fed
into the agent's own context once spawned — "who you act as right now," distinct from
`directives` ("how to do it"). `tools` is a name allowlist (mirrored against
`agent/tools/catalog.py` to guard against drift); `permissions` lets a subagent further
*restrict* — never escalate beyond — the session's grant. `auto_assignable` is orthogonal to
`user_invocable`: `code-expert`/`test-expert` are both (sequencer may assign them, `/agent-x` can
seed them); a verify/repair agent is `user_invocable=True, auto_assignable=False` (slash-summonable,
but never assigned by phase-1 decomposition). See [System Prompt Assembly](#system-prompt-assembly)
for the full funnel and [Harness — Tool Loop](#harness--tool-loop) for how the allowlist and
permissions jointly determine the *effective* tool set (and therefore the `<tools>` prompt content).

Package layout: `__init__.py` exports `Subagent`, `SUBAGENTS`, `NAMESPACES`, `validate_registry`,
and `_discover()` auto-discovery.
One file per subagent — currently just `code_expert.py`.
Namespace-level shared directives live in `_coding.py` (namespace = `"coding"`) — composed into
`subagent.directives` at import time via `dataclasses.replace`.
Adding a subagent = drop one file; zero other changes required.

`NAMESPACES` includes `"coding"`, `"testing"`, and `"generic"` (innate — no subagents; selector
skipped). There is no fallback subagent: each subagent stands on its own `description`. `main` is the
**generalist default** — general or simple requests, reading/explaining/running code, and
and all general/glue/scaffolding work in a plan; pick a subagent only when
the request clearly fits its specialty. `code-expert` handles **code work by kind** —
features, fixes, and behavior-changing rewrites where the approach is decided; it owns its assigned step's implementation in full;
its `description` excludes general/scaffolding/glue work (that's `main`) and
pure refactors / complexity-reduction passes with no behavior change (those go to `code-refactorer`).

`validate_registry()` runs at startup — raises if any namespace in `NAMESPACES` has no badge
color, a subagent has an unknown namespace, or names collide.

Subagent selection is performed by the Sequencer (phase-1 decomposition, auto-assignable roster
only) — see [Sequencer + Interpreter](#sequencer--interpreter) above — or explicitly via `/agent-x`
(seeds the sequencer; unknown `/agent-x` rejected at the command layer).

> **Subagent vs harness-worker:** `Subagent` serves a *user-turn* — it is spawned from user
> intent via routing. A separate, still-unbuilt `harness-worker` category would instead serve
> the *system/lifecycle* directly — work that runs outside any single user turn. Zero code
> exists for that category yet; the name marks the conceptual slot.

### Permission overlay

`subagent.permissions` is **ANDed** with `session.permissions` per field — a subagent can restrict but never escalate beyond what the session granted.

```python
effective = Permissions(
    read  = session.read  and subagent.read,
    write = session.write and subagent.write,
    exec  = session.exec  and subagent.exec,
)
```

### TUI status indicator

`#input-area` container `border_title` shows the current pipeline stage as it advances
(`route` → `main`/sequencer-badge names as `SubAgentStartEvent`/`DelegationStartEvent` arrive), not
a single per-turn label keyed on `route.subagent` (that field is gone — plan 27 improvement 4).
A `/agent-x` seed no longer means "the whole turn runs as that subagent's identity"; it seeds the
sequencer, so the top-level label stays `main`/`sequencer` and individual plan steps render their own
specialist badges as the interpreter dispatches them.

---

## Harness — Tool Loop

`Harness` (`agent/harness/core.py`) is the single, always-on session holder users always talk to.
`Harness.stream(...)` is the entry point for every turn; its shape now branches on the
guard/estimate/seed outcome (see [Overview](#overview)) rather than on a bare `subagent` toggle:

```python
async def stream(
    self, session, user_input, permission_callback=None,
    subagent: Subagent | None = None, extra_params: dict | None = None,
    hidden_grant_callback=None, seed: str | None = None,
)
```

- `subagent=None, seed=None`, trivial estimate (or no estimator wired) → single-agent path (below).
- `subagent=None`, mutate estimate **or** `seed` given → `_stream_plan`: Sequencer + interpreter
  (see [Sequencer + Interpreter](#sequencer--interpreter)).
- `subagent=<Subagent>` → single-agent path built *as* that subagent — used internally by
  `run_subagent` for a nested dispatch, not a production top-level entry point any more (that role
  moved to `seed`).

`extra_params` overrides the Harness's own `self._extra_params` (set at construction from
`resolve_thinking_params`) for this call only — `None` (the default) means "use the instance
default"; an explicit `{}` means "no thinking params for this turn" (the `TRIVIAL`-route case —
see [Gate](#gate)).

No `delegate` tool is ever registered (plan 27 decision 11 — removed from main outright, no
hybrid); the harness (sequencer + interpreter) owns all cross-agent control flow instead.

`stream()` delegates prompt assembly to the module-level `_build_agent(...)`, which is the
**single point** where the effective tool set — and therefore the final `<tools>` block — is
computed, for both modes:

```
1. compute `effective` permissions = session.permissions ANDed field-wise with subagent.permissions (spawn mode only)
2. build `selected`: walk make_tools(working_dir), drop any tool not in subagent.tools (when an allowlist is set),
   then drop any tool whose required_permission isn't granted by `effective` (when there's no permission_callback to escalate)
3. system = system_base + "\n<tools>\n" + render_tool_instruction([t.name for t in selected])
4. construct Agent(system=system, extra_params=effective_extra_params, …), register exactly the
   `selected` tools, attach the PermissionGate
```

| Mode                       | Trigger              | `system_base`                  | Prior context                                  |
|----------------------------|----------------------|--------------------------------|------------------------------------------------|
| **direct**                 | `subagent=None`      | `SYSTEM_PROMPT`                | `_recency_turns(session.messages, _RECENCY_N=2)` — last 2 user/assistant pairs + current input |
| **spawn**                  | `subagent=<Subagent>`| `subagent.build_system_base()` | cold — `prior = []` + current input only; no context inheritance, no async/resume (fire-and-forget by design, for now) |

The `<tools>` block therefore always reflects the *effective, post-filtering* tool set — never
the subagent's bare declared allowlist — so the activation prompt never references a tool the
agent can't actually call (e.g. a read-only subagent's prompt omits all shell/edit guidance).

`SubAgentStartEvent` carries `name="main"` / `description="thinking"` in direct mode, or
`subagent.name` / `subagent.description` in spawn mode — both paths emit it; both stream through
the same unified event flow.

`_recency_turns(session.messages, _RECENCY_N)` extracts the last `_RECENCY_N=2` user/assistant
pairs (skipping system messages, excluding the current trailing user input); the current user
input is then appended explicitly.

When `--debug` is active, `stream()` calls
`append_debug(session, {"content": {"system": agent.system, "extra_params": effective_extra_params}})`
**after** `_build_agent` returns — `agent.system` is the true, fully-assembled prompt string
(mutable field on `llmstitch.Agent`), and lands in `.debug.jsonl` alongside the workspace
context block. `extra_params` here is the *effective* value actually passed to the model for
this turn (`{}` for `TRIVIAL` routes), not the Harness's instance default.

```mermaid
sequenceDiagram
    participant TUI
    participant Harness
    participant llmstitch
    participant CoreModel

    TUI->>Harness: stream(session, user_input, extra_params={} if route.trivial else None, seed=forced_seed)
    Note over Harness: single-agent path (trivial estimate, or no mutate/seed) — the<br/>mutate/seeded path instead runs Sequencer + interpreter (_stream_plan)
    Harness->>Harness: _build_agent(...) — filter tools, render <tools>, construct Agent
    Harness->>llmstitch: agent.run(prior_messages)
    loop tool-calling
        llmstitch->>CoreModel: messages + tools
        CoreModel-->>llmstitch: ToolUseBlock / TextBlock
        llmstitch-->>Harness: ToolExecutionStarted event
        Harness-->>TUI: LogEvent (e.g. "Edit src/main.py")
        llmstitch->>llmstitch: execute tool, append result
        llmstitch-->>Harness: ToolExecutionCompleted event
        Harness-->>TUI: DiffEvent (edit_file only; old_str vs new_str)
    end
    llmstitch-->>Harness: final history
    Harness-->>TUI: DoneEvent → text response
```

Events flow through `EventBus` → async queue → TUI stream. The final answer text is yielded
once, after `DoneEvent` — there is no live token-by-token streaming to the user (the `Agent`
loop's internal streaming is only used to simplify token accounting via `InferEndEvent`).

---

## Session Persistence

Sessions stored at `~/.gekai/workspaces/{normalized-repo-path}/{session-id}.jsonl`.
Every entry is timestamped JSON with a `kind` field:

```
{ts, kind:"turn",    role:"user|assistant|system", content}   ← LLM context only
{ts, kind:"command", content:"/clear"}                        ← slash command typed by user
{ts, kind:"event",   source:"...", content:"..."}             ← system-side, non-LLM
```

Event sources: `gate` · `error` · `interrupted` · `farewell` · `max_iterations` · `command` (command result).
Entries without `kind` (legacy) default to `"turn"`.

**Boundary:** `session.jsonl` = everything the user saw on screen. `{session-id}.debug.jsonl` = internal plumbing (system prompts, route tokens, per-turn debug context) — `--debug` only.
Litmus: *did the user see it on screen?* → session; *did only the developer need it?* → debug.

**Two readers:**
- `load_session(id)` → turns only — model context for resuming the LLM conversation.
- `load_timeline(id)` → full ordered list — drives visual chat rebuild on `--resume`.

**Max-iterations:** on agent hit with no text, writes `event(source="max_iterations")` instead of an empty assistant turn. Context stays clean; rebuild shows the warning.

**`/clear` is the first entry of the new session** — `/clear` is persisted to the *new* session immediately after creation, marking it visually at the top. The old session retains the `/clear` command entry too (no result entry, since execution returns early).
