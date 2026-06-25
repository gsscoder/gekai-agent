# Architecture

## Overview

Every turn passes through three layers: **routing → file location → Harness dispatch**.
`TRIVIAL` routes skip the middle layer entirely — no file location, no rewrite. `EXPLORE` routes go to a
separate read-only `FileExplorer`, not shown in the single-step diagram below (see
[Multi-step Plans](#multi-step-plans) for `<plan>` routes).

```
user input
    │
    ▼
Router                 [support model] — one call; emits a single Route
    │                  (trivial | explore | subagent | main | plan)
    │
    ├── EXPLORE ───────────────────────────────────────────────────────► FileExplorer (read-only investigation)
    │
    ├── TRIVIAL ────────────────────────────────────────────────────┐
    │   greeting / identity / general knowledge —                   │
    │   answerable with no codebase access                          │
    │                                                               │
    ├── main (Route(), no subagent) ────────┐                       │
    │                                       │                       │
    └── <subagent> ─────────────┐           │                       │
                                │           │                       │
                                ▼           ▼                       │
                       FileLocator   [support model] — locate       │
                       relevant/affected files (path | keywords)    │
                                │                                   │
                       entries non-empty? ── no ────────────────────┤
                                │ yes                               │
                                ▼                                   │
                       PromptRewriter  [core model, non-thinking] — │
                       weave located paths into the request         │
                                │                                   │
                                └───────────────────┬───────────────┘
                                                    ▼
                                          Harness    [core model]
                                            direct mode (no subagent, with recency)
                                            spawn mode  (subagent.build_system_base() + harness-assembled <tools>, cold)
                                            trivial route → extra_params={} (no thinking budget)

    └── <plan> (Route(plan=[PlanStep, ...])) ───► N × (locate → rewrite → Harness), one step
                                                   at a time, sequential, fail-stop — see below
```

### Multi-step plans

When a request clearly needs multiple *different* specialists run in order, the same router call
emits a `<plan>` block instead of a single token — no extra LLM round-trip. A single specialist
task phrased with multiple clauses still collapses to one token (router bias is hard toward the
single-token case); a parsed plan with exactly one step also collapses to the equivalent
single-token `Route`.

```python
@dataclass
class PlanStep:
    subagent: Subagent | None   # None => this step runs on main
    raw: str                    # verbatim slice of the request this step covers
```

Executor (`agent/tui/app.py::_run_step`, looped over `route.plan`): the same locate → rewrite
→ dispatch pipeline used for a single-token route, run once per `PlanStep`, in order. Steps are
**sequential and fail-stop, with no revert** — a step that hits the iteration budget with no answer,
or errors, stops the whole plan immediately; completed steps' work (files
written, etc.) is left in place, and the chat surfaces
`"step {k} of {n} failed: {reason}; completed steps 1..{k-1}"`. On full success, one trailing
`"* {verb} for {duration} ({n} steps)"` operation line closes the turn. Each step that produces an
answer is rendered as its own assistant message — N assistant messages can follow a single user
turn. All N steps + the one user turn persist under a shared `turn_id` (`process_stream(...,
append_user=...)`: `True` for step 1, `False` for steps 2..N).

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

`agent/persona.py` is the neutral module shared by `subagents`, `pipeline.router`, and
`harness` — extracted to break an import cycle (`subagents` needs the persona pieces to build
its own prompts; `router`/`harness` need `Subagent`, which lives in `subagents`).

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

FileExplorer.stream(session, user_input, hidden_grant_callback)  agent/harness/file_explorer.py
  └─ make_tools(session.working_dir, grant_cb=hidden_grant_callback)  (read-only subset)

GekaiAgent.process_stream(..., hidden_grant_callback)  agent/agent.py
  ├─ route.explore  → self._explorer.stream(..., hidden_grant_callback=hidden_grant_callback)
  └─ main           → self._main.stream(..., hidden_grant_callback=hidden_grant_callback)

TUI._hidden_grant_callback(self, rel, mode) -> bool   agent/tui/app.py
  └─ passed as hidden_grant_callback into process_stream(...)
```

The TUI implementation pauses the status timer, asks `"Grant {mode} access to hidden path '{rel}' (excluded by .gitignore)?"` via `_ask_choice` (yes/no), and returns `choice == "y"`. If the worker was already cancelled (`self._worker_cancelled`), it short-circuits to `False` without prompting.

---

## Router

`Router` is a pure **guard**, not an intent classifier. `Router.route(user_input, history=None)`
makes **one LLM call** on the **support model**, temperature 0, and returns a single `Route`.

```python
@dataclass
class Route:
    subagent: Subagent | None = None
    trivial: bool = False
    explore: bool = False
    plan: list[PlanStep] | None = None
```

There is no `namespace` property — callers read `route.subagent.namespace` directly when
`route.subagent is not None`. `trivial`/`explore`/`subagent`/`plan` are mutually exclusive in
practice — the router maps each single token before checking the subagent menu, and a `<plan>`
block is only emitted instead of (never alongside) a token.

**History context:** last 6 user/assistant turns prepended before the user message.

The router prompt offers five kinds of output:

```
ROUTER_PROMPT
├── TRIVIAL          answerable with no codebase access — greetings, identity/capability
│                    questions, acknowledgments, general knowledge unrelated to this
│                    workspace; when unsure, NOT this
├── EXPLORE          read-only investigation ending in an answer about files/structure;
│                    never chosen if the request also asks for an edit/fix/change
├── <subagent-name>  one of the subagents in the menu (built from SUBAGENTS,
│                    "name — description"); when the request fits its specialty,
│                    or explicitly asks to use/delegate the task to it by name
├── <plan>           the request clearly needs multiple *different* specialists run in
│                    order — see Multi-step Plans above
└── main             anything else — handled directly by Harness
```

`main` is listed **last** with no explicit "host-retained"/"bias" line: the choices are
self-defining and position signals that `main` is host-retained by default. Terminology is uniform
(`subagent`, never "specialist") so the model reads one concept, not two. `TRIVIAL` is
deliberately conservative — the prompt tells the model to prefer `main` when unsure, since a
false `main` only costs one extra (often near-empty) `FileLocator` call, while a false
`TRIVIAL` would deny a real codebase question its file context.

### Routing table

| Output token        | Route                  | Notes                                                |
|---------------------|------------------------|------------------------------------------------------|
| `main`              | `Route()`              | Harness handles directly, no subagent spawned        |
| `TRIVIAL`           | `Route(trivial=True)`  | skips FileLocator + PromptRewriter; Harness still answers, with `extra_params={}` |
| `EXPLORE`           | `Route(explore=True)`  | dispatched to `FileExplorer`, a separate read-only investigation path |
| `<subagent-name>`   | `Route(subagent=p)`    | matched subagent spawned                             |
| `<plan>` (2+ steps) | `Route(plan=[...])`    | multi-step executor, see Multi-step Plans above      |
| `<plan>` (1 step)   | `Route(subagent=p)`    | collapses to the equivalent single-token route       |
| unknown/malformed   | `Route()`              | warning log + host-retained, same as `main`          |

---

## File Location

Runs on every **non-`TRIVIAL`** route — `main` and `<subagent>` alike. `TRIVIAL` routes skip
this step entirely (`entries = []`, no locate call).

### Locate step

`FileLocator.locate(working_dir, request)` — agentic support-model call (up to 5 iterations, read-only tools).
Returns `list[tuple[path, keywords]]`.

Any exception propagates — there is no fail-open; a broken locate blocks the turn.

---

## Prompt Rewriter

Runs whenever `FileLocator` returned **non-empty entries** — for `main` and `<subagent>` routes
alike. `TRIVIAL` routes, and any route where the locator found nothing, skip it.

`PromptRewriter.rewrite(request, entries) -> tuple[str, str]` — a single **core-model** call, temperature 0,
**no thinking params** (non-thinking even on a reasoning-capable core model). Not agentic, no tools:
the locator already discovered and verified the files, so this stage only *attributes* them.

It takes the original request plus the located `path | keywords` list and returns
`(rewritten_request, ui_label)`:

```
1. weave the full path inline wherever a file clearly maps to a phrase in the request
   "update the passcode dialog to allow 8 chars" + "src/app/auth/passcoder.tsx | passcode, dialog"
   -> "update 'src/app/auth/passcoder.tsx' to allow 8 chars"
2. located files it cannot confidently attribute go in a trailing <reference_files> block
3. paths are quoted verbatim from the list — never invented or altered
```

**Fail-hard:** `rewritten_request` keeps the locator's contract — any exception (including empty
output → `ValueError`) propagates and blocks the turn, same as `FileLocator`. `ui_label` is
**fail-soft** — a missing/malformed label degrades to `""` rather than blocking the turn; it is
cosmetic (UI badge text), not load-bearing.

**Original vs processed input:** the rewritten string is fed to `Harness` as the live user
turn (`process_stream`'s `user_input`), while the user's verbatim text is passed as `original_input`
and is what gets **persisted and displayed**. Later recency windows therefore show the user's real
phrasing, not the rewrite. When `--debug` is active the rewritten text is written to `.debug.jsonl`.

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
    description: str       # router's selection signal — positive scope + "Not for…" boundary
    mandate: str = ""      # 1-2 line role-identity sentence: "you act as a …"
    directives: str = ""   # the *how* — operational specifics
    tools: list[str] | None = None        # allowlist; None = all tools
    permissions: Permissions | None = None # overlay, ANDed with session permissions
    is_fallback: bool = False

    def build_system_base(self) -> str: ...  # _IDENTITY_SUB + mandate + _SHARED_BODY + <directives>
                                              # <tools> appended later by the Harness
```

`description` doubles as the router's menu entry (`name — description`) and carries a "Not for…"
boundary clause to sharpen the router's selection. `mandate` is the role-identity sentence fed
into the agent's own context once spawned — "who you act as right now," distinct from
`directives` ("how to do it"). `tools` is a name allowlist (mirrored against
`agent/tools/catalog.py` to guard against drift); `permissions` lets a subagent further
*restrict* — never escalate beyond — the session's grant. See
[System Prompt Assembly](#system-prompt-assembly) for the full funnel and
[Harness — Tool Loop](#harness--tool-loop) for how the allowlist and permissions jointly
determine the *effective* tool set (and therefore the `<tools>` prompt content).

Package layout: `__init__.py` exports `Subagent`, `SUBAGENTS`, `NAMESPACES`, `validate_registry`,
and `_discover()` auto-discovery (plus the internal `_by_namespace` index it relies on).
One file per subagent — currently just `code_expert.py`.
Namespace-level shared directives live in `_coding.py` (namespace = `"coding"`) — composed into
`subagent.directives` at import time via `dataclasses.replace`.
Adding a subagent = drop one file; zero other changes required.

`NAMESPACES` includes `"coding"`, `"testing"`, `"generic"` (innate — no subagents; selector
skipped), and `"worker"`. The `coding`-namespace fallback is `code_expert` (`is_fallback=True`):
general code changes — features, fixes, tests — when no specialized subagent fits; its
`description` explicitly excludes pure refactors / complexity-reduction passes with no behavior
change. `worker`'s sole member, `ws-manager` (`agent/subagents/worker/ws_manager.py`), is now
`user_invocable=True` and `is_fallback=True` — the router/plan can assign it like any other
subagent. Its mandate is scoped to repo/filesystem scaffolding only (project skeletons,
directories, manifest files, conventional layout) — never application logic.

`validate_registry()` runs at startup — raises if any non-`generic` namespace in `NAMESPACES` has
no members, has duplicate names, or doesn't have exactly one fallback.

Subagent selection is performed by the `Router` in a single guard call — see [Router](#router) above.

> **Subagent vs harness-worker:** `Subagent` serves a *user-turn* — it is spawned from user
> intent via routing. `ws-manager` (`worker` namespace) is the first built, user-invocable example
> of a system-flavored member, scoped to repo/filesystem scaffolding only. A separate,
> still-unbuilt `harness-worker` category would instead serve the *system/lifecycle* directly
> (e.g. a future workspace-scan revival) — work that runs outside any single user turn. Zero code
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

`#input-area` container `border_title` always shows current routing state:

| State                     | Label shown              | Border background color                        |
|---------------------------|--------------------------|------------------------------------------------|
| idle / between turns      | `default`                | `#3a3a3a`                                      |
| `route.subagent is not None` | `<namespace>/<name>`  | `_NS_COLORS.get(namespace, _ACTION_COLOR)` — `#FFD700` for `coding`, `#4169E1` otherwise |
| `route.subagent is None`  | `main` (incl. `TRIVIAL`) | `#3a3a3a` (`_DEFAULT_ROUTE_COLOR`)             |
| finally (any exit)        | `default`                | `#3a3a3a`                                      |

Debug label (`--debug`, shown as `[router: ...]`): `[subagent.namespace, subagent.name]` joined
by `/` when a subagent is selected; `["trivial"]` when `route.trivial`; else `["main"]`. The
status-indicator border title itself does not distinguish `TRIVIAL` from `main` — both show
`main` / `#3a3a3a`; only the debug label surfaces the distinction.

---

## Harness — Tool Loop

`Harness` (`agent/harness/core.py`, formerly `MainAgent` in `handlers/main_agent.py`) is the
single, always-on session holder users always talk to. Every turn that doesn't spawn a subagent
is handled directly by it — chat, inspection, and action all flow through one `Agent`-tool-loop.
One method, two modes selected by a single parameter:

```python
async def stream(self, session, user_input, permission_callback=None, subagent: Subagent | None = None, extra_params: dict | None = None)
```

`extra_params` overrides the Harness's own `self._extra_params` (set at construction from
`resolve_thinking_params`) for this call only — `None` (the default) means "use the instance
default"; an explicit `{}` means "no thinking params for this turn" (the `TRIVIAL`-route case —
see [Router](#router)).

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

    TUI->>Harness: stream(session, user_input, subagent=route.subagent, extra_params={} if route.trivial else None)
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

Event sources: `router` · `error` · `interrupted` · `farewell` · `max_iterations` · `command` (command result).
Entries without `kind` (legacy) default to `"turn"`.

**Boundary:** `session.jsonl` = everything the user saw on screen. `{session-id}.debug.jsonl` = internal plumbing (system prompts, route tokens, locate list, rewritten text) — `--debug` only.
Litmus: *did the user see it on screen?* → session; *did only the developer need it?* → debug.

**Two readers:**
- `load_session(id)` → turns only — model context for resuming the LLM conversation.
- `load_timeline(id)` → full ordered list — drives visual chat rebuild on `--resume`.

**Max-iterations:** on agent hit with no text, writes `event(source="max_iterations")` instead of an empty assistant turn. Context stays clean; rebuild shows the warning.

**`/clear` is the first entry of the new session** — `/clear` is persisted to the *new* session immediately after creation, marking it visually at the top. The old session retains the `/clear` command entry too (no result entry, since execution returns early).
