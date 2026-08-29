# Architecture

## Overview

**Plan 27 supersedes plans 25/26; plan 33 folds the old two-guard `Gate → Estimator` design into
one.** `delegate`-in-root (a tool root could elect to call) and NL agent-quoting ("use code-expert
to…") are retired outright — no hybrid, no flag gate. Control flow for any workspace mutation lives
in engineered harness code, not in the core model's turn-by-turn judgement: a single **Estimator**
call (FAST-tier model, non-thinking) classifies every turn onto an ordinal 3-rung scale, `CHAT ⊂ SOLO
⊂ MUTATE`, and only `MUTATE` reaches the **Sequencer** (CORE tier, thinking explicitly off) and the fixed interpreter.
`Gate`/`Route` no longer exist anywhere in the codebase — plan 33 merged the old chit-chat/act axis
into the Estimator's own `CHAT` rung (an unrecognized token falls to `SOLO`, never silent — the same
"never fail to silence" property `Gate`'s old ACT-fallback had, just at the new safe-middle rung).
The only explicit way to summon a specific agent is an invocable subagent's own slash alias (e.g.
`/refactor`, `/fix`, `/build`) — this `seed` dispatches the whole turn directly to that agent,
bypassing the Estimator, the Sequencer, and the task graph entirely, not "seeding" the sequencer's
decomposition.

```
user input
    │
    ├── explicit /<alias> seed (e.g. /refactor, /fix, /build) — bypasses the Estimator ────┐
    │                                                                                        │
    ▼                                                                                        │
Estimator        [FAST-tier model, non-thinking] — one call per turn (subagent=None only)    │
    │             CHAT | SOLO | MUTATE — unparseable output, or a call failure, falls to      │
    │             SOLO, never silent: over-estimating from CHAT just keeps the codebase       │
    │             available (harmless); under-estimating from MUTATE just skips planning      │
    │             (still solo-capable)                                                        │
    │                                                                                          │
    ├── CHAT ── tools_override=frozenset() (no tool schemas), thinking off ──────┐            │
    │   greeting / identity / general knowledge — no codebase access needed      │            │
    │                                                                             │            │
    ├── SOLO ── full tool set, normal effort/thinking ───────────────────────────┤            │
    │   a single small file, a few small edits, or a read/query                  │            │
    │                                                                             │            │
    └── MUTATE ──────────────┐                                                   │            │
        multi-file/module,   │                                                   │            │
        or a distinct unit   ▼                                                   │            │
        of work         Sequencer   [CORE-tier model, thinking off — one call]   │            │
                         decomposition (auto-assignable steps only, dependency    │            │
                         order) + measurement (preventive verify/repair          │            │
                         placement by complexity)                                │            │
                              │                                                  │            │
                              ▼                                                  │            │
                         TaskGraph   data — summary + flat list of               │            │
                                     {agent, instruction, verify, repair, scope} │            │
                              │                                                  │            │
                              ▼                                                  │            │
                         interpreter   fixed, engineered, knows no agent by      │            │
                         name; root is never a step agent. execute → verify →   │            │
                         repair → re-verify → halt, per step; empty dispatch    │            │
                         output is always a failure. an invalid graph           │            │
                         (ValueError) falls back to the SOLO path below         │            │
                              │                                                  │            │
    ┌─────────────────────────┴──────────────────────────────────────────────────┴────────────┘
    │
    ▼
root's session — every spawn and its outcome recorded; a halt reports which step failed and
keeps completed work (no rollback); root then synthesizes the user-facing answer (`_respond`,
a real LLM call, not a mechanical recap — see Sequencer + Interpreter below)
```

---

## Session Message Structure

`Session.messages` is the live context passed to the core model each turn.

```
index  role      content
─────────────────────────────────────────────────────────
  0    system    ROOT_SYSTEM_PROMPT
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
_IDENTITY_ROOT   "you are Gekai, a coding agent…" + capability statement   — root only
_IDENTITY_SUB    "you are part of Gekai…"          + tool-neutral capability — subagent only
_SHARED_BODY     meta-rule + <behavior> + <file_handling> + <response_style> + <output_format>
                 — specialization-independent, reused verbatim by both

ROOT_SYSTEM_PROMPT = _IDENTITY_ROOT + _SHARED_BODY + "\n<directives>\n" + _ROOT_DIRECTIVES
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

In **direct mode** the `Harness` composes: `ROOT_SYSTEM_PROMPT + "\n<tools>\n" + render_tool_instruction(<full registered set>)`.

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
per-role text a specialist gets from `build_system_base()` — this never escapes to root).

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

## GEKAI.md & Foreign Instruction Files

Plan 35 (v3) gives Gekai two distinct ways a project can hand it standing instructions, and one
shared, advisory-only check that asks a single yes/no question about either.

**`GEKAI.md` — ingested.** `Session.gekai_md: IngestedFile | None` (`agent/session.py`) is read
once, at session start (including after `/clear`), from the workspace root
(`GekaiAgent._read_gekai_md`, `agent/agent.py`) — a missing file or a read failure is telemetry
and a skip, never a crash. When present, `_gekai_md_system_base()` (`agent/harness/core.py`)
appends the file **verbatim** to root's system base, under its own tag:

```
<project_instructions source="GEKAI.md">
…file text, byte-for-byte…
```

This happens beside the dynamic-directive pump append, **root-only** — a spawned subagent never
sees it, mirroring the pump's own root-only rule (plan 28 decision 13). The system base is chosen
over a simulated "read and understand" turn in message history specifically because message
history is what `/compact` evicts; a rule seeded there would silently stop applying somewhere
around turn 40 with no signal to anyone.

**Foreign instruction files (`AGENTS.md`, `CLAUDE.md`, `.cursorrules`, …) — merely read.** There is
no dedicated command and no `Session` field for these — no `/ingest` command exists, and never did
in shipped code. When the user asks Gekai to read one, root calls the ordinary `read_file` tool
(`agent/tools/files.py`) and the file becomes a tool result in message history, exactly like
reading any other file — it is compaction fodder, on purpose: the user asked for it once, for
context, not as standing law.

**The shared check — one question, no prefilter, no corpus.** Both paths feed the same async,
advisory-only check, `agent/directive_audit.py`'s `Auditor` — a single yes/no question asked of the
file's text alone: *does this file contain rules, instructions, or directives intended to influence
how an AI coding agent behaves?* There is no corpus, no comparison against Gekai's own directives,
no classification, no counting — a file may describe an AI product or mention agents throughout and
still answer `NO`; what matters is whether the file addresses the assistant reading it, not its
subject matter.

- GEKAI.md always triggers the check — it is Gekai's own file, instruction-file by definition.
- A foreign file triggers the check on **every** root-dispatched `.md` `read_file` result — v3
  deletes the regex prefilter that used to gate this. `_maybe_flag_foreign_instruction_file`
  (`agent/harness/core.py`, fired from `ToolExecutionCompleted` handling) gates only on: the read
  was root-dispatched (never a subagent's own incidental read), the path ends in `.md`, and the
  read succeeded — routing, not content judgment. A `.py`/`.json` read never reaches the check.
- The call runs at the `directive-audit` touchpoint — plain **FAST tier, no policy or effort
  override** (`agent/harness/touchpoints.py`), the same shape `estimator`/`micro` already use: a
  one-word answer needs no reasoning model. `AuditVerdict(has_directives: bool, raw: str)` — a
  first-token parse mirroring `estimate.py`; any unexpected output, or any call failure, falls back
  to `NO`, the safe, silent answer.
- Results cache at `.gekai/directive-audit.json`, keyed on `file_sha` alone — there is no corpus
  any more to invalidate against. A `directive_audit.enabled` setting (`agent/settings.py`, default
  on) turns the whole check off without touching GEKAI.md ingestion.
- `GekaiAgent.start_directive_audit()` (the GEKAI.md path) and `start_foreign_file_audit()` (the
  foreign-file path) both funnel into a shared private `_start_audit()` — same cache check, fire,
  swallow-all-failures, and telemetry flow either way.

**The verdict never enters any model's context.** It has exactly one consumer — the human, via a
single `#directive-notice` TUI slot and the `directive_audit` telemetry event. One slot, one line,
last verdict wins:

| file      | verdict    | slot shows                                |
|-----------|------------|--------------------------------------------|
| GEKAI.md  | in flight  | grey `⋯ checking GEKAI.md`                  |
| GEKAI.md  | YES        | yellow `⚠  GEKAI.md contains agent directives` |
| GEKAI.md  | NO         | green `✓ GEKAI.md loaded` (persists)        |
| foreign   | in flight  | grey `⋯ checking <path>`                    |
| foreign   | YES        | yellow `⚠  <path> contains agent directives` |
| foreign   | NO         | nothing                                     |

GEKAI.md always leaves a mark because its read is invisible — silence on a clean file would be
indistinguishable from the file not existing or the check being broken. A foreign file stays silent
on `NO` because the user asked for that read and watched it happen; a confirmation there would be
noise. The slot is one line, last-verdict-wins — not stacked, not per-file — so a foreign file's
verdict landing after GEKAI.md's own warning silently overwrites it; this is a known, accepted
limitation (two different lifetimes sharing one slot, with nothing in the notice itself
distinguishing which is which), not a bug.

This is a **linter, not a guardrail**: it reports and stops there. It never blocks a turn, never
strips or rewrites the file, and never changes what any model sees — the enforcement surface a
stronger control would otherwise protect is already held mechanically, by the permission system
(`agent/permissions.py`, `agent/settings.py`).

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

## Estimator

`Estimator` (`agent/pipeline/estimate.py`) replaces the old two-guard `Gate → Estimator` design
(plan 33) with a single guard: one LLM call, FAST-tier model, non-thinking, temperature 0, run only
when `subagent is None` and no explicit `seed` was given. It classifies the turn onto an ordinal
3-rung scope, not a binary — `CHAT ⊂ SOLO ⊂ MUTATE` — replacing both the old `Gate`'s
chit-chat/act split and the old `Estimator`'s trivial/mutate split with one call and one scale.

```python
@dataclass
class ScopeEstimate:
    scope: str = "solo"  # "chat" | "solo" | "mutate"
```

| Output   | Meaning                                                              | Next stage |
|----------|-----------------------------------------------------------------------|------------|
| `CHAT`   | answerable with no codebase access: greetings, identity/capability questions, acknowledgments, general knowledge unrelated to this workspace | root solo, no tool schemas (`tools_override=frozenset()`), thinking explicitly disabled |
| `SOLO`   | a single small file, a few small edits, or a read/query; no specialist unit of work is implied | root solo, full tool set, normal effort/thinking — deliberately loose, no sequencer |
| `MUTATE` | implementation-sized: multiple files/modules, or a distinct unit of work such as a full module or test suite | Sequencer + interpreter (`_stream_graph`) |

Any unparseable model output, or a call failure, logs a warning and falls back to `SOLO` — never
silent, and never the two extremes: `CHAT` would wrongly deny codebase access, `MUTATE` would spend
a planning call the request never asked for. This mirrors the old `Gate`'s "unknown token falls to
ACT, never silence" guarantee, just landing on the new middle rung instead of a separate guard
stage.

**History context:** last 6 user/assistant turns prepended before the user message — this is why a
follow-up like `"yes"` / `"do it"` classifies correctly using prior turns instead of reading as a
context-free `CHAT`.

An explicit `/<alias>` seed (an invocable subagent's own slash command, e.g. `/refactor`, `/fix`,
`/build` — see [Subagent Routing](#subagent-routing)) **bypasses the Estimator entirely**
(`estimate_decision = "dispatch"`) — `Harness.stream()` resolves it directly against the subagent
roster and dispatches that agent on the same no-graph path `SOLO`/`CHAT` use, with warm session
context (not the cold `prior=[]` a graph-spawned subagent step gets).

---

## Sequencer + Interpreter

Plan 27 replaces `delegate`-in-root with two engineered pieces:

- **Sequencer** (`agent/pipeline/sequencer.py`) — CORE tier, one call per `MUTATE` turn, thinking
  explicitly off (a live probe measured ~90s median with CORE's own thinking-on default against
  ~13s with it off, at equal-or-better plan quality — see [LLM Integration](#llm-integration)).
  `Sequencer.sequence(user_input)` takes no seed (an explicit `/<alias>` seed never reaches the
  sequencer at all — it bypasses this stage entirely). Given the request, the model decomposes it
  into ordered steps assigned to an **auto-assignable** subagent (`code-expert`, `test-expert`,
  …) — never to root; each step carries `verify`/`repair` when the model judges the step's
  complexity warrants a preventive check (the complexity *metric* itself remains an open design
  point — today the model's own in-prompt judgment is the mechanical placeholder). The raw output
  is parsed and validated by `agent/pipeline/plan.py`'s `parse_task_graph` — schema, roster, and
  phase-eligibility checked, fail loud (`ValueError`) on any violation.
- **Interpreter** (`agent/harness/interpreter.py`, `run_task_graph`) — a **fixed**, engineered
  step-runner. Walks the validated `TaskGraph` with `execute → verify → repair → re-verify → halt`.
  It knows no agent by name or role; `dispatch(agent, instruction, mission, signal, step_scope)` is
  the only seam, supplied by the caller. An empty dispatch output, or one that starts with
  `run_subagent`'s `ERROR_PREFIX` (a crashed nested run), is always treated as a failure. A step
  failing verify once is repaired (the named `repair` agent, or a re-dispatch of `step.agent` if
  none is named) and re-verified; a second failure raises `TaskGraphHalted` — completed steps' work
  is kept, nothing rolls back, and the exception carries every `StepResult` so far.

`agent/harness/core.py`'s `Harness._stream_graph` wires the two together for production: the
`dispatch` closure runs every step — root is never a step agent, `ROOT_AGENT` fails
`parse_task_graph`'s roster check by construction — via `run_subagent`
(`agent/tools/delegate.py`), a cold, fire-and-forget nested run whose start/outcome are traced on
the same `EventBus` the no-graph path uses, so diff/log/delegation rendering is shared, not
reimplemented. If `Sequencer.sequence()` raises `ValueError` (an invalid graph), `_stream_graph`
falls back to the same no-graph solo path `SOLO` takes (`_stream_solo`, `emit_start_event=False`)
instead of failing the turn.

Once the graph finishes — or halts — **root synthesizes the turn's user-facing answer itself**, in
`Harness._respond()`: a real root LLM call (session recency + the graph's own `summary` + every
`StepResult.output`), run at the `root-dispatch` touchpoint like any other root call. This is not a
mechanical recap — root writes the actual reply the user reads, and on a halt it also narrates what
happened and why. Only if `_respond()` itself raises (empty synthesis, a model/network error) does
the harness fall back to `_recap()`, a mechanical, no-LLM summary built from `graph.summary` plus
halt info — this fail-soft path exists so a turn is never lost to its own wrap-up, but it is not the
normal completion behavior.

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
    user_invocable: bool = True    # router menu + slash-alias eligibility; False = system-managed worker
    alias: str = ""                # if set and user_invocable, the slash command typed instead of `name`
    auto_assignable: bool = False  # plan 27: phase-1 decomposition may assign it;
                                    # False = post-planning-only (verify/repair agents) — never
                                    # assigned by prompt decomposition

    def build_system_base(self) -> str: ...  # _IDENTITY_SUB + mandate + _SHARED_BODY + <directives>
                                              # <tools> appended later by the Harness
```

`description` appears in the sequencer's roster string (`name: description`) and carries a "Not for…"
boundary clause to sharpen decomposition decisions. `mandate` is the role-identity sentence fed
into the agent's own context once spawned — "who you act as right now," distinct from
`directives` ("how to do it"). `tools` is a name allowlist (mirrored against
`agent/tools/catalog.py` to guard against drift); `permissions` lets a subagent further
*restrict* — never escalate beyond — the session's grant. `auto_assignable` is orthogonal to
`user_invocable`: `code-expert`/`test-expert` are both (sequencer may assign them, their own
`/<alias>` seed can dispatch them directly, bypassing the sequencer); a verify/repair-only agent is
`user_invocable=True, auto_assignable=False` (slash-summonable, but never assigned by phase-1
decomposition). `code-fixer` is an ordinary `user_invocable=True, auto_assignable=True` specialist
(alias `/fix`). See [System Prompt Assembly](#system-prompt-assembly)
for the full funnel and [Harness — Tool Loop](#harness--tool-loop) for how the allowlist and
permissions jointly determine the *effective* tool set (and therefore the `<tools>` prompt content).

Package layout: `__init__.py` exports `Subagent`, `SUBAGENTS`, `NAMESPACES`, `validate_registry`,
and `_discover()` auto-discovery.
One file per subagent.
Namespace-level shared directives live alongside each namespace's `__init__.py` (namespace =
`"coding"`, `"testing"`, `"generic"`) — composed into `subagent.directives` at import time.
Adding a subagent = drop one file; zero other changes required.

`NAMESPACES` includes `"coding"`, `"testing"`, and `"generic"`. There is no fallback subagent to
route an unmatched request to — root itself is that fallback: general or simple requests, and
anything the sequencer decomposes into scaffolding/glue work, land on whichever auto-assignable
subagent the sequencer picks, since root is never a step agent. `code-expert` handles **decided
behavior work** — features and behavior-changing rewrites where the approach is already decided; it
owns its assigned step's implementation in full; its `description` excludes all bug fixing
regardless of how obvious the fix (that's `code-fixer`'s domain — trace root cause first, then
apply the minimal correction), and pure refactors / complexity-reduction passes with no behavior
change (those go to `code-refactorer`).

`validate_registry()` runs at startup — raises if any namespace in `NAMESPACES` has no badge
color, a subagent has an unknown namespace, or names/aliases collide.

Subagent selection is performed by the Sequencer (phase-1 decomposition, auto-assignable roster
only) — see [Sequencer + Interpreter](#sequencer--interpreter) above — or explicitly via a
subagent's own `/<alias>` seed (dispatches that agent directly, bypassing the sequencer/task graph
entirely; unknown seeds are rejected at the command layer).

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
(`estimate` → `root`/sequencer-badge names as `SubAgentStartEvent`/`DelegationStartEvent` arrive),
not a single per-turn label. An explicit `/<alias>` seed means "the whole turn runs as that
subagent's identity" — it dispatches directly to that agent, bypassing the sequencer, so a seeded
turn renders one `SubAgentStartEvent` badge for the named subagent itself, never the
`DelegationStartEvent` badges an interpreter step walk would produce.

---

## Harness — Tool Loop

`Harness` (`agent/harness/core.py`) is the single, always-on session holder users always talk to.
`Harness.stream(...)` is the entry point for every turn; its shape branches on the estimate/seed
outcome (see [Overview](#overview)):

```python
async def stream(
    self, session, user_input, permission_callback=None,
    subagent: Subagent | None = None, extra_params: dict | None = None,
    hidden_grant_callback=None, seed: str | None = None,
) -> AsyncIterator[AgentEvent | str]
```

- `seed=None`, `chat`/`solo` estimate (or no estimator wired) → no-graph single-agent path
  (`_stream_solo`, below), run as root.
- `seed=None`, `mutate` estimate → `_stream_graph`: Sequencer + interpreter
  (see [Sequencer + Interpreter](#sequencer--interpreter)); on an invalid graph, falls back to
  `_stream_solo`.
- `seed=<alias>` → the Estimator is skipped entirely; `stream()` resolves `seed` against
  `SUBAGENTS` and calls `_stream_solo` with that subagent bound in place of root — same no-graph
  path, warm session context, one `SubAgentStartEvent` for the named subagent.
- A genuine graph-spawned subagent step (`_stream_graph`'s own `dispatch`) never calls `stream()`
  again — it goes straight through `agent/tools/delegate.py::run_subagent`, cold (`prior=[]`).

`extra_params` overrides the effective `extra_params` computed for this call only — `None` (the
default) means "use the rung's own default"; the `chat` rung's own default is root's model's
explicit thinking-*disable* payload (not a bare `{}`, which some providers treat as "unspecified"
and default to reasoning on).

No `delegate` tool is ever registered for root (plan 27 decision 11 — removed outright, no
hybrid); the harness (sequencer + interpreter) owns all cross-agent control flow instead. A
subagent may register `delegate` for itself when it declares `delegates_to`, bounded to depth 1.

`stream()` delegates prompt assembly to the module-level `_build_agent(...)`, which is the
**single point** where the effective tool set — and therefore the final `<tools>` block — is
computed, for both root and subagent runs:

```
1. compute `effective` permissions = session.permissions ANDed field-wise with subagent.permissions (spawn mode only)
2. build `selected`: walk make_tools(working_dir), drop any tool not in subagent.tools (when an allowlist is set),
   then drop any tool whose required_permission isn't granted by `effective` (when there's no permission_callback to escalate),
   then apply `tools_override` (chat's frozenset(), or a step's scope-derived ceiling) when set
3. system = system_base + "\n<tools>\n" + render_tool_instruction([t.name for t in selected])
4. construct Agent(system=system, extra_params=effective_extra_params, …), register exactly the
   `selected` tools, attach the PermissionGate
```

| Mode                       | Trigger                                    | `system_base`                  | Prior context                                  |
|-----------------------------|--------------------------------------------|---------------------------------|------------------------------------------------|
| **root, no graph**          | `chat`/`solo` estimate, no seed             | `ROOT_SYSTEM_PROMPT` (+ directive pump + GEKAI.md) | `_recency_turns(session.messages, _RECENCY_N=2)` — last 2 user/assistant pairs + current input |
| **seed dispatch**           | explicit `/<alias>`                         | `subagent.build_system_base()` | same warm recency as root, not cold             |
| **graph-spawned step**      | `mutate` estimate, one `TaskGraph` step     | `subagent.build_system_base()` | cold — `prior = []` + current input only        |

The `<tools>` block therefore always reflects the *effective, post-filtering* tool set — never
the subagent's bare declared allowlist — so the activation prompt never references a tool the
agent can't actually call (e.g. a read-only subagent's prompt omits all shell/edit guidance).

`SubAgentStartEvent` carries `name="root"` / `description="thinking"` for a root run, or
`subagent.name` / `subagent.description` for a seed dispatch or graph-spawned step — every path
emits it; all three stream through the same unified event flow.

`_recency_turns(session.messages, _RECENCY_N)` extracts the last `_RECENCY_N=2` user/assistant
pairs (skipping system messages, excluding the current trailing user input); the current user
input is then appended explicitly.

When `--debug` is active, `stream()` calls
`append_debug(session, {"content": {"system": agent.system, "extra_params": effective_extra_params}})`
**after** `_build_agent` returns — `agent.system` is the true, fully-assembled prompt string
(mutable field on `llmstitch.Agent`), and lands in `.debug.jsonl` alongside the workspace
context block. `extra_params` here is the *effective* value actually passed to the model for
this turn (the `chat` rung's explicit thinking-disable payload, not the Harness's instance default).

```mermaid
sequenceDiagram
    participant TUI
    participant Harness
    participant llmstitch
    participant CoreModel

    TUI->>Harness: stream(session, user_input, seed=forced_seed)
    Note over Harness: Estimator picks chat/solo/mutate (skipped entirely when seed is set) —<br/>only mutate runs Sequencer + interpreter (_stream_graph); chat/solo/seed take the no-graph path
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
    Harness-->>TUI: DoneEvent
    Harness-->>TUI: final answer text
```

Events flow through `EventBus` → async queue → TUI stream. The final answer text is yielded
once, after `DoneEvent` — there is no live token-by-token streaming to the user for the graph path
(the `chat`/`solo`/seed no-graph path does stream text chunks live; the graph path's answer only
ever appears as the one final assembled string from `_respond`).

---

## Session Persistence

Sessions stored at `~/.gekai/workspaces/{normalized-repo-path}/{session-id}.jsonl`.
Every entry is timestamped JSON with a `kind` field:

```
{ts, kind:"turn",    role:"user|assistant|system", content}   ← LLM context only
{ts, kind:"command", content:"/clear"}                        ← slash command typed by user
{ts, kind:"event",   source:"...", content:"..."}             ← system-side, non-LLM
```

Event sources: `error` · `interrupted` · `max_iterations` · `command` (command result).
Entries without `kind` (legacy) default to `"turn"`.

**Boundary:** `session.jsonl` = everything the user saw on screen. `{session-id}.debug.jsonl` = internal plumbing (system prompts, estimate decisions, per-turn debug context) — `--debug` only.
Litmus: *did the user see it on screen?* → session; *did only the developer need it?* → debug.

**Two readers:**
- `load_session(id)` → turns only — model context for resuming the LLM conversation.
- `load_timeline(id)` → full ordered list — drives visual chat rebuild on `--resume`.

**Max-iterations:** on agent hit with no text, writes `event(source="max_iterations")` instead of an empty assistant turn. Context stays clean; rebuild shows the warning.

**`/clear` is the first entry of the new session** — `/clear` is persisted to the *new* session immediately after creation, marking it visually at the top. The old session retains the `/clear` command entry too (no result entry, since execution returns early).
