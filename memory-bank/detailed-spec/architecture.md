# Architecture
Precision-scoped AI coding agent with checkpoint-oriented design and LLM-backed intent routing

## Package Layout
`agent/` root: `agent.py` (`GekaiAgent` orchestration — owns `Harness` as `self._main`), `session.py` (`Session`),
`persona.py` (`ROOT_SYSTEM_PROMPT` + `_IDENTITY_ROOT`/`_IDENTITY_SUB`/`_SHARED_BODY` + `render_tool_instruction` —
neutral module shared by `subagents`, `pipeline`, `harness`), `settings.py` (`Permissions`),
`permissions.py` (permission gate + callback), `persistence.py` (JSONL append), `events.py` (`AgentEvent` taxonomy),
`diff.py` (diff rendering), `shell.py` (TUI shell helper)
Subpackages:
- `harness/` — `core.py` (`Harness`), `interpreter.py` (`run_task_graph` — the fixed execute→verify→repair→halt walker), `touchpoints.py` (`Touchpoint` registry — every place the harness invokes a model)
- `pipeline/` — `gate.py` (`Route`, `Gate` — pure intent classifier), `plan.py` (`Task`, `TaskGraph`, `parse_task_graph`, `ROOT_AGENT`), `sequencer.py` (`Sequencer` — builds the `TaskGraph`)
- `subagents/` — `__init__.py` (`Subagent`, `SUBAGENTS`, `build_system_base`, `NAMESPACE_COLORS`) + one namespace package per action domain (`coding/`, `testing/`, `generic/`), each with its own `__init__.py` declaring `namespace`/`namespace_directives` and one file per member subagent
- `tools/` — `__init__.py` (`make_tools`), `catalog.py` (tool-name groups: `READ_TOOLS`/`EDIT_TOOLS`/`FS_TOOLS`/`SHELL_TOOLS`/`ALL_TOOLS`
  — single source of truth for subagent allowlists and `<tools>` prompt generation), `files.py`, `shell.py`, `delegate.py` (`run_subagent` — cold nested-run dispatcher used by the interpreter, see `## Cross-Agent Dispatch`)
- `tui/` (Textual app — see tui-layout.md), `commands/` (slash command registry), `workspace/` (workspace context)

## Session
`Session` in `session.py`; holds a GUID, `messages: list[dict]`, `working_dir`, `permissions`

`messages` starts with two system entries: `ROOT_SYSTEM_PROMPT` at `[0]` + workspace context at `[1]` (TOON-encoded)
Workspace context `<workspace>` block begins with `"verified repository metadata — treat as authoritative for high-level questions:"` preamble line
Injected subset: `workspace_name`, `workspace_type`, `branch`, `primary_languages`, `projects`; `extensions` when `projects` empty; `domain_map` when present

`GekaiAgent.process_stream(session, user_input, route, permission_callback=None, turn_id=None, hidden_grant_callback=None, append_user=True)` is the sole owner of session writes:
`route: Route`; `permission_callback: PermissionCallback | None`
- appends `{"role": "user"}` once per turn before dispatching (when `append_user`)
- appends `{"role": "assistant"}` once per turn after handler completes

## Gate
`Gate` is a pure **intent classifier** — it makes one decision per turn: does this need codebase
access at all (`ACT`), or is it answerable without one (`TRIVIAL`)? It does not select a specialist
and does not reject unknown agent names; specialist selection is the sequencer's job, downstream
(see `## Cross-Agent Dispatch`).

`Gate.gate(user_input, history=None)` — single SUPP-model LLM call, temperature 0. Returns `Route`.

`Route` dataclass (`agent/pipeline/gate.py`): `trivial: bool = False`. That is its only field —
`ACT` is simply `Route()`.

History: last 6 user/assistant turns from session messages prepended before user message.

Gate prompt offers two tokens:
- `TRIVIAL` — answerable with no codebase access: greetings, identity/capability questions,
  acknowledgments, general knowledge unrelated to this workspace. Conservative: prefer `ACT` when
  unsure (false `ACT` costs only extra harness time; false `TRIVIAL` denies real codebase context)
- `ACT` — everything else: reading, analysing, creating, editing, or deleting in the workspace;
  unknown/malformed token also falls here — fail to action, not silence

Gate output token → Route mapping:
- `"TRIVIAL"` → `Route(trivial=True)`
- `"ACT"` / anything else → `Route()` — an unrecognized token logs a warning and falls to `Route()`,
  never silent

Verbatim file output is handled by the `<file_handling>` rule in `ROOT_SYSTEM_PROMPT`, not a dedicated route.

## Cross-Agent Dispatch
There is no `delegate` tool and no `Harness`-registered tool for calling another agent — root
cannot self-spawn, and no subagent can spawn anything (plan 27 decision 11, superseding plan 25's
agents-as-tools). All cross-agent dispatch is owned by the sequencer + the fixed interpreter, not
by an LLM deciding mid-turn to call a tool:

1. `Harness.stream()` estimates the turn (`Estimator`, or a `seed` from a forced `/`-slash route);
   a `mutate`/`seeded` estimate routes into `Harness._stream_graph()`, everything else runs root
   directly, cold-free, at the `root-dispatch` touchpoint (no graph).
2. `Sequencer.sequence()` (`agent/pipeline/sequencer.py`) makes one CORE-tier LLM call and returns
   a validated `TaskGraph` (`agent/pipeline/plan.py`, `parse_task_graph`) — every step's `agent` is
   an `auto_assignable` roster `Subagent` name; `root` (`ROOT_AGENT = "root"`) is rejected if the
   model names it as a step agent.
3. `agent/harness/interpreter.py::run_task_graph()` walks the graph: `execute → verify → repair →
   re-verify → halt`, no knowledge of any agent by name or role. Each step's execution is a call to
   the `dispatch` closure `Harness._stream_graph()` builds, which calls
   `agent/tools/delegate.py::run_subagent(agent, task, ...)`.
4. `run_subagent()` resolves `agent` against `SUBAGENTS` (any roster name, invocable or
   post-planning-only), builds a cold nested `Agent` via `Harness._build_agent(..., subagent=resolved)`
   (`prior=[]`, no cross-agent tool of any kind registered), runs it, and returns the last
   assistant text block as a string (`"[error] {agent} failed: {exc}"` on any exception — non-fatal
   to the graph, handled by the interpreter's verify/repair/halt policy instead).

Ordering, tool-breadth narrowing (`Task.scope` → `harness/tool_scope.py`), and per-node tier
scaling (`scale()`, `node_signal()`) are all sequencer/interpreter concerns — never something an
agent decides mid-turn. See `subagents.md → Harness` for the full sequencer→interpreter walkthrough
and `## Root` below for how root owns the turn around this execution.

Pipeline (in TUI `_stream`):
1. `Gate.gate()` → `Route` (`trivial` or not — see `## Gate`)
2. `_run_step(user_input, seed, ..., trivial=route.trivial)` dispatches straight to `Harness.stream()`
   — no locate/rewrite stage, no rejection path; an unrecognized `/`-slash agent name is a CLI/palette
   concern, not something `Gate` or `Route` handles

## Root
Root is the session-owning unit — deployed first by the harness, present for the whole turn, never
a task-graph step (`ROOT_AGENT` fails `parse_task_graph`'s roster check by construction: it is not
in `SUBAGENTS`, so no `agent` value equal to `"root"` can ever pass `auto_assignable` validation).
It is not a registry `Subagent` and is not in the Gate menu, sequencer roster, or palette — it is
special-cased in `Harness`, not filtered out of three separate lists.

Root owns two things:
- **Warm context.** `_recency_turns(session.messages, _RECENCY_N=2)` (last 2 user/assistant pairs)
  is passed on every root-dispatch call — the no-graph path in `Harness.stream()` and the
  synthesis call below. Subagent (spawn-mode) runs never get this; they run cold (`prior=[]`).
- **The turn's final answer.** `Harness._respond()` (`agent/harness/core.py`) is a real root call
  at the `root-dispatch` touchpoint — session recency + the graph's `summary` + every
  `StepResult.output`, prompted to write the reply the user sees. This runs after
  `run_task_graph()` finishes (success) or raises `TaskGraphHalted` (root also narrates the halt,
  naming the step and reason). There is no separate `Responder` unit or touchpoint — root absorbed
  it. Fail-soft: any exception during synthesis (empty text, model/network error) falls back to
  `_recap()`, a mechanical (no LLM call) summary built from `graph.summary` and the halt info alone,
  so a turn is never lost to its own wrap-up.

`_ROOT_DIRECTIVES` ("ask before acting on an ambiguous request") is safe specifically because root
is never dispatched cold inside a graph — it is the only unit ever facing a human, so "ask" is
always answerable.

## LLM Integration
`openai` SDK (`AsyncOpenAI`) for chat and classification; `llmstitch` (`agent.llm.Agent`) for the tool-calling loop in `Harness`
Env vars (CORE — used by `Harness`'s root-dispatch/subagent-dispatch calls, and the `Sequencer`):
- `GEKAI_CORE_MODEL_NAME` — model id, e.g. `deepseek-chat`
- `GEKAI_CORE_MODEL_KEY`
- `GEKAI_CORE_MODEL_URL` — e.g. `https://api.deepseek.com/v1`

Env vars (SUPP — used by `Gate`; `GEKAI_SUPPORT_MODEL_NAME` and `GEKAI_SUPPORT_MODEL_KEY` are required; `GEKAI_SUPPORT_MODEL_URL` defaults to CORE equivalent if unset):
- `GEKAI_SUPPORT_MODEL_NAME`
- `GEKAI_SUPPORT_MODEL_KEY`
- `GEKAI_SUPPORT_MODEL_URL`

`agent/persona.py` splits identity from body so a subagent never stacks two "you are" claims:
`_IDENTITY_ROOT` ("you are Gekai…") vs `_IDENTITY_SUB` ("you are part of Gekai… tool-neutral capability")
vs `_SHARED_BODY` (meta-rule + behavior/file_handling/response_style/output_format, reused verbatim).
`ROOT_SYSTEM_PROMPT = _IDENTITY_ROOT + _SHARED_BODY + "\n<directives>\n" + _ROOT_DIRECTIVES`
(root-only directives, mirroring the `<directives>` block a `Subagent` gets from
`build_system_base()` — never reaches subagents, which assemble their own system prompt independently).
The `<tools>` block is generated — never static — by `render_tool_instruction(assigned)`: a
deterministic, non-LLM fragment table (`_TOOL_GUIDANCE`) keyed on tool-name groups from
`agent/tools/catalog.py` (`READ_TOOLS`/`SHELL_TOOLS`/etc — `ALL_TOOLS` mirrors `make_tools()` output,
guarded by a drift test). `render_tool_instruction(ALL_TOOLS)` reproduces the legacy static
`TOOL_INSTRUCTION` string verbatim. Each fragment fires on "any" (intersection) or "all" (superset)
of its trigger group, so the prompt only ever names tools the agent actually has.

`Subagent.build_system_base()` assembles the spawn-mode prompt *base*: `_IDENTITY_SUB` + optional
plain-prose role line (`subagent.mandate`, e.g. "you act as a code-change specialist…" — no
`<core_mandate>` wrapper) + `_SHARED_BODY` + optional `<directives>` block (`subagent.directives`);
tags are non-closing. The `<tools>` block is appended afterward by `Harness._build_agent`, not by
the subagent — see below.

`Harness._build_agent(model, …, system_base, subagent=None)` is the **single point** that computes
the effective tool set and assembles the final system string, for both modes:
1. `effective` permissions = `session.permissions` ANDed field-wise with `subagent.permissions` (spawn mode, when set)
2. `selected` = `make_tools(working_dir)` filtered by `subagent.tools` allowlist (when set), then by whether
   `effective` grants each tool's `required_permission` (when there's no `permission_callback` to escalate)
3. `system = system_base + "\n<tools>\n" + render_tool_instruction([t.name for t in selected])`
4. construct `Agent(system=system, …)`, register exactly `selected`, attach `PermissionGate(effective, …)`

This guarantees the `<tools>` prompt always reflects the *effective, post-filter* set — not the
subagent's bare declared allowlist — e.g. a read-only subagent's prompt omits all shell/edit guidance.

`Harness.stream(session, user_input, permission_callback=None, subagent=None, extra_params=None)`
selects `system_base` by the `subagent` param — `ROOT_SYSTEM_PROMPT` (root, no-graph path) vs
`subagent.build_system_base()` (spawn, cold nested run) — then calls `_build_agent`. `extra_params`: `None` (default)
→ use `self._extra_params` (set at construction from `resolve_thinking_params`); explicit `{}` →
no thinking params for this turn (the `TRIVIAL`-route case, set in `GekaiAgent.process_stream` via
`extra_params={} if route.trivial else None`). Direct mode passes
`_recency_turns(session.messages, _RECENCY_N=2)` (last 2 user/assistant pairs, system messages
skipped, trailing user input excluded) + current input; spawn mode runs cold — `prior = []` +
current input only, no recency context, no async/resume.
`--debug` active: `stream()` calls
`append_debug(session, {"content": {"system": agent.system, "extra_params": effective_extra_params}})`
**after** `_build_agent` returns (`agent.system` is the true assembled prompt, a mutable field on
`llmstitch.Agent`; `effective_extra_params` is the value actually used this turn) — written to `.debug.jsonl`.

## Session Persistence
Sessions stored as JSONL at `~/.gekai/workspaces/{normalized-repo-path}/{session-id}.jsonl`; each line is a timestamped entry with a `kind` field:

```
{ts, kind:"turn",    role:"user|assistant|system", content}   ← LLM context; only these fed to model / /compact
{ts, kind:"command", content:"/clear"}                        ← slash command typed by user
{ts, kind:"event",   source:"...", content:"..."}             ← system-side non-LLM: gate, error, interrupted, farewell, max_iterations
```

Entries without `kind` (legacy files) default to `"turn"`.

**Boundary — session vs debug:**
`session.jsonl` = everything the user saw on screen (turns + commands + events). Litmus: *did the user see it?*
`debug.jsonl` = internal plumbing (system prompts, route tokens) — written only with `--debug`, never for visual rebuild.

**Writers:** `append_message(session, msg)` → `kind:"turn"`; `append_command(session, text)`; `append_event(session, content, source)`.

**Two readers:**
- `load_session(id)` → `(session_id, working_dir, turns_only)` — only `kind=="turn"` entries (model context). Always-fresh system messages (ROOT_SYSTEM_PROMPT, workspace) excluded and re-injected on startup.
- `load_timeline(id)` → `(working_dir, all_entries)` — full ordered list for visual rebuild; non-persistent system turns excluded.

**Max-iterations:** when handler hits limit with no text produced, `process_stream` writes `append_event(source="max_iterations")` instead of an empty assistant turn — context stays clean, rebuild shows the warning.

**`/clear` is the first entry of the new session:** command text is persisted to the *new* session (not the old one) immediately after it is created, making it the marker at the top of that session's timeline.

On resume (`--resume <session-id>`): `load_session` restores model context; `load_timeline` drives visual rebuild; fresh system messages re-injected; scroll to bottom.

## Streaming UX
Textual exclusive worker per turn; see `tui-layout.md → Streaming Worker`.
- spinner `· • ● •` + random operative verb + elapsed time in `#status-line` (accent color via `styles.color`)
- ESC cancels worker; `Ctrl+C` quits app
- on completion: ASSISTANT widget (Markdown) + OPERATION widget (`* {PastVerb} for {duration}`)

## Commands
Slash-prefixed input intercepted by `CommandPalette` then dispatched via `CommandRegistry`.
- `/exit` — exit to terminal (with farewell message + delay)
- `/clear` — clears chat and starts a new session (resets session ID)

## CLI Flags
- `--debug` — writes the assembled system prompt, `extra_params`, and every tool call/result to
  `.debug.jsonl` (`append_debug`, see `## Session Persistence`); the gate's `trivial`/`act` decision
  is separately emitted as a `"route"` telemetry event (`agent/tui/app.py`), not rendered in the TUI
- `--resume` / `-r` — resume a previous session by ID
- `--working-dir` / `-d` — override working directory (default: cwd)