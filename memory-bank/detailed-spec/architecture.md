# Architecture
Precision-scoped AI coding agent with checkpoint-oriented design and LLM-backed intent routing

## Package Layout
`agent/` root: `agent.py` (`GekaiAgent` orchestration — owns `Harness` as `self._main`), `session.py` (`Session`),
`persona.py` (`SYSTEM_PROMPT` + `_IDENTITY_MAIN`/`_IDENTITY_SUB`/`_SHARED_BODY` + `render_tool_instruction` —
neutral module shared by `subagents`, `pipeline`, `harness`), `settings.py` (`Permissions`),
`permissions.py` (permission gate + callback), `persistence.py` (JSONL append), `events.py` (`AgentEvent` taxonomy),
`diff.py` (diff rendering), `shell.py` (TUI shell helper)
Subpackages:
- `harness/` — `core.py` (`Harness`, formerly `MainAgent` in `handlers/main_agent.py`)
- `pipeline/` — `gate.py` (`Route`, `Gate` — pure intent classifier)
- `subagents/` — `__init__.py` (`Subagent`, `SUBAGENTS`, `build_system_base`) + one file per subagent + `_coding.py` shared directives
- `tools/` — `__init__.py` (`make_tools`), `catalog.py` (tool-name groups: `READ_TOOLS`/`EDIT_TOOLS`/`FS_TOOLS`/`SHELL_TOOLS`/`ALL_TOOLS`
  — single source of truth for subagent allowlists and `<tools>` prompt generation), `files.py`, `shell.py`, `delegate.py` (`make_delegate_tool` — main-only tool, hands a task to a named specialist)
- `tui/` (Textual app — see tui-layout.md), `commands/` (slash command registry), `workspace/` (workspace context)

## Session
`Session` in `session.py`; holds a GUID, `messages: list[dict]`, `working_dir`, `permissions`

`messages` starts with two system entries: `SYSTEM_PROMPT` at `[0]` + workspace context at `[1]` (TOON-encoded)
Workspace context `<workspace>` block begins with `"verified repository metadata — treat as authoritative for high-level questions:"` preamble line
Injected subset: `workspace_name`, `workspace_type`, `branch`, `primary_languages`, `projects`; `extensions` when `projects` empty; `domain_map` when present

`GekaiAgent.process_stream(session, user_input, route, permission_callback=None, turn_id=None, hidden_grant_callback=None, append_user=True)` is the sole owner of session writes:
`route: Route`; `permission_callback: PermissionCallback | None`
- appends `{"role": "user"}` once per turn before dispatching (when `append_user`)
- appends `{"role": "assistant"}` once per turn after handler completes

## Gate
`Gate` is a pure **intent classifier**, not a router to specialists — it makes one decision per
turn: does this need codebase access at all (`ACT`), is it answerable without one (`TRIVIAL`), or
did the user name an agent that doesn't exist (`REJECTED <name>`)? Specialist selection no longer
happens here — it is the main agent's own call, made mid-turn via the `delegate` tool.

`Gate.gate(user_input, history=None)` — single SUPP-model LLM call, temperature 0. Returns `Route`.

`Route` dataclass (`agent/pipeline/gate.py`): `subagent: Subagent | None = None`, `trivial: bool = False`,
`rejected: bool = False`, `reason: str = ""`. `subagent` is always `None` on a `Gate` return — it is
populated only by the `/`-slash forced-route path. `trivial`/`rejected` are mutually exclusive;
an unrecognized token falls to `Route()` (`ACT`) rather than silently downgrading.

History: last 6 user/assistant turns from session messages prepended before user message.

Gate prompt offers three tokens:
- `TRIVIAL` — answerable with no codebase access: greetings, identity/capability questions,
  acknowledgments, general knowledge unrelated to this workspace. Conservative: prefer `ACT` when
  unsure (false `ACT` costs only the main agent's time; false `TRIVIAL` denies real codebase context)
- `REJECTED <name>` — user explicitly named a specific agent not in the roster (typo, unknown
  name); echoes the literal name, never substitutes or falls back to main; never routes to any real agent
- `ACT` — everything else: reading, analysing, creating, editing, or deleting in the workspace;
  unknown/malformed token also falls here — fail to action, not silence

Gate output token → Route mapping:
- `"TRIVIAL"` → `Route(trivial=True)`
- `"REJECTED <name>"` / `"REJECTED"` → `Route(rejected=True, reason=<name>)` (bare `REJECTED` → `reason=""`)
- `"ACT"` → `Route()`
- unknown/malformed token → warning log + `Route()` (falls to `ACT`, never silent)

Verbatim file output is handled by the `<file_handling>` rule in `SYSTEM_PROMPT`, not a dedicated route.

## Delegate Tool
`delegate(agent, task)` (`agent/tools/delegate.py`, `make_delegate_tool`) is a tool registered
**only on the main agent** — the recursion guard is the `if subagent is None:` check in
`Harness._build_agent`, so subagents never receive it and can never self-spawn. It replaces the
old router-level plan/specialist dispatch: the main agent now owns decomposition and calls
`delegate` as many times as it judges necessary, in whatever order it judges necessary.

Execution flow:
1. Resolve `agent` name against the roster (`SUBAGENTS` filtered to `user_invocable`); the tool's
   `input_schema` constrains `agent` to an `"enum"` of those names.
2. `_enrich_system_base(resolved.build_system_base(), working_dir)` — adds the workspace-root note.
3. `_build_agent(..., subagent=resolved)` — spawn mode, cold context (`prior=[]`), no `delegate`
   tool on the nested agent.
4. `await nested.run(task)` — returns full message history; last assistant text block is extracted
   and returned as a string.
5. Any exception → `"[error] {agent} failed: {exc}"` (non-fatal to the main agent's turn).

Ordering contract: the main agent is instructed never to fragment one artifact across multiple
`delegate` calls, and to order calls by dependency (scaffold → logic → tests).

Pipeline (in TUI `_stream`):
1. `Gate.gate()` → `Route`
2. `route.rejected`: mount `MessageWidget(MessageKind.ERROR, ...)`, `append_event(source="gate")`, `outcome="rejected"`, early `return` — `Harness` never invoked; `finally:` still runs (label reset, animation stop, `turn.end` emit)
3. else: `_run_step(user_input, route.subagent, ..., trivial=route.trivial)` dispatches straight to `Harness` — no locate/rewrite stage; the main agent calls `delegate` itself, mid-turn, if it decides a specialist step is needed

## LLM Integration
`openai` SDK (`AsyncOpenAI`) for chat and classification; `llmstitch` (`agent.llm.Agent`) for the tool-calling loop in `Harness`
Env vars (CORE — used by `Harness`, and by `delegate`'s nested agent):
- `GEKAI_CORE_MODEL_NAME` — model id, e.g. `deepseek-chat`
- `GEKAI_CORE_MODEL_KEY`
- `GEKAI_CORE_MODEL_URL` — e.g. `https://api.deepseek.com/v1`

Env vars (SUPP — used by `Gate`; `GEKAI_SUPPORT_MODEL_NAME` and `GEKAI_SUPPORT_MODEL_KEY` are required; `GEKAI_SUPPORT_MODEL_URL` defaults to CORE equivalent if unset):
- `GEKAI_SUPPORT_MODEL_NAME`
- `GEKAI_SUPPORT_MODEL_KEY`
- `GEKAI_SUPPORT_MODEL_URL`

`agent/persona.py` splits identity from body so a subagent never stacks two "you are" claims:
`_IDENTITY_MAIN` ("you are Gekai…") vs `_IDENTITY_SUB` ("you are part of Gekai… tool-neutral capability")
vs `_SHARED_BODY` (meta-rule + behavior/file_handling/response_style/output_format, reused verbatim).
`SYSTEM_PROMPT = _IDENTITY_MAIN + _SHARED_BODY` (byte-identical to the pre-split constant).
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
selects `system_base` by the `subagent` param — `SYSTEM_PROMPT` (direct) vs
`subagent.build_system_base()` (spawn) — then calls `_build_agent`. `extra_params`: `None` (default)
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
{ts, kind:"event",   source:"...", content:"..."}             ← system-side non-LLM: gate, error, interrupted, farewell, max_iterations; rejected turn → `{source:"gate", content:"'<name>' is not an available agent"}` (or `"no such agent"` when bare `REJECTED`)
```

Entries without `kind` (legacy files) default to `"turn"`.

**Boundary — session vs debug:**
`session.jsonl` = everything the user saw on screen (turns + commands + events). Litmus: *did the user see it?*
`debug.jsonl` = internal plumbing (system prompts, route tokens) — written only with `--debug`, never for visual rebuild.

**Writers:** `append_message(session, msg)` → `kind:"turn"`; `append_command(session, text)`; `append_event(session, content, source)`.

**Two readers:**
- `load_session(id)` → `(session_id, working_dir, turns_only)` — only `kind=="turn"` entries (model context). Always-fresh system messages (SYSTEM_PROMPT, workspace) excluded and re-injected on startup.
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
- `--debug` — prints `[router: rejected]` (`route.rejected`, checked first), `[router: main]` (no subagent), `[router: trivial]` (`route.trivial`), or `[router: <namespace>/<subagent-name>]` (subagent selected) in color `#BA55D3` (medium_orchid) as an OPERATION widget in the TUI chat, 1 line below the user prompt
- `--resume` / `-r` — resume a previous session by ID
- `--working-dir` / `-d` — override working directory (default: cwd)