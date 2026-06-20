# Architecture
Precision-scoped AI coding agent with checkpoint-oriented design and LLM-backed intent routing

## Package Layout
`agent/` root: `agent.py` (`GekaiAgent` orchestration — owns `Harness` as `self._main`), `session.py` (`Session`),
`persona.py` (`SYSTEM_PROMPT` + `_IDENTITY_MAIN`/`_IDENTITY_SUB`/`_SHARED_BODY` + `render_tool_instruction` —
neutral module shared by `subagents`, `pipeline.router`, `harness`), `settings.py` (`Permissions`),
`permissions.py` (permission gate + callback), `persistence.py` (JSONL append), `events.py` (`AgentEvent` taxonomy),
`diff.py` (diff rendering), `shell.py` (TUI shell helper)
Subpackages:
- `harness/` — `core.py` (`Harness`, formerly `MainAgent` in `handlers/main_agent.py`), `file_locator.py` (`FileLocator`)
- `pipeline/` — `router.py` (`Route`, guard router), `blast_radius.py` (gate only), `rewriter.py` (prompt rewriter)
- `subagents/` — `__init__.py` (`Subagent`, `SUBAGENTS`, `build_system_base`) + one file per subagent + `_coding.py` shared directives
- `tools/` — `__init__.py` (`make_tools`), `catalog.py` (tool-name groups: `READ_TOOLS`/`EDIT_TOOLS`/`FS_TOOLS`/`SHELL_TOOLS`/`ALL_TOOLS`
  — single source of truth for subagent allowlists and `<tools>` prompt generation), `files.py`, `shell.py`
- `tui/` (Textual app — see tui-layout.md), `commands/` (slash command registry), `workspace/` (workspace context)

## Session
`Session` in `router.py`; holds a GUID, `messages: list[dict]`, `working_dir`, `permissions`, `scope_gate: bool = True`, `blast_radius_limit: int = 5`

`messages` starts with two system entries: `SYSTEM_PROMPT` at `[0]` + workspace context at `[1]` (TOON-encoded)
Workspace context `<workspace>` block begins with `"verified repository metadata — treat as authoritative for high-level questions:"` preamble line
Injected subset: `workspace_name`, `workspace_type`, `branch`, `primary_languages`, `projects`; `extensions` when `projects` empty; `domain_map` when present

`GekaiAgent.process_stream(session, user_input, route, entries=None, original_input=None, permission_callback=None)` is the sole owner of session writes:
`route: Route`; `entries: list[tuple[str, list[str]]] | None` — pre-computed locate results (None when no subagent selected); `original_input: str | None` — user's verbatim text when `user_input` has been rewritten (see Prompt Rewriter); `permission_callback: PermissionCallback | None`
- appends `{"role": "user"}` once per turn before dispatching — persists `original_input` when set, else `user_input`
- appends `{"role": "assistant"}` once per turn after handler completes

## Router
`Router` is a pure **guard**, not an intent classifier — it makes one decision: does this turn
stay with `Harness` directly, or does it match a specialist subagent (or get rejected)?

`Router.route(user_input, history=None)` — single SUPP-model LLM call, temperature 0. Returns `Route`.

`Route` dataclass: `subagent: Subagent | None = None`, `rejected: bool = False`, `trivial: bool = False`.
No `namespace` property — callers read `route.subagent.namespace` directly when `route.subagent is not None`.
`trivial` and `subagent` are mutually exclusive — `TRIVIAL` token is mapped before the subagent menu check.

History: last 6 user/assistant turns from session messages prepended before user message.

Router prompt offers exactly four kinds of output token:
- `main` — default; chat, inspection, workspace questions, general code changes, light edits — anything `Harness` handles directly. Bias: prefer `main` unless a specialist clearly fits
- `TRIVIAL` — answerable with no codebase access: greetings, identity/capability questions, acknowledgments, general knowledge unrelated to this workspace. Conservative: prefer `main` when unsure (false `main` costs one extra near-empty `FileLocator` call; false `TRIVIAL` denies real codebase context)
- `REJECTED` — non-English input
- `<subagent-name>` — one of the subagents in the menu (built from `SUBAGENTS` as `name — description`); only when the request clearly and specifically matches that subagent's specialty

Router output token → Route mapping:
- `"main"` → `Route()`
- `"trivial"` → `Route(trivial=True)`
- `"rejected"` → `Route(rejected=True)`
- `<subagent-name>` (matched) → `Route(subagent=p)`
- unknown token → warning log + host-retained `Route()` (same as `main`)

Verbatim file output is handled by the `<file_handling>` rule in `SYSTEM_PROMPT`, not a dedicated route.

## File Location & Blast-Radius Gate
Locate runs on every **non-`TRIVIAL`** route (`main` and `<subagent>` alike). Gate applies only
when a subagent is selected (`route.subagent is not None`); `main` and `TRIVIAL` bypass the gate.

Pipeline (in TUI `_stream`):
1. `Router.route()` → `Route`
2. `route.trivial`: `entries = []`, skip locate + rewrite entirely
3. else: `FileLocator.locate(working_dir, user_input)` → `entries: list[tuple[path, keywords]]`
4. `route.subagent is not None`: `evaluate_blast_radius_gate(entries, session.blast_radius_limit)` → `(rejected, reason)`; if rejected and `session.scope_gate`: display rejection, return
5. `entries` non-empty: `PromptRewriter.rewrite(user_input, entries)` → `(processed_input, ui_label)`; `original_input = user_input` (see Prompt Rewriter)

`FileLocator` (`agent/harness/file_locator.py`) — agentic SUPP-model call (up to 5 iterations, read-only tools). Roams freely — reads any file type. Any exception propagates (fail-hard).

Area metric — **ancestor-collapsed directory count, code files only:**
1. Filter `entries` to `_CODE_EXTENSIONS` paths only
2. Collect parent dir of each surviving file
3. Drop any dir that has an ancestor also in the set
4. Count survivors

`_CODE_EXTENSIONS`: all popular languages — `.py .pyi .ipynb` · `.js .jsx .mjs .cjs` · `.ts .tsx` · `.vue .svelte` · `.go` · `.java` · `.cs` · `.kt .kts` · `.swift` · `.rs` · `.c .h .cpp .cc .cxx .hpp` · `.rb` · `.php` · `.scala` · `.dart` · `.ex .exs` · `.lua` · `.hs` · `.r`
Broader than `_EXT_TO_LANG` (AST support) — gate coverage ≠ symbol-parse coverage.
Manifests/configs/docs (`pyproject.toml`, `package.json`, `.yaml`, `.md`, etc.) are inspected but never counted.
All-config change → 0 areas → always passes.

`blast_radius_limit` loaded via `load_blast_radius_limit(working_dir)`: project `.gekai/settings.local.json` overrides user `~/.gekai/settings.json`; absent → `5`. Manual JSON edit only — no slash command. Gate on/off reuses `/config:gate on|off`.

## Prompt Rewriter
`PromptRewriter` (`rewriter.py`) runs whenever `FileLocator` returned **non-empty `entries`** —
for `main` and `<subagent>` routes alike, *after* the gate passes (subagent only). `TRIVIAL`
routes, and any route with empty `entries`, skip it.

`PromptRewriter.rewrite(request, entries) -> tuple[str, str]` — returns `(rewritten_request, ui_label)`. Single **CORE-model** call, temperature 0, **no thinking params** (constructed with no `extra_params`, so non-thinking even on a reasoning-capable core model). Not agentic, no tools — the locator already discovered/verified files, so this stage only *attributes* them.

Behavior: weave each located path inline where it maps to a phrase in the request; leftover located files go in a trailing `<reference_files>` block; paths quoted verbatim from the list. `rewritten_request` becomes the live user turn.

Fail-hard (`rewritten_request` only): any exception, including empty output (`ValueError`), propagates and blocks the turn — same contract as `FileLocator`. `ui_label` is fail-soft — missing/malformed degrades to `""`, cosmetic only.

Original vs processed input: rewritten string → `process_stream`'s `user_input` (what `Harness` sees); user's verbatim text → `original_input` (what is persisted + displayed). Recency windows on later turns show the original phrasing. `--debug`: rewritten text written to `.debug.jsonl` as `{"content": {"rewritten": ...}}`.

## LLM Integration
`openai` SDK (`AsyncOpenAI`) for chat and classification; `llmstitch` (`agent.llm.Agent`) for the tool-calling loop in `Harness`
Env vars (CORE — used by `Harness`, `PromptRewriter`):
- `GEKAI_CORE_MODEL_NAME` — model id, e.g. `deepseek-chat`
- `GEKAI_CORE_MODEL_KEY`
- `GEKAI_CORE_MODEL_URL` — e.g. `https://api.deepseek.com/v1`

Env vars (SUPP — used by `Router`, `FileLocator`; `GEKAI_SUPPORT_MODEL_NAME` and `GEKAI_SUPPORT_MODEL_KEY` are required; `GEKAI_SUPPORT_MODEL_URL` defaults to CORE equivalent if unset):
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
{ts, kind:"command", content:"/config:gate off"}              ← slash command typed by user
{ts, kind:"event",   source:"...", content:"..."}             ← system-side non-LLM: gate, router, error, interrupted, farewell, max_iterations
```

Entries without `kind` (legacy files) default to `"turn"`.

**Boundary — session vs debug:**
`session.jsonl` = everything the user saw on screen (turns + commands + events). Litmus: *did the user see it?*
`debug.jsonl` = internal plumbing (system prompts, route tokens, locate list, rewritten text) — written only with `--debug`, never for visual rebuild.

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
- `/config:gate on|off` — enable or disable the scope gate for the current project; persists to `.gekai/settings.local.json`; updates `session.scope_gate` immediately

## CLI Flags
- `--debug` — prints `[router: main]` (no subagent), `[router: trivial]` (`route.trivial`), or `[router: <namespace>/<subagent-name>]` (subagent selected) in color `#BA55D3` (medium_orchid) as an OPERATION widget in the TUI chat, 1 line below the user prompt; trivial turns also get a `{"route": "trivial", "skipped": ["locate", "rewrite"]}` debug entry
- `--resume` / `-r` — resume a previous session by ID
- `--working-dir` / `-d` — override working directory (default: cwd)