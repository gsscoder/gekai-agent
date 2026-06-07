# Architecture
Precision-scoped AI coding agent with checkpoint-oriented design and LLM-backed intent routing

## Package Layout
`agent/` root: `agent.py` (orchestration), `router.py` (`Route`, `Session`, guard router), `tools.py` (read/search/grep),
`blast_radius.py` (locator + gate), `rewriter.py` (prompt rewriter), `persona.py` (`SYSTEM_PROMPT`, `TOOL_INSTRUCTION` — neutral module shared by `subagents`, `router`, `handlers`),
`settings.py` (permissions), `permissions.py` (permission gate + callback), `persistence.py` (JSONL append), `normalizer.py` + `subagent.py` (`AgentEvent` taxonomy)
Subpackages: `handlers/` (`main_agent.py` — `MainAgent`), `subagents/` (`__init__.py` + one file per subagent + `_coding.py` shared directives),
`tui/` (Textual app — see tui-layout.md), `commands/` (slash command registry)

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
stay with `MainAgent` directly, or does it match a specialist subagent (or get rejected)?

`Router.route(user_input, history=None)` — single SUPP-model LLM call, temperature 0. Returns `Route`.

`Route` dataclass: `subagent: Subagent | None = None`, `rejected: bool = False`. No `namespace`
property — callers read `route.subagent.namespace` directly when `route.subagent is not None`.

History: last 6 user/assistant turns from session messages prepended before user message.

Router prompt offers exactly three kinds of output token:
- `main` — default; chat, inspection, workspace questions, general code changes, light edits — anything `MainAgent` handles directly. Bias: prefer `main` unless a specialist clearly fits
- `REJECTED` — non-English input
- `<subagent-name>` — one of the subagents in the menu (built from `SUBAGENTS` as `name — description`); only when the request clearly and specifically matches that subagent's specialty

Router output token → Route mapping:
- `"main"` → `Route()`
- `"rejected"` → `Route(rejected=True)`
- `<subagent-name>` (matched) → `Route(subagent=p)`
- unknown token → warning log + fallback `Route()` (same as `main`)

Verbatim file output is handled by the `<file_handling>` rule in `SYSTEM_PROMPT`, not a dedicated route.

## Blast-Radius Gate
Applies only when a subagent is selected (`route.subagent is not None`). `main` routes bypass the gate.

Pipeline (in TUI `_stream`):
1. `Router.route()` → `Route`
2. If `route.subagent is not None`: `BlastRadiusLocator.locate(working_dir, user_input)` → `entries: list[tuple[path, keywords]]`
3. `evaluate_blast_radius_gate(entries, session.blast_radius_limit)` → `(rejected, reason)`
4. If rejected and `session.scope_gate`: display rejection, return
5. If `entries` non-empty: `PromptRewriter.rewrite(user_input, entries)` → rewritten text becomes `processed_input`; `original_input = user_input` (see Prompt Rewriter)

`BlastRadiusLocator` — agentic SUPP-model call (up to 5 iterations, read-only tools). Roams freely — reads any file type. Any exception propagates (fail-hard).

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
`PromptRewriter` (`rewriter.py`) runs **only on routes with a selected subagent whose locator returned entries**, *after* the gate passes. `main` routes, and subagent routes with empty `entries`, skip it.

`PromptRewriter.rewrite(request, entries) -> str` — single **CORE-model** call, temperature 0, **no thinking params** (constructed with no `extra_params`, so non-thinking even on a reasoning-capable core model). Not agentic, no tools — the locator already discovered/verified files, so this stage only *attributes* them.

Behavior: weave each located path inline where it maps to a phrase in the request; leftover located files go in a trailing `<reference_files>` block; paths quoted verbatim from the list. Output (rewritten request + optional block) becomes the live user turn.

Fail-hard: any exception, including empty output (`ValueError`), propagates and blocks the turn — same contract as `BlastRadiusLocator`.

Original vs processed input: rewritten string → `process_stream`'s `user_input` (what `MainAgent` sees); user's verbatim text → `original_input` (what is persisted + displayed). Recency windows on later turns show the original phrasing. `--debug`: rewritten text written to `.debug.jsonl` as `{"content": {"rewritten": ...}}`.

## LLM Integration
`openai` SDK (`AsyncOpenAI`) for chat and classification; `llmstitch` (`agent.llm.Agent`) for the tool-calling loop in `MainAgent`
Env vars (CORE — used by `MainAgent`, `PromptRewriter`):
- `GEKAI_CORE_MODEL_NAME` — model id, e.g. `deepseek-chat`
- `GEKAI_CORE_MODEL_KEY`
- `GEKAI_CORE_MODEL_URL` — e.g. `https://api.deepseek.com/v1`

Env vars (SUPP — used by `Router`, `BlastRadiusLocator`; `GEKAI_SUPPORT_MODEL_NAME` and `GEKAI_SUPPORT_MODEL_KEY` are required; `GEKAI_SUPPORT_MODEL_URL` defaults to CORE equivalent if unset):
- `GEKAI_SUPPORT_MODEL_NAME`
- `GEKAI_SUPPORT_MODEL_KEY`
- `GEKAI_SUPPORT_MODEL_URL`

`Subagent.build_system()` assembles the spawn-mode system prompt: `SYSTEM_PROMPT` + optional `<core_mandate>` block (`subagent.mandate` when non-empty) + optional `<directives>` block (`subagent.directives` when non-empty) + `<tools>` block (`TOOL_INSTRUCTION`); tags are non-closing. In direct mode `MainAgent` composes `SYSTEM_PROMPT + "\n<tools>\n" + TOOL_INSTRUCTION` itself — no mandate/directives. `SYSTEM_PROMPT` and `TOOL_INSTRUCTION` live in `agent/persona.py`.
`TOOL_INSTRUCTION` — if the question requires file contents, implementation details, logic, or architecture depth, use tools to read actual files; do not guess or rely on training knowledge; when multiple targets are nearby, prefer one wider ranged read_file call over many individual reads; no `<workspace>` reference
`MainAgent.stream(session, user_input, permission_callback=None, subagent=None)` selects mode by the `subagent` param: direct mode (`subagent=None`) passes `_recency_turns(session.messages, _RECENCY_N=2)` (last 2 user/assistant pairs, system messages skipped, trailing user input excluded) + current input; spawn mode (`subagent=<Subagent>`) runs cold — `prior = []` + current input only, no recency context, no async/resume
`--debug` active: `stream()` calls `append_debug(session, {"content": system})` before dispatch — the system string written to `.debug.jsonl`

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
- `--debug` — prints `[router: main]` (no subagent) or `[router: <namespace>/<subagent-name>]` (subagent selected) in color `#BA55D3` (medium_orchid) as an OPERATION widget in the TUI chat, 1 line below the user prompt
- `--resume` / `-r` — resume a previous session by ID
- `--working-dir` / `-d` — override working directory (default: cwd)