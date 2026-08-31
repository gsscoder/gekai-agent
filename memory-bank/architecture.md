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
- `pipeline/` — `estimate.py` (`ScopeEstimate`, `Estimator` — three-rung scope classifier), `plan.py` (`Task`, `TaskGraph`, `parse_task_graph`, `ROOT_AGENT`), `sequencer.py` (`Sequencer` — builds the `TaskGraph`)
- `subagents/` — `__init__.py` (`Subagent`, `SUBAGENTS`, `build_system_base`, `NAMESPACE_COLORS`) + one namespace package per action domain (`coding/`, `testing/`, `generic/`), each with its own `__init__.py` declaring `namespace`/`namespace_directives` and one file per member subagent
- `tools/` — `__init__.py` (`make_tools`), `catalog.py` (tool-name groups: `READ_TOOLS`/`EDIT_TOOLS`/`FS_TOOLS`/`SHELL_TOOLS`/`ALL_TOOLS`
  — single source of truth for subagent allowlists and `<tools>` prompt generation), `files.py`, `shell.py`, `delegate.py` (`run_subagent` — cold nested-run dispatcher used by the interpreter, see `## Cross-Agent Dispatch`)
- `tui/` (Textual app — see tui-layout.md), `commands/` (slash command registry), `workspace/` (workspace context)

## Session
`Session` in `session.py`; holds a GUID, `messages: list[dict]`, `working_dir`, `permissions`

`messages` starts with two system entries: `ROOT_SYSTEM_PROMPT` at `[0]` + workspace context at `[1]` (TOON-encoded)
Workspace context `<workspace>` block begins with `"verified repository metadata — treat as authoritative for high-level questions:"` preamble line
Injected subset: `workspace_name`, `workspace_type`, `branch`, `primary_languages`, `projects`; `extensions` when `projects` empty; `domain_map` when present

`GekaiAgent.process_stream(session, user_input, permission_callback=None, turn_id=None, hidden_grant_callback=None, append_user=True, seed=None)` is the sole owner of session writes:
`permission_callback: PermissionCallback | None`; `seed` — an explicit `/`-slash agent name, passed through to `Harness.stream()`, which resolves it against `SUBAGENTS` and dispatches that agent directly on the no-graph path (sequencer/task graph bypassed entirely — see `## Estimator` / `## Cross-Agent Dispatch`)
- appends `{"role": "user"}` once per turn before dispatching (when `append_user`)
- appends `{"role": "assistant"}` once per turn after handler completes

## GEKAI.md & Foreign Instruction Files
Plan 35 (v3): `GEKAI.md` is *ingested* (root system base, applies every turn, survives `/compact`);
a foreign file (`AGENTS.md`, `CLAUDE.md`, …) is *only read* (ordinary `read_file` tool call, lands
in message history, is compaction fodder like any other file read) — no `/ingest` command exists,
no `Session` field is ever populated for a foreign file; "ingested" is singular to GEKAI.md by
construction.

`Session.gekai_md: IngestedFile | None` (`session.py`; `IngestedFile = {rel_path, text, sha}`,
`sha` via `agent/directive_audit.py::file_sha` — shared, not rehashed). `GekaiAgent.start_session()`
sets it via `_read_gekai_md()`: reads `{working_dir}/GEKAI.md` if present; a missing file returns
`None`; a read failure (`OSError`/`UnicodeDecodeError`) emits `gekai_md.read_failed` telemetry and
returns `None` — never raises. Fires on every `start_session()` call, including after `/clear`
(TUI's `_clear_session` calls it again) — auto-read is per-session, not per-process.

Injection: `_gekai_md_system_base(system_base, session)` (`agent/harness/core.py`) appends
`session.gekai_md.text` **verbatim** — no truncation/normalization/reordering — under
`<project_instructions source="GEKAI.md">`, beside the dynamic-directive pump append
(`_pumped_system_base`; see `## System Prompt Assembly` in the top-level `docs/architecture.md`).
Both call sites sit in the `subagent is None` branch only — root-only by construction, mirroring
the pump's own root-only rule (plan 28 decision 13); a subagent's `build_system_base()` never
touches this function. This half of the mechanism is unchanged from v2.

**Foreign-file trigger — no prefilter (v3).** Lives entirely in `agent/harness/core.py`'s
`ToolExecutionCompleted` handling: `Harness.stream()`'s `_on_event` calls
`_maybe_flag_foreign_instruction_file(event, queue)` only when `subagent is None` (root dispatch; a
cold subagent's own incidental read never raises this). The function gates on:
`event.call.name == "read_file"`, `not event.result.is_error`, and `path.lower().endswith(".md")`
— routing only, no content judgment. v2's mechanical prefilter
(`agent/instruction_file_detect.py::looks_like_instruction_file`, regex-based) is deleted along
with its tests; every gate hit now fires unconditionally:
`queue.put_nowait(ForeignFileDetectedEvent(rel_path=path, text=event.result.content))`. The TUI
wires that event to `GekaiAgent.start_foreign_file_audit(rel_path, text, callback)`. A
`.py`/`.json`/etc. read never reaches the check; an unflagged (non-`.md`) read costs zero model
calls — but every root-dispatched `.md` read now does reach the check, once, first time (cached
forever after by `file_sha`).

**The audit itself — one question, no corpus (v3).** `agent/directive_audit.py`: `Auditor.audit(file_text)`
— one-shot, non-streaming, `temperature=0`, a single system-prompt question (`AUDIT_PROMPT`,
copied verbatim from plan 35 v3 concept 1): *does this file contain rules, instructions, or
directives intended to influence how an AI coding agent behaves?* — answered `YES`/`NO`, nothing
more. `AuditVerdict(has_directives: bool = False, raw: str = "")` replaces the old
redundant/conflicting/findings shape entirely. First-token parse (`_parse`, mirrors
`pipeline/estimate.py`'s shape) — any unexpected shape or exception falls back to
`AuditVerdict(has_directives=False)`, never raises out of the call. There is no corpus parameter,
no `persona.directive_corpus()`, no `DIRECTIVE_CORPUS_SHA` — v3 deletes all three along with their
tests, and with them the PEP 562 lazy `__getattr__` on `persona.py` that existed only to dodge the
circular import building the corpus required.

Touchpoint: `directive-audit` (`agent/harness/touchpoints.py`) — plain **FAST tier, no policy or
effort override** — the same shape `estimator` and `micro` already use. A one-word answer needs no
reasoning model; v2's SUPP-tier, `effort="high"` operating point is gone.

Cache: `.gekai/directive-audit.json`, keyed by `rel_path`, entry valid when **`file_sha` alone**
matches the current file — v3 drops the `corpus_sha` half of the key entirely, since there is no
corpus left to invalidate against. `load_cached_verdict`/`save_cached_verdict`
(`agent/directive_audit.py`). Off switch: `load_directive_audit_enabled(working_dir)`
(`agent/settings.py`, default `True`, `.gekai/settings.local.json`'s `directive_audit.enabled`) —
checked once, inside `GekaiAgent._start_audit`, before either path fires; GEKAI.md ingestion itself
is untouched by this flag (only the check is gated).

Entry points, both thin wrappers over the shared private `GekaiAgent._start_audit(ingested,
on_verdict)`: `start_directive_audit(session, on_verdict)` (fixed to `session.gekai_md`, no-op if
`None`) and `start_foreign_file_audit(rel_path, text, on_verdict)` (builds a transient
`IngestedFile` — nothing stored on `Session`). `_start_audit` checks the enabled flag, then the
cache synchronously (a local file read, not a model call — a hit calls `on_verdict` immediately, no
"in flight" state); a genuine miss calls `on_verdict(path, None)` (= in flight) then
`asyncio.create_task(self._run_directive_audit(ingested, on_verdict))`, tracked via a strong
`self._background_tasks` set (`add` + `add_done_callback(self._background_tasks.discard)`) so the
task isn't GC'd mid-flight. Every failure inside the task — bad tier config, network error,
anything the `Auditor` itself didn't already swallow — degrades to
`on_verdict(path, AuditVerdict())` (the default `has_directives=False` verdict = no notice, or the
green loaded line for GEKAI.md), never a crash and never a stuck "in flight" line.
`DirectiveAuditEvent` telemetry (`path`, `has_directives`, `cached`, `duration_ms`, `file_bytes`)
fires on every resolution, cached or not — `redundant`/`conflicting` are gone from the event shape.

TUI surface: see `tui-layout.md → Directive Notice`.

## Estimator
`Estimator` (`agent/pipeline/estimate.py`) is a pure **scope classifier** — one FAST-tier LLM call,
temperature 0, per turn, on an ordinal 3-rung scale (`CHAT ⊂ SOLO ⊂ MUTATE`): how much machinery
does this turn need? It does not select a specialist and does not reject unknown agent names;
specialist selection is the sequencer's job, downstream (see `## Cross-Agent Dispatch`). Formerly
two sequential classifiers (a `Gate` intent check plus this estimator); plan 33 folded `Gate`'s axis
into this one call — `Gate`/`Route` no longer exist anywhere in the codebase.

`ScopeEstimate` dataclass: `scope: str = "solo"` — `"chat"` (root solo, no codebase access
needed) | `"solo"` (root solo, codebase available) | `"mutate"` (sequencer + interpreter).

`Estimator.estimate(user_input, history=None) -> ScopeEstimate` — `history` is the session's
`messages`; filtered to `role in ("user", "assistant")` and sliced to the last 6, prepended before
the user message. This is why a follow-up like `"yes"`/`"do it"`/`"same for the other file"`
classifies correctly using prior turns instead of reading as a context-free `chat` — the
highest-risk behavior carried over from `Gate`, which always got history too.

Prompt offers three tokens:
- `CHAT` — answerable with no codebase access: greetings, identity/capability questions,
  acknowledgments, general knowledge unrelated to this workspace; when unsure, `SOLO` instead (an
  over-estimate just answers with the codebase available, harmless; an under-estimate wrongly skips
  needed codebase access)
- `SOLO` — a single small file, a few small edits, or a read/query; no specialist unit of work is
  implied; the fallback rung when unsure between any two rungs
- `MUTATE` — implementation-sized: multiple files/modules, or a distinct unit of work such as a full
  module or a test suite; when unsure between `SOLO` and `MUTATE`, `SOLO` (an over-estimate just lets
  the agent proceed solo, harmless; an under-estimate wrongly skips planning)

Output token → `ScopeEstimate` mapping: `"chat"`/`"solo"`/`"mutate"` (case-insensitive, first
token only) map directly; any unparseable output, or an exception from the call itself, logs a
warning and falls back to `ScopeEstimate(scope="solo")` — the safe middle rung, never `chat`
(would skip needed codebase access) and never `mutate` (would spend a planning call on a greeting).

`Harness.stream()` (`agent/harness/core.py`) is the sole consumer: `mutate` routes into
`_stream_graph()`; `chat` and `solo` both run root directly, no graph. An explicit `seed` bypasses
the Estimator entirely (decision `"dispatch"`) and also runs the no-graph path, but with the
named subagent bound in place of root — see `## Cross-Agent Dispatch`. The `chat` rung additionally drives dispatch (plan 34): when the caller passed no explicit
`extra_params` override, `chat` resolves `effective_extra_params =
resolve_thinking_params(root_dispatch_resolved.model, enabled=False)` — root's model's explicit
thinking-*disable* payload (e.g. `{"extra_body": {"thinking": {"type": "disabled"}}}` for a
DeepSeek-style model), not a bare `{}`. A bare `{}` means "unspecified" to a provider like
DeepSeek, which then defaults to reasoning ON — plan 34 phase 1 found this live: root's classifier
touchpoint (`Estimator`, FAST tier) was reasoning on every call despite being bound
`thinking=False`, because nothing rendered that `False` into an actual provider parameter. `chat`
also dispatches with `tools_override=frozenset()` — no tools registered for that turn (see
`## Cross-Agent Dispatch` / `_build_agent`). `solo` and `mutate` use
`root_dispatch_resolved.extra_params` (the SUPP/CORE tier's normal params, unaffected) — this is
the "don't burn reasoning on a greeting" capability `Gate`'s `trivial` flag used to gate, now keyed
off the Estimator's own rung instead.

Root's own no-tool-schema reasoning suppression on the `chat` rung comes from this same
explicit-disable `extra_params` mechanism — not from having fewer (or no) tools registered. A live
measurement (10 runs per config) found tool-schema presence has no causal effect on whether root
reasons: tools-present+bare-`{}` reasoned 3/10 (stochastic), tools-removed+bare-`{}` reasoned
10/10 (worse), and both tools-present+explicit-disable and tools-removed+explicit-disable (the
shipped state) reasoned 0/10. Dropping the `chat` rung's tool schemas (`tools_override=frozenset()`)
is a separate, purely prompt-token-reduction win (measured: `"hi"` dropped from ~2116 to 391
prompt tokens), not a reasoning-suppression mechanism.

Verbatim file output is handled by the `<file_handling>` rule in `ROOT_SYSTEM_PROMPT`, not a dedicated pipeline stage.

## Cross-Agent Dispatch
There is no `delegate` tool and no `Harness`-registered tool for calling another agent — root
cannot self-spawn, and no subagent can spawn anything (plan 27 decision 11, superseding plan 25's
agents-as-tools). All cross-agent dispatch is owned by the sequencer + the fixed interpreter, not
by an LLM deciding mid-turn to call a tool:

1. `Harness.stream()` estimates the turn (`Estimator`); a `mutate` estimate routes into
   `Harness._stream_graph()`. Everything else — including an explicit `seed`, which skips the
   Estimator entirely — runs at the `root-dispatch` touchpoint, no graph: root by default, or the
   `seed`'s named subagent (warm session context, same as root) when one is given.
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
1. `_run_step(user_input, seed, ...)` dispatches straight to `harness_turn.run_step()` →
   `GekaiAgent.process_stream()` → `Harness.stream()` — there is no separate pre-classification
   stage; the Estimator call happens inside `Harness.stream()` itself (see `## Estimator`)
2. no locate/rewrite stage, no rejection path; an unrecognized `/`-slash agent name is a CLI/palette
   concern, not something the harness or `Estimator` handles

## Root
Root is the session-owning unit — deployed first by the harness, present for the whole turn, never
a task-graph step (`ROOT_AGENT` fails `parse_task_graph`'s roster check by construction: it is not
in `SUBAGENTS`, so no `agent` value equal to `"root"` can ever pass `auto_assignable` validation).
It is not a registry `Subagent` and is not in the sequencer roster or palette — it is
special-cased in `Harness`, not filtered out of two separate lists.

Root owns two things:
- **Warm context.** `_recency_turns(session.messages, _RECENCY_N=2)` (last 2 user/assistant pairs)
  is passed on every no-graph `Harness.stream()` call — root as well as a `seed`-dispatched
  subagent — and on the synthesis call below. Only a graph-spawned subagent step (`run_subagent`,
  called from inside `_stream_graph()`) runs cold (`prior=[]`).
- **The turn's final answer.** `Harness._respond()` (`agent/harness/core.py`) is a real root call
  at the `root-dispatch` touchpoint — session recency + the graph's `summary` + every
  `StepResult.output`, prompted to write the reply the user sees. This runs after
  `run_task_graph()` finishes (success) or raises `TaskGraphHalted` (root also narrates the halt,
  naming the step and reason). There is no separate `Responder` unit or touchpoint — root absorbed
  it. Fail-soft: any exception during synthesis (empty text, model/network error) falls back to
  `_recap()`, a mechanical (no LLM call) summary built from `graph.summary` and the halt info alone,
  so a turn is never lost to its own wrap-up. A `seed`-dispatched subagent turn never reaches this
  — it skips synthesis entirely and yields its own completed history's last assistant text
  directly, the same as root's no-graph path.

`_ROOT_DIRECTIVES` ("ask before acting on an ambiguous request") is safe specifically because root
is never dispatched cold inside a graph — it is the only unit ever facing a human, so "ask" is
always answerable.

## Post-Turn Review (removed)
No automatic post-turn review exists. Both branches of `Harness.stream()` — the no-graph path
(`chat`/`solo`/`seed`, via `_stream_solo()`) and the graph path (`mutate`, via `_stream_graph()`) —
are plain passthrough loops: whatever `files_touched`/final-answer/`budget_exhausted` values either
path collects go straight back to the caller, with no fresh-eyes pass over the turn's own diff.

Formerly `Harness._maybe_review()` ran here, dispatching a read-only `change-reviewer` subagent,
repairing any `FINDINGS` verdict via `code-fixer`, re-reviewing, and surfacing a
`ReviewFindingsEvent`/`TurnResult.review_report` warning to the TUI on a still-broken second
verdict. That subagent, dataclass, and all wiring through `harness/turn.py` and `agent/tui/app.py`
are deleted; no `review`-sourced session entry can be written anymore (see `## Session Persistence`
below).

## LLM Integration
`openai` SDK (`AsyncOpenAI`) for chat and classification; `llmstitch` (`agent.llm.Agent`) for the tool-calling loop in `Harness`
Every touchpoint (`estimator`/`sequencer`/`root-dispatch`/`subagent-dispatch`/`micro` —
`agent/harness/touchpoints.py`) resolves to a model through the FAST/SUPP/CORE tier catalog +
bindings (`agent/settings.py::load_model_catalog`/`load_tier_bindings`), configured via the TUI's
`/tiers` grid — not raw env vars. `load_model_catalog()` always returns a fresh dict built directly
from `agent/llm/tiers.py::DEFAULT_MODEL_CATALOG` — capability facts (efforts, thinking, base_url,
suitability) are never copied to disk, so a newly-added model shows up in `/tiers` immediately for
existing installs. The only thing persisted at user level for tiers is `TierBinding` (which model +
effort + thinking each of FAST/SUPP/CORE points to), under settings.json's `"tiers"` node. `estimator` runs at a bare FAST (no mobility); `sequencer`
defaults to CORE (mobile down to SUPP); `root-dispatch`/`subagent-dispatch` default to SUPP (mobile
up to CORE) — see `harness/scaling.py` for the per-call tier-scaling mechanism.

**Tier binding vs touchpoint operating point.** A tier binding (`/tiers`, user config) decides
WHICH model + credential fills a capability slot (FAST/SUPP/CORE); a `Touchpoint` (code,
`agent/harness/touchpoints.py`) decides HOW that model is operated for its specific job —
`Touchpoint.effort`/`.thinking`, `None` meaning "inherit the binding's value" (every touchpoint
but `sequencer` today). `resolve_tier(tier, ..., touchpoint_name=...)` (`agent/llm/resolve.py`)
applies the touchpoint's override to `extra_params` only; the credential lookup (`credentials.credential_key`)
always keys on the *binding's* configured `default_effort`/`thinking`, never the override — an
override must not require a fresh `/tiers` entry to have a stored key. `sequencer` is the standing
case: it inherits CORE's model/credential but declares `effort="high", thinking=False` on itself,
because a live probe found CORE's own thinking-on default cost ~90s median to sequence a small JSON
task graph vs ~13s with thinking off, at equal-or-better plan quality — and fixing that via `/tiers`
would mean turning CORE's thinking off for every CORE-tier call, not just sequencing. `TierPolicy`
(`agent/llm/tiers.py`) documents the same split: effort/thinking default to the binding's values but
are a per-touchpoint axis, not a per-workload `/tiers` knob.

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

`Harness.stream(session, user_input, permission_callback=None, subagent=None, extra_params=None, hidden_grant_callback=None, seed=None)`
selects `system_base` by the `subagent` param — `ROOT_SYSTEM_PROMPT` (root, no-graph path) vs
`subagent.build_system_base()` (spawn, cold nested run) — then calls `_build_agent`. `extra_params`
resolution in the no-graph path (see `## Estimator`): an explicit argument always wins; otherwise
the Estimator's `chat` rung yields `resolve_thinking_params(root_dispatch_resolved.model,
enabled=False)` (root's model's explicit thinking-disable payload, plan 34 — not a bare `{}`) and
every other rung yields `root_dispatch_resolved.extra_params` (the root-dispatch tier's resolved
params). Direct mode passes
`_recency_turns(session.messages, _RECENCY_N=2)` (last 2 user/assistant pairs, system messages
skipped, trailing user input excluded) + current input; spawn mode runs cold — `prior = []` +
current input only, no recency context, no async/resume.
Written by default (`--lean-telemetry` suppresses it): `stream()` calls
`append_debug(session, {"content": {"system": agent.system, "extra_params": effective_extra_params}})`
**after** `_build_agent` returns (`agent.system` is the true assembled prompt, a mutable field on
`llmstitch.Agent`; `effective_extra_params` is the value actually used this turn) — written to `.debug.jsonl`.

The no-graph root path also streams answer text live (plan 34 Phase 2): `stream()`'s `_on_event`
bridge passes `emit_text_chunks=True` into `_bridge_llm_event`, so each provider `TextDelta`
(`agent/llm/agent.py::_run_loop`) becomes a `TextChunkReceived` bus event and then a
`TextChunkEvent` the TUI renders incrementally. `_stream_graph()` never sets `emit_text_chunks=True`,
so a graph-routed (`mutate`) turn emits zero `TextChunkEvent`s — its answer only ever appears as the
one final assembled string, same as before plan 34. Either way the persisted answer comes from the
completed history, never from the streamed chunks (display-only). See `subagents.md → Harness` for
the full mechanism.

## Session Persistence
Sessions stored as JSONL at `~/.gekai/workspaces/{normalized-repo-path}/{session-id}.jsonl`; each line is a timestamped entry with a `kind` field:

```
{ts, kind:"turn",    role:"user|assistant|system", content}   ← LLM context; only these fed to model / /compact
{ts, kind:"command", content:"/clear"}                        ← slash command typed by user
{ts, kind:"event",   source:"...", content:"..."}             ← system-side non-LLM: command, error, interrupted, max_iterations
{ts, kind:"compact", content:"<summary>"}                     ← /compact boundary; supersedes every turn before it
```

Entries without `kind` (legacy files) default to `"turn"`.

**Boundary — session vs debug:**
`session.jsonl` = everything the user saw on screen (turns + commands + events). Litmus: *did the user see it?*
`debug.jsonl` = internal plumbing (system prompts, `extra_params`, tool calls/results) — written by default, suppressed by `--lean-telemetry`, never for visual rebuild.

**Writers:** `append_message(session, msg)` → `kind:"turn"`; `append_command(session, text)`; `append_event(session, content, source)`; `append_compact(session, summary)` → `kind:"compact"`.

**Two readers:**
- `load_session(id)` → `(session_id, working_dir, turns_only)` — only `kind=="turn"` entries (model context). If a `kind=="compact"` entry exists, only entries after the *last* one are read, and the compact's `content` is seeded as the first (synthetic `user`-role) message — every turn before the boundary is dropped. Always-fresh system messages (ROOT_SYSTEM_PROMPT, workspace) excluded and re-injected on startup.
- `load_timeline(id)` → `(working_dir, all_entries)` — full ordered list for visual rebuild; non-persistent system turns excluded; `compact` entries pass through unfiltered so resume shows the boundary.

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
- `/compact [instructions]` — summarizes the transcript through the `"micro"` touchpoint and replaces `session.messages` with `[system, synthetic-user-summary]`; optional free-text instructions steer what the summary focuses on. Blocked while a turn is streaming. Auto-triggers with no instructions and no opt-out once transcript size (chars/4, `agent/tui/app.py::_estimate_session_tokens`) crosses 80% of `_context_limit`; a sticky status-bar hint warns at 75%. See `agent/compact.py` (`context_state`/`summarize`/`apply_summary`) and `## Session Persistence` for the `kind:"compact"` boundary.

## CLI Flags
- `--lean-telemetry` — suppresses the `.debug.jsonl` log (assembled system prompt, `extra_params`,
  every tool call/result via `append_debug`, see `## Session Persistence`); written by default when
  the flag is absent (`verbose_telemetry=True` threaded through `GekaiAgent`/`Harness`/
  `DispatchContext`). The Estimator's `chat`/`solo`/`mutate` decision is separately emitted as an
  `"estimate"` telemetry event (`agent/harness/turn.py::run_step`), not rendered in the TUI
- `--resume` / `-r` — resume a previous session by ID
- `--working-dir` / `-d` — override working directory (default: cwd)