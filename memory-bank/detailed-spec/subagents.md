# Subagent System
Structured async generators that stream typed events to the TUI for live progress rendering

## AgentEvent Protocol
There is no enforced base class — `agent/subagent.py` holds only the `AgentEvent` taxonomy
(plain no-field dataclass + its typed subclasses). The `SubAgent(ABC)` protocol class that used
to live there was deleted as dead code once its only subclass was removed; nothing inherits from
it today.

The convention lives on **by usage, not by enforcement**: anything that streams typed progress to
the TUI is an `AsyncIterator[AgentEvent | str]` generator following this shape:
- First yield must be `SubAgentStartEvent` — triggers `SubAgentRenderer` construction and header mount
- Last yield must be `DoneEvent` — triggers progress bar removal and summary line mount
- Intermediate yields: any `AgentEvent` subclass in any order

`Harness.stream()` is the live example — see [Harness](#harness) below.

## Event Catalog
All dataclasses inherit from `AgentEvent` (itself a no-field dataclass)

| Event | Fields | TUI action |
|---|---|---|
| `SubAgentStartEvent` | `name: str`, `description: str`, `color: str` | construct `SubAgentRenderer`, mount spacer + `HEADER` widget with colored badge |
| `LogEvent` | `message: str`, `tool_name: str` | mount `Static` with `⎿` prefix; in non-debug mode, consecutive calls with same `tool_name` increment a call counter on the existing widget instead of mounting a new one; messages ending in `"..."` are silently dropped |
| `InferEndEvent` | `prompt_tokens: int \| None`, `completion_tokens: int \| None` | `accumulate_tokens()` sums both into `_total_tokens`; displayed in Done summary |
| `StatusUpdateEvent` | `progress: int`, `total: int \| None` | lazily mounts a `ProgressBar` (40% width, no ETA, percentage shown) on first call; subsequent calls update `progress`/`total` |
| `DoneEvent` | _(none)_ | removes progress bar if present, mounts `⎿ Done ({tokens} tokens · {elapsed})` summary line |

## Existing Streamers

### Harness — `agent/harness/core.py`
- Not a class hierarchy member of anything — `stream(session, user_input, permission_callback=None, subagent: Subagent | None = None, extra_params: dict | None = None)` is an async generator that yields `AgentEvent | str`, the live example of the protocol-by-convention above
- One method, two modes selected by the `subagent` param — both run the same no-graph branch, both get warm session context:
  - **root, no-graph** (`subagent=None`, `chat`/`solo`-estimated single-agent turn): system = `ROOT_SYSTEM_PROMPT + "\n<tools>\n" + render_tool_instruction(...)`; prior context = `_recency_turns(session.messages, _RECENCY_N=2)` (last 2 user/assistant pairs) + current input — root's warm session context; `SubAgentStartEvent(name="root", description="thinking", color="#4169E1")`. On the `chat` rung specifically (plan 34 Phase 3), `_build_agent` is called with `tools_override=frozenset()` — no tools registered, `render_tool_instruction([])` renders nothing, so the `<tools>` block is empty; the system prompt itself is otherwise unchanged (`solo` and `mutate` still register the full tool set)
  - **seed-dispatched subagent** (`subagent=<Subagent>`, resolved from an explicit `/`-slash `seed` — the only way `stream()`'s own `subagent` param is ever bound; a graph-spawned step never calls `stream()` at all, it goes through `run_subagent()` instead, cold): handles a subagent's specialty (e.g. `code-expert` for **substantial or specialized code work** — features, fixes, behavior-changing rewrites; owns its assigned task's implementation in full); system = `subagent.build_system_base()` + `<tools>` appended by `_build_agent` (tool-ceiling clamp via `tool_scope(subagent.tool_policy, None)`); prior context = the same `_recency_turns` + current input as root — warm, not cold — with no directive-pump/GEKAI.md injection (root-only); `SubAgentStartEvent(name=subagent.name, description=subagent.description, color="#4169E1")`
- Emits `SubAgentStartEvent`, `LogEvent` (one per `ToolExecutionStarted` bus event), `DiffEvent` (on `edit_file` completion when `old_str != new_str`), `InferEndEvent`, `ThinkingTokenEvent`, `TextChunkEvent` (root no-graph path only — see below), `MaxIterationsEvent` (iteration-limit path), and `DoneEvent`
- After `DoneEvent`, yields a plain `str` with the final LLM answer — the TUI consumer appends this to `answer_chunks`. This final assembly is unconditional (from the completed history's `TextBlock`s) regardless of whether chunks streamed; streaming is a display-only side channel, never the source of the persisted answer.
- **Live text streaming is path-specific (plan 34 Phase 2), not blanket.** Root's no-graph direct-dispatch path (`Harness.stream()`'s non-graph branch) passes `emit_text_chunks=True` to `_bridge_llm_event`, so each `TextDelta` from the provider becomes a `TextChunkReceived` bus event (`agent/llm/agent.py::_run_loop`) and then a `TextChunkEvent` the TUI renders incrementally (`_run_step`'s `_on_event` in `agent/tui/app.py`
appends each chunk's text to a running `_streamed_answer` and mounts/updates a live `MessageWidget`
in place, reset per turn). The graph/subagent-step path (`Harness._stream_graph`) never sets `emit_text_chunks=True` — a graph-routed turn emits zero `TextChunkEvent`s (would otherwise interleave with `LogEvent`/`DiffEvent` mid-transcript), and its answer still only appears as the final assembled string.
- Uses `llmstitch` (`agent.llm.Agent`) `EventBus` to bridge tool-call events from the agent loop into the typed event stream
- `_build_agent()` registers tools from `make_tools(working_dir)`, filtered by `subagent.tools` allowlist when set, computes the effective permission overlay (AND of `session.permissions` and `subagent.permissions`); no cross-agent tool is ever registered, for root or any subagent (see `architecture.md → Cross-Agent Dispatch`)

> **Sequencer → interpreter flow, not delegate-driven decomposition:** the `Estimator` only
> classifies scope, on a `chat`/`solo`/`mutate` ordinal scale (see `architecture.md → Estimator`;
> `Gate`, a separate classifier this once ran alongside, was folded into the `Estimator` and deleted,
> plan 33). A `mutate`-estimated turn routes into
> `Harness._stream_graph()`: `Sequencer.sequence()` makes one CORE-tier call (CORE's model/credential,
> but the `sequencer` touchpoint's own `effort="high", thinking=False` — see
> `architecture.md → LLM Integration → Tier binding vs touchpoint operating point`) and returns a validated
> `TaskGraph` where every step is assigned to an `auto_assignable` roster specialist — never to
> root (`parse_task_graph` rejects `ROOT_AGENT` as a step `agent`). `agent/harness/interpreter.py`'s
> `run_task_graph()` then walks the graph deterministically (`execute → verify → repair → halt`),
> calling `run_subagent()` for each step; no agent decides mid-turn whether to dispatch another
> agent. Root owns the turn around this execution — deployed first, present the whole time via its
> warm session context — and synthesizes the final answer from the graph's `summary` + every step's
> output once the walk finishes (or halts). See `architecture.md → Root` for the full synthesis
> contract (`Harness._respond()`, fail-soft to `_recap()`).

> **Subagent vs harness-worker:** `Subagent` (and the streamers spawned for it) serve a
> *user-turn* — assigned by the sequencer and walked by the interpreter, or dispatched directly by
> an explicit `/`-slash `seed` (bypassing the sequencer/interpreter entirely). No `worker`-namespace, system-managed unit exists in the codebase today (no
> `agent/subagents/worker/` directory, no `worker` entry in `NAMESPACE_COLORS`) — a prior draft of
> this doc described a `ws-manager`/`ws_manager.py` unit in that role; it was never built. A
> `harness-worker` category serving the system/lifecycle directly (e.g. a workspace-scan run
> outside any single user turn) remains a possible future addition with zero code today.

## Generic Namespace
`agent/subagents/generic/` is no longer dormant. `namespace_directives` (rank 0,
`agent/subagents/generic/__init__.py`) is the cold-dispatch contract every dispatched unit needs:
never ask a clarifying question — state the assumption and proceed; stay inside the step's
boundary; report to the next step, not a person; name the blocker plainly rather than emit a
partial result that reads as done. `omni-worker` (`agent/subagents/generic/omni_worker.py`) is its
one member — the residual specialist for a task-graph step nothing else owns: scaffolding, project
layout, manifests, config/CI files, docs, data/asset files, dependency/build chores,
investigation-that-must-produce-a-finding. `user_invocable=False` (excluded from the palette, per
the `NAMESPACE_COLORS` comment: a namespace with no invocable members is innate —
no selector to build), `auto_assignable=True` (routable by the sequencer), full tool ceiling with a
per-step `tool_policy` for narrowing, `directive_domains=("*",)` — the one subagent that composes
every other namespace's directives (rank-ordered) into its own, since a step can land it in any
domain. The four coding/testing specialists (`code-expert`, `code-refactorer`, `test-expert`,
`test-fixer`) declare `directive_domains=("generic",)`, composing the same cold-dispatch contract
into their own directives (`Subagent._compose_directives`, `agent/subagents/__init__.py`).

## Adding a New Subagent-Style Streamer
There is no base class to inherit — any async generator yielding `AgentEvent`s following the
start/done convention qualifies. To add one:
1. Implement an `async def stream(...) -> AsyncIterator[AgentEvent | str]:` (or similarly named) generator
   - First yield: `SubAgentStartEvent(name=..., description=..., color=...)`
   - Last yield: `DoneEvent(...)`
   - Intermediate yields: any `AgentEvent` subclass, in any order
2. Bridge tool/inference events into the typed stream via `EventBus` if the streamer runs an `Agent` tool loop — follow `Harness.stream()`'s `_consume_bus()` pattern
3. Wire into `app.py`: iterate the generator in a worker, dispatch events to a `SubAgentRenderer` instance — follow the pattern in `_stream()`

No registration mechanism — discovery is explicit at call sites.

To add a new **`Subagent`** (the routable specialist entity, distinct from the streamer
protocol above): drop one file in `agent/subagents/` exporting a module-level `subagent =
Subagent(...)`; `_discover()` picks it up automatically — see `architecture.md → Subagent Routing`.

## SubAgentRenderer
Full implementation detail in `tui-layout.md → SubAgentRenderer`. Contract summary:

Consumed event types (anything else is ignored):
- `SubAgentStartEvent` → `renderer.start(name, description, color)`
- `LogEvent` → `renderer.log(message, tool_name)`
- `InferEndEvent` → `renderer.accumulate_tokens(event)` (sync)
- `StatusUpdateEvent` → `await renderer.status_update(event)`
- `DoneEvent` → `await renderer.done()`

One `SubAgentRenderer` instance per subagent block per turn. `debug` flag passed at construction controls `LogEvent` deduplication behavior. `renderer.name` is set by `start()` and used by the streaming worker to track `query_tool_count`