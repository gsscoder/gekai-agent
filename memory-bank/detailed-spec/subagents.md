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
- One method, two modes selected by the `subagent` param:
  - **direct** (`subagent=None`): handles the **generalist default** — general or simple requests, reading/explaining/running code, and all general/glue/scaffolding work; owns decomposition itself, calling the `delegate` tool as many times as it judges necessary; system = `SYSTEM_PROMPT + "\n<tools>\n" + render_tool_instruction(...)`; prior context = `_recency_turns(session.messages, _RECENCY_N=2)` (last 2 user/assistant pairs) + current input; `SubAgentStartEvent(name="main", description="thinking", color="#4169E1")`
  - **spawn** (`subagent=<Subagent>`): handles a subagent's specialty (e.g. `code-expert` for **substantial or specialized code work** — features, fixes, behavior-changing rewrites; owns its assigned task's implementation in full); system = `subagent.build_system_base()` + `<tools>` appended by `_build_agent`; prior context = `[]` (cold — no recency, no inheritance, no async/resume); `SubAgentStartEvent(name=subagent.name, description=subagent.description, color="#4169E1")`
- Emits `SubAgentStartEvent`, `LogEvent` (one per `ToolExecutionStarted` bus event), `DiffEvent` (on `edit_file` completion when `old_str != new_str`), `InferEndEvent`, `ThinkingTokenEvent`, `MaxIterationsEvent` (iteration-limit path), and `DoneEvent`
- After `DoneEvent`, yields a plain `str` with the final LLM answer — the TUI consumer appends this to `answer_chunks`. There is no live token-by-token text streaming; the final answer is assembled once from the completed history's `TextBlock`s
- Uses `llmstitch` (`agent.llm.Agent`) `EventBus` to bridge tool-call events from the agent loop into the typed event stream
- `_build_agent()` registers tools from `make_tools(working_dir)`, filtered by `subagent.tools` allowlist when set, computes the effective permission overlay (AND of `session.permissions` and `subagent.permissions`), and — direct mode only — registers the `delegate` tool (see `architecture.md → Delegate Tool`)

> **Decomposition model:** there is no more router-level plan. `Gate` only classifies intent
> (`TRIVIAL`/`REJECTED`/`ACT` — see `architecture.md → Gate`); once a turn reaches `Harness` in
> direct mode, the main agent decides for itself whether the request needs a specialist, and calls
> `delegate(agent, task)` mid-turn, as many times as needed, in dependency order (scaffold → logic
> → tests). Example: "create a Python module that converts HTML to Markdown, scaffold a
> conventional structure with requirements files, add minimal test coverage" — main scaffolds
> directly, then calls `delegate("code-expert", ...)` for the converter and `delegate("test-expert",
> ...)` for the suite, sequencing the calls itself instead of a router pre-computing steps.

> **Subagent vs harness-worker:** `Subagent` (and the streamers spawned for it) serve a
> *user-turn* — selected via the `delegate` tool from the main agent's own judgment (or forced by a
> `/`-slash command). `ws-manager` (`agent/subagents/worker/ws_manager.py`,
> namespace `worker`) is a system-managed worker: `user_invocable=False`, never routed or surfaced
> in the menu/palette. It is dispatched only via `run(task, ...)` (currently `onboard` →
> `build_index`), a direct call that bypasses the LLM/spawn path — so its `description` exists only
> for the registry, and it carries no mandate/directives/tools. A separate, still-unbuilt
> `harness-worker` category would serve the system/lifecycle directly (e.g. a future workspace-scan
> revival) — work that runs outside any single user turn; zero code exists for this category yet.

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