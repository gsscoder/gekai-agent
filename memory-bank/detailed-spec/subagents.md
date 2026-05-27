# Subagent System
Structured async generators that stream typed events to the TUI for live progress rendering

## SubAgent Protocol
Abstract base class in `agent/subagent.py`

```python
class SubAgent(ABC):
    name: str          # class-level; used as TUI badge label
    color: str = ""    # class-level; hex color for the TUI header badge background

    @property
    def description(self) -> str: ...  # optional; shown in header parenthetical

    @abstractmethod
    def run(self) -> AsyncIterator[SubAgentEvent]: ...
```

`run()` is an async generator. Convention enforced by docstring, not type system:
- First yield must be `SubAgentStartEvent` — triggers `SubAgentRenderer` construction and header mount
- Last yield must be `DoneEvent` — triggers progress bar removal and summary line mount
- Intermediate yields: any `SubAgentEvent` subclass in any order

## Event Catalog
All dataclasses inherit from `SubAgentEvent` (itself a no-field dataclass)

| Event | Fields | TUI action |
|---|---|---|
| `SubAgentStartEvent` | `name: str`, `description: str`, `color: str` | construct `SubAgentRenderer`, mount spacer + `HEADER` widget with colored badge |
| `LogEvent` | `message: str`, `tool_name: str` | mount `Static` with `⎿` prefix; in non-debug mode, consecutive calls with same `tool_name` increment a call counter on the existing widget instead of mounting a new one; messages ending in `"..."` are silently dropped |
| `InferEndEvent` | `prompt_tokens: int \| None`, `completion_tokens: int \| None` | `accumulate_tokens()` sums both into `_total_tokens`; displayed in Done summary |
| `StatusUpdateEvent` | `progress: int`, `total: int \| None` | lazily mounts a `ProgressBar` (40% width, no ETA, percentage shown) on first call; subsequent calls update `progress`/`total` |
| `DoneEvent` | _(none)_ | removes progress bar if present, mounts `⎿ Done ({tokens} tokens · {elapsed})` summary line |

## Existing Subagents

### WsExplorer — `agent/ws_explorer/subagent.py`
- `name = "ws-explorer"`, `color = "#008000"`
- `description` — `"Onboarding workspace"` when `.gekai/workspace.json` absent, `"Scan workspace"` otherwise
- Constructor: `__init__(working_dir, client, model)` — support-model client injected by `GekaiAgent.create_ws_explorer()`
- Activation: at startup when `workspace.json` absent; on `/workspace:rebuild`; on `query+plan` intent when workspace may be stale (Y/N confirmation shown first)
- Flow: `scan_workspace()` (sync, thread) → `StatusUpdateEvent(0, total)` → `enrich_workspace()` via async queue draining → `DoneEvent`
- Post-run state: `.workspace` and `.enrichment` attributes populated; caller reads them to update session context
- `.gekai/` directory excluded from stale-detection triggers

### QueryHandler — `agent/handlers/query.py`
- Not a `SubAgent` subclass; `stream()` is an async generator that yields `SubAgentEvent | str`
- `name` implicit: yields `SubAgentStartEvent(name="Query", description="Inspecting workspace", color="#4169E1")`
- Emits only `SubAgentStartEvent`, `LogEvent` (one per `ToolExecutionStarted` bus event), and `DoneEvent`
- After `DoneEvent`, yields a plain `str` with the final LLM answer — the TUI consumer appends this to `answer_chunks`
- Uses `llmstitch` `EventBus` to bridge tool-call events from the agent loop into the subagent event stream
- Shares the main session: `session.messages[1:-1]` used as prior history (excludes current user turn boundary entries)

## Adding a New Subagent
1. Inherit `SubAgent` from `agent/subagent.py`
2. Set `name` (class-level `str`) and `color` (class-level hex string, e.g. `"#8B0000"`)
3. Optionally override `description` property
4. Implement `run()` as `async def run(self) -> AsyncIterator[SubAgentEvent]:`
   - First yield: `SubAgentStartEvent(name=self.name, description=self.description, color=self.color)`
   - Last yield: `DoneEvent()`
5. Wire into `app.py`: instantiate the subagent, iterate `run()` in a worker, dispatch events to a `SubAgentRenderer` instance — follow the pattern in `_init_session()` or `_stream()`

No base class registration — discovery is explicit at call sites

## SubAgentRenderer
Full implementation detail in `tui-layout.md → SubAgentRenderer`. Contract summary:

Consumed event types (anything else is ignored):
- `SubAgentStartEvent` → `renderer.start(name, description, color)`
- `LogEvent` → `renderer.log(message, tool_name)`
- `InferEndEvent` → `renderer.accumulate_tokens(event)` (sync)
- `StatusUpdateEvent` → `await renderer.status_update(event)`
- `DoneEvent` → `await renderer.done()`

One `SubAgentRenderer` instance per subagent block per turn. `debug` flag passed at construction controls `LogEvent` deduplication behavior. `renderer.name` is set by `start()` and used by the streaming worker to track `query_tool_count`