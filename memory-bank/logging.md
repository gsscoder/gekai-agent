# Event Logging
Always-on JSONL telemetry log, independent from `--lean-telemetry`

## EventLogger (`agent/logging.py`)
Instantiated once per process as `agent.events` (attr on `GekaiAgent`). `run_id` = 8-hex uuid; logger name `gekai.events.{run_id}`.

One file per process run: `~/.gekai/logs/events-{YYYYMMDD}-{HHMMSS}-{run_id}.jsonl`, written via `_JsonFormatter`. If the log dir/file can't be created, falls back to `NullHandler` — never blocks startup.

Record shape:
```json
{"ts": "<utc, persistence.now_utc_str>", "run": "<run_id>", "evt": "<event name>", "level": "info|warning|...", "...fields": "..."}
```
`fields` are the kwargs passed to `emit`, merged flat into the record.

API:
- `emit(evt, *, level="info", **fields)` — fire-and-forget; wrapped in try/except, never raises
- `new_turn() -> str` — 8-hex turn id, increments `turn_count`
- `runtime_s() -> float` — monotonic seconds since construction
- `close()` — closes and removes handlers

## Join Key: `turn_id`
`events.new_turn()` mints a turn id at the start of `_stream` (`tui/app.py`). Stamped as `turn=` on every per-turn event below, and passed through to `process_stream(..., turn_id=turn_id)` → `append_message(session, msg, turn=turn_id)`. This is the cross-reference between `events-*.jsonl` and the `turn` field on `session.jsonl` entries.

Multi-step plans no longer exist (dissolved into the `delegate` tool — see `architecture.md →
Delegate Tool`); there is no `step` field any more. Any specialist work the main agent decides to
run happens as `delegate` tool calls inside its own `harness` event, not as separate per-step events.

## Event Catalog
Process-level (once per run):
| Event | Where | Fields |
|---|---|---|
| `run.start` | `GekaiAgent.__init__` | `version`, `platform`, `core_model`, `supp_model`, `permissions={read,write,exec}`, `verbose_telemetry` |
| `run.exit` | `main.py`, after `app.run()` | `duration_s` (`runtime_s()`), `turn_count`, `reason` (`app.exit_reason`) |

Session-level — `session.start` (emitted on initial mount and on `/clear`, `tui/app.py`): `session` (session id), `resumed` (bool — `False` on `/clear` and on fresh mount, `True` on resume)

Per-turn — all carry `session=`, `turn=`, emitted from `_stream` (`tui/app.py`):
| Event | Fields | Notes |
|---|---|---|
| `turn.start` | `input_len` | |
| `estimate` | `decision` (`chat`/`solo`/`mutate`/`dispatch`/`skipped`), `specialists` (always `[]` — vestigial, plan 33 open point 2), `duration_ms` | emitted by `agent/harness/turn.py::run_step` from the harness's `EstimateEvent`; formerly a separate `route` event (`trivial`/`act`) alongside this one — folded into this single classifier and event, plan 33 |
| `harness` | `outcome` (`ok`/`max_iterations`), `llm_calls`, `prompt_tokens`, `completion_tokens`, `thinking_chars`, `tools` (dict tool→count), `duration_ms`, `budget_exhausted` | `outcome` here is a local `harness_outcome` variable computed in `tui/app.py::_run_step` — `"max_iterations"` iff the loop hit its cap *and* produced no answer text; it is independent of, and not renamed by, the `budget_exhausted` flag below |
| `delegation` | `host`, `delegate`, `namespace`, `status` (`ok`/`failed`), `files`, `files_touched`, `summary_len`, `budget_exhausted` | only when `subagent is not None` (forced `/`-slash route to a subagent); mirrors `SubagentResult`. The `delegate` *tool*'s own nested-agent calls (main agent choosing a specialist mid-turn) are not separately logged here — they surface inside the `harness` event's `tools` count as `delegate` calls |
| `error` | `stage`, `error_type`, `message` | level=`warning`; `stage` is the pipeline stage executing when caught — always `harness` today (the separate `route` stage this used to distinguish was removed along with `Gate`, plan 33) |
| `turn.end` | `outcome`, `duration_ms` | emitted in `finally`, once per turn |

`budget_exhausted` (new, on both `harness` and `delegation`) is a diagnostic signal, not a pass/fail
axis: it is `True` whenever the underlying `Agent` run had to fall back to a forced tool-free
"salvage" completion after exhausting `max_iterations` — this can happen on an otherwise-successful
run (`outcome`/`status` still `ok`) as well as on a genuine failure. It is orthogonal to the
`harness_outcome`/`status` value, which is keyed only on whether feedback (answer text) was produced.

Command — `command` (slash-command dispatch, `tui/app.py`): `session`, `name` (`cmd_name`)

## `outcome` State Machine (`turn.end`)
Starts `"ok"`, last write wins, evaluated in `finally`:
- `"error"` — exception caught (also emits `error`)
- `"max_iterations"` — harness hit iteration cap with no answer chunks
- `"interrupted"` — user cancelled the worker

## Relationship to Session Persistence / `--lean-telemetry`
This log is always-on, regardless of `--lean-telemetry`. Three distinct streams:
- `events-*.jsonl` (this doc) — per-run telemetry, `~/.gekai/logs/`
- `session.jsonl` — visible chat history (`turn`/`command`/`event` entries), see `architecture.md → Session Persistence`
- `debug.jsonl` — internal plumbing (system prompts, `extra_params`, tool calls/results), written by default, suppressed by `--lean-telemetry`

Shared `turn_id` links a `session.jsonl` turn entry (`turn=` field via `append_message`) to its corresponding events in `events-*.jsonl`.