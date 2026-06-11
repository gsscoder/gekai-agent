# Event Logging
Always-on JSONL telemetry log, independent from `--debug`

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

## Event Catalog
Process-level (once per run):
| Event | Where | Fields |
|---|---|---|
| `run.start` | `GekaiAgent.__init__` | `version`, `platform`, `core_model`, `supp_model`, `permissions={read,write,exec}`, `debug` |
| `run.exit` | `main.py`, after `app.run()` | `duration_s` (`runtime_s()`), `turn_count`, `reason` (`app.exit_reason`) |

Session-level — `session.start` (emitted on initial mount and on `/clear`, `tui/app.py`): `session` (session id), `resumed` (bool — `False` on `/clear` and on fresh mount, `True` on resume)

Per-turn — all carry `session=`, `turn=`, emitted from `_stream` (`tui/app.py`):
| Event | Fields | Notes |
|---|---|---|
| `turn.start` | `input_len` | |
| `route` | `decision` (`_route_decision(route)`), `duration_ms` | decision: `main` / `trivial` / `rejected` / `<namespace>/<subagent>` |
| `locate` | `files`, `duration_ms` | skipped for `route.trivial` (no locate stage) |
| `gate` | `areas` (`count_blast_areas`), `limit` (`session.blast_radius_limit`), `passed` | only when `route.subagent is not None` |
| `rewrite` | `ok=True`, `duration_ms` | only when `entries` non-empty |
| `harness` | `outcome` (`ok`/`max_iterations`), `llm_calls`, `prompt_tokens`, `completion_tokens`, `thinking_chars`, `tools` (dict tool→count), `duration_ms` | |
| `error` | `stage`, `error_type`, `message` | level=`warning`; `stage` is the pipeline stage executing when caught (`route`/`locate`/`gate`/`rewrite`/`harness`) |
| `turn.end` | `outcome`, `duration_ms` | emitted in `finally`, once per turn |

Command — `command` (slash-command dispatch, `tui/app.py`): `session`, `name` (`cmd_name`)

## `outcome` State Machine (`turn.end`)
Starts `"ok"`, last write wins, evaluated in `finally`:
- `"rejected"` — router rejected non-English input
- `"gate_blocked"` — blast-radius gate blocked
- `"error"` — exception caught (also emits `error`)
- `"max_iterations"` — harness hit iteration cap with no answer chunks
- `"interrupted"` — user cancelled the worker

## Relationship to Session Persistence / `--debug`
This log is always-on, regardless of `--debug`. Three distinct streams:
- `events-*.jsonl` (this doc) — per-run telemetry, `~/.gekai/logs/`
- `session.jsonl` — visible chat history (`turn`/`command`/`event` entries), see `architecture.md → Session Persistence`
- `debug.jsonl` — internal plumbing (system prompts, route tokens, locate list, rewritten text), `--debug` only

Shared `turn_id` links a `session.jsonl` turn entry (`turn=` field via `append_message`) to its corresponding events in `events-*.jsonl`.
