# TUI Layout
Textual-based full-screen terminal UI; fixed footer with command palette and input, scrollable conversation body

## Package Layout
`agent/tui/` groups the full-screen shell: `app.py` owns app lifecycle, stream worker, subagent rendering (`SubAgentRenderer`), and command dispatch; `widgets.py` and `palette.py` hold the reusable display components; `permissions.py` handles first-run modal flow.

## Screen Structure
```
┌─ ScrollableContainer #conversation ──────────────────┐  1fr, no scrollbar
│  MessageWidget (BANNER)                              │
│  MessageWidget (SYSTEM)  version / dir / branch      │
│  MessageWidget (USER)    ❯ input text                │
│  MessageWidget (ASSISTANT)  ● markdown response      │
│  MessageWidget (OPERATION)  * PastVerb for Xs        │
│  ...                                                 │
├─ Container #footer ───────────────────────────────── │  dock: bottom, height: auto
│  Static #question-bar  Y/N confirmation prompt       │  hidden by default; shown for rescan / inline questions
│  Static #status-line   spinner + verb + elapsed      │  hidden when idle
│  Static #status-spacer blank line below status       │  hidden when idle
│  CommandPalette #command-palette                     │  hidden when input ≠ /…
│  Static #hint-area     transient hint text           │  hidden by default; right-aligned, grey
│  Container #input-area                               │  height: 3, layered
│    Static #prompt-marker   ❯                         │  layer: marker, absolute
│    Input  #prompt                                    │  layer: input, padding-left: 2
│  Static #context-bar       X.X% context              │  height: 1, left-aligned, grey, padding-left: 2
└──────────────────────────────────────────────────────┘  padding-bottom: 1
```
All widgets use `background: ansi_default` / `color: ansi_default` for terminal transparency

## MessageWidget
`Widget` subclass; `kind` selects rendering branch

| Kind        | Rendering                                                        |
|-------------|------------------------------------------------------------------|
| `BANNER`    | `[cyan]{pyfiglet ASCII}[/cyan]`                                 |
| `SYSTEM`    | `[dim]{text}[/dim]`                                             |
| `USER`      | `❯ {text}` on dark background (`#3a3a3a`)                      |
| `ASSISTANT` | `[cyan]●[/cyan]` (width 2) + `Markdown` (1fr), horizontal layout |
| `OPERATION` | `[{color}]* {past_verb} for {duration}[/{color}]`              |
| `HEADER`    | `[cyan]●[/cyan]` dot + plain `Static` text, horizontal layout   |

`.update(content)` and `.append_text(chunk)` refresh the inner child widget in place

## Startup Flow
1. `main.py` resolves permissions, creates `GekaiApp`, calls `.run()`
2. `on_mount`: if `needs_permissions` → `push_screen(PermissionScreen, callback=_on_permission_selected)` else → `await _init_session()`
3. `_on_permission_selected(choice)`: resolves + saves permissions, runs `_init_session()` via worker
4. `_init_session()`: mounts BANNER + SYSTEM widgets; if `workspace.json` absent runs `WsExplorer` via `SubAgentRenderer` (same event loop as streaming); loads or builds workspace, starts session, replays restored messages as USER/ASSISTANT widgets; updates `#context-bar` from session message estimate

## Input Handling
`on_input_submitted`:
- `_pending_yesno` future set → interpret value as Y/N (`y`/`yes` = `True`), hide `#question-bar`, resolve future, return
- `_pending_question` future set → treat value as free-text answer, mount USER widget with answer, hide `#question-bar`, resolve future, return
- palette visible → take `palette.selected_command`, hide palette, override `stripped`
- `stripped` starts with `/` → slash command branch; `workspace:rebuild` runs `_rebuild_workspace()` worker; others dispatched via `CommandRegistry`
- else → mount USER widget, run `_stream(user_input)` as exclusive worker

`on_input_changed`:
- value starts with `/` → `palette.filter(value[1:])`
- else → `palette.hide()`

`on_key`:
- palette visible + Up/Down → `palette.move_up()` / `palette.move_down()`, `event.stop()`
- prompt not focused + printable → redirect keystroke to `#prompt`

## Streaming Worker (`_stream`)
Runs as exclusive Textual worker.
- picks random operative verb pair and accent color for the turn
- normalizes input via `agent.normalize`, classifies via `agent.classify`, then iterates `agent.process_stream`
- if any `QUERY+plan` segment detected, calls `_maybe_rescan_workspace` before streaming starts
- `str` item → append to `answer_chunks`
- `SubAgentEvent` items dispatched to a `SubAgentRenderer` instance:
  - `SubAgentStartEvent` → construct renderer, mount spacer + `HEADER` widget
  - `LogEvent` → mount tool-call line (deduplicated by `tool_name` in non-debug mode)
  - `InferEndEvent` → accumulate prompt/completion tokens
  - `StatusUpdateEvent` → mount / update `ProgressBar` widget
  - `DoneEvent` → remove progress bar, mount summary line with tokens + elapsed
- on complete: mount ASSISTANT widget + OPERATION widget (with tool count if > 0), scroll end
- `finally`: `_stop_status_animation()`, `_worker = None`, `_focus_prompt()`

## SubAgentRenderer
Helper class in `app.py`; one instance per subagent block within a turn.
- `start(name, description, color)` — mounts `assistant-spacer` Static then a `HEADER` MessageWidget with colored name badge
- `log(message, tool_name)` — mounts a `Static` with `⎿` (first item) or space prefix; in non-debug mode deduplicates consecutive calls from the same `tool_name` by incrementing a call count on the existing widget
- `status_update(event)` — lazily mounts a `ProgressBar` (40% width, no ETA, percentage shown) on first call; subsequent calls update progress/total
- `accumulate_tokens(event)` — sums `prompt_tokens + completion_tokens` from `InferEndEvent`
- `done()` — removes the progress bar if present, mounts a summary `Static` with format `⎿ Done ({tokens} tokens · {elapsed})`; elapsed uses `_fmt_duration_verbose` (ms / s / m s)

## Context Bar
`Static #context-bar`; sits below `#input-area` inside `#footer`; always visible.

Displays `"X.X% context"` — an approximation of how much of the core model's context window the current session consumes.

**Computation**: `sum(len(content) for all session.messages) // 4` (chars ÷ 4 ≈ tokens); divided by `_context_limit(model)`.

**Context limit resolution** (`_context_limit`): checks `GEKAI_CORE_MODEL_CONTEXT_LIMIT` env var first (exact override for exotic models); falls back to prefix matching against a built-in table (`claude` → 200k, `gemini-1.5`/`gemini-2` → 1M, `gpt-4o`/`gpt-4-turbo` → 128k, `gpt-4` → 8k, `gpt-3.5` → 16k); defaults to 128k if no match.

**Update triggers**: `_init_session` (on startup or session restore), `_clear_session` (after `/new`), end of every `_stream` turn.

**Session resume**: messages restored from disk are in `session.messages` before the first estimate — the bar correctly reflects prior session size from turn 1.

No persistence — recomputed from message content each time.

## Status / Spinner
`_tick_status(color)` cycles `| / - \` frames, derives text from `_status_verb` + elapsed time, calls `_set_status`
`_set_status(text, color)`: `status.update(text)`, `status.styles.color = color`, sets both `#status-line` and `#status-spacer` to `display: True`
`_clear_status()`: clears text, sets both to `display: False`
`_start_status_animation(text, color)` / `_stop_status_animation()`: asyncio task that calls `_tick_status` every 150 ms; used by `/workspace:rebuild` which has multi-phase async progress

## CommandPalette
`Static` subclass; `display: none` by default
`_max_name_len` computed once at init from all registered commands; descriptions align to this column across filtered subsets
`filter(typed)`: prefix-match on `cmd.name.lower()`; resets `_selected = 0`; calls `_refresh_display()`
`_refresh_display()`: renders all matched items as a single Rich markup string; selected item `[bold cyan]❯ …[/bold cyan]`, others `[dim]  …[/dim]`; sets `self.display = True/False`

## PermissionScreen
`ModalScreen[str|None]`; shown on first run (no `.gekai/settings.local.json`)

Three choices rendered as `Static` widgets pre-populated in `compose()` to avoid height collapse
`reactive(selected, init=False)` + `is_mounted` guard in `watch_selected` to prevent `NoMatches` during init
Dismisses with choice key (`"read_only"` / `"full"` / `"deny"`) or `None` on Escape → app exits

## Key Bindings
| Key     | Action                                                          |
|---------|-----------------------------------------------------------------|
| `esc`   | `action_cancel_stream`: hide palette+clear input, or cancel worker+clear status |
| `ctrl+c`| `quit`                                                          |

## Accent Colors
Defined in `agent/ui.py`; all validated against `textual.color.Color.parse()`:
yellow, ansi_bright_yellow, gold, orange, darkorange, red, ansi_bright_red, indianred, salmon

Use `styles.color = color` (Textual CSS), not Rich inline style — CSS inherits override Rich markup color