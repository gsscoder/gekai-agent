# TUI Layout
Textual-based full-screen terminal UI; fixed footer with command palette and input, scrollable conversation body

## Package Layout
```
agent/tui/
  app.py         — GekaiApp(App), _PlaceholderApp, startup/stream/command orchestration
  widgets.py     — MessageWidget, MessageKind enum
  permissions.py — PermissionScreen(ModalScreen): first-run workspace access dialog
  palette.py     — CommandPalette(Static): slash command filter popup
  completer.py   — SlashCommandSuggester (unused; kept as reference)
```

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
│  Static #status-line   spinner + verb + tokens       │  hidden when idle
│  Static #status-spacer blank line below status       │  hidden when idle
│  CommandPalette #command-palette                     │  hidden when input ≠ /…
│  Container #input-area                               │  height: 3, layered
│    Static #prompt-marker   ❯                         │  layer: marker, absolute
│    Input  #prompt                                    │  layer: input, padding-left: 2
└──────────────────────────────────────────────────────┘  padding-bottom: 2
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
4. `_init_session()`: mounts BANNER + SYSTEM widgets; checks workspace cache (15 min fresh / 30 min resume); if stale runs `scan_workspace` in thread with live SYSTEM widget updates via `call_from_thread`; starts session; replays restored messages as USER/ASSISTANT widgets

## Input Handling
`on_input_submitted`:
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
- calls `agent.classify(user_input)` then iterates `agent.process_stream(session, input, segments)`
- string chunk → append to `answer_chunks`, `_tick_status(verb_status, color)`
- `EnrichmentEvent(kind="start"|"done")` → status text update
- `UsageInfo` → update `completion_tokens` in status
- on complete: mount ASSISTANT widget + OPERATION widget, scroll end
- `finally`: `_clear_status()`, `_worker = None`, `_focus_prompt()`

## Status / Spinner
`_tick_status(text, color)` cycles `| / - \` frames and calls `_set_status`
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