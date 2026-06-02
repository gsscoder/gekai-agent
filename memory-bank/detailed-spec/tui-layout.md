# TUI Layout
Textual-based full-screen terminal UI; fixed footer with command palette and input, scrollable conversation body

## Package Layout
`agent/tui/` groups the full-screen shell: `app.py` owns app lifecycle, stream worker, subagent rendering (`SubAgentRenderer`), command dispatch, and first-run permission flow (via `ChoiceBar`); `widgets.py` holds all reusable display components; `palette.py` holds `CommandPalette`

## Screen Structure
```
┌─ ScrollableContainer #conversation ──────────────────┐  1fr, no scrollbar
│  MessageWidget (BANNER)                              │
│  MessageWidget (SYSTEM)  version / dir / branch      │
│  MessageWidget (USER)    ❯ input text                │
│  MessageWidget (ASSISTANT)  ● markdown response      │
│  MessageWidget (OPERATION)  * PastVerb for Xs        │
│  ...                                                 │
├─ Container #footer ───────────────────────────────── │  dock: bottom, height: auto, padding-bottom: 1
│  ChoiceBar #choice-bar  question + option list       │  hidden by default; shown for permission/inline questions
│  Static #status-line   spinner + verb + elapsed      │  hidden when idle
│  Static #status-spacer blank line below status       │  hidden when idle
│  CommandPalette #command-palette                     │  hidden when input ≠ /…
│  Static #hint-area     transient hint text           │  hidden by default; right-aligned, grey
│  Container #scroll-hint-wrap                         │  hidden; shown when scrolled up and not streaming
│  FilePanel #file-panel                               │  hidden by default; shown on @ trigger
│  HistoryPanel #history-panel                         │  hidden by default; shown on Ctrl+R
│  Static #copy-notice   clipboard feedback            │  hidden by default; right-aligned, grey
│  Container #input-area                               │  height: 3, layered, border-top + border-bottom solid #3a3a3a
│    Input  #prompt                                    │  layer: input, padding-left: 2
│    Static #prompt-marker   ❯                         │  layer: marker, absolute, offset 0 0
│  Static #context-bar       [model] | dir [⎇ branch] | X.X% context  │  height: 1, left-aligned, grey, padding-left: 2
│  Static #version-bar       version label             │  height: 1, right-aligned, padding-right: 2
└──────────────────────────────────────────────────────┘
```
All widgets use `background: ansi_default` / `color: ansi_default` for terminal transparency

## MessageWidget
`Widget` subclass in `widgets.py`; `kind` selects rendering branch; CSS class set to `kind.value`

| Kind             | Rendering                                                                                        |
|------------------|--------------------------------------------------------------------------------------------------|
| `BANNER`         | `[cyan]{pyfiglet ASCII}[/cyan]`                                                                 |
| `SYSTEM`         | `[dim]{text}[/dim]`                                                                             |
| `USER`           | `❯ [bold]{text}[/bold]` on dark background (`#3a3a3a`)                                         |
| `ASSISTANT`      | `[cyan]●[/cyan]` (width 2) + `Markdown` (1fr) horizontal; if `color` set: `Static` body instead |
| `OPERATION`      | `[{color}]{text}[/{color}]`; default color `grey50`                                            |
| `HEADER`         | `.header-dot` Static (width 2) + `.header-text` Static (1fr), horizontal layout                 |
| `INTERRUPTED`    | `[red]●[/red]` + body text; if no text: grey dot + "Interrupted / How should Gekai proceed?" |
| `ERROR`          | `[red]●[/red]` + "Error / ⎿ {text}"                                                           |
| `REJECTED`       | `[red]●[/red] [white]Rejected[/white] — [#666666]{text}[/#666666]`                            |
| `COMMAND_RESULT` | `[#666666]⎿[/#666666] [#ffd700]{text}[/#ffd700]`; if empty: `[dim]⎿ (no output)[/dim]`       |

`.update(content)` and `.append_text(chunk)` refresh the inner child widget in place

## Startup Flow
1. `main.py` resolves permissions, creates `GekaiApp`, calls `.run()`
2. `on_mount`: disables terminal mouse tracking; spawns clipboard poll task; runs `_init_session()` as exclusive worker
3. `_init_session()`: mounts BANNER widget; loads `workspace.json` unconditionally if it exists (no live code writes this file — WsExplorer scan is dead code); starts session via `agent.start_session()`; loads `scope_gate` from settings; sets context limit from settings or model-prefix lookup; updates `#context-bar`; replays restored messages as USER/ASSISTANT widgets; if `needs_permissions` → calls `_ask_choice()` with `PERMISSION_CHOICES`, resolves + saves result
— [dead code] no workspace scan or enrichment runs at startup; `_run_ws_explorer()` and `_maybe_rescan_workspace()` are unreachable from this path

## Input Handling
`on_input_submitted`:
- `_pending_choice` future set → read `choice_bar.selected_key`, hide `ChoiceBar`, clear input, resolve future, return
- `_session` None or worker running → refocus prompt, return
- palette visible → take `palette.selected_command`, hide palette; if command in `_COMMANDS_WITH_ARGS` → insert `/{cmd} ` into input and return (no dispatch); else override `stripped`
- `stripped` starts with `/` → mount USER widget, dispatch via `CommandRegistry`; if `result.scope_gate` not None → update `session.scope_gate`; if `result.clear_session` → `_clear_session()`; if `result.exit_app` → mount farewell ASSISTANT widget, sleep, exit
- else → append to prompt history, mount USER widget, run `_stream(_resolve_at_refs(stripped))` as exclusive worker

`on_input_changed`:
- if `history_panel` visible → return (suppress palette + file panel while history open)
- value starts with `/` → `palette.filter(value[1:])`; else → `palette.hide()`
- find last `@` via `rfind("@")`; if found and no space after `@` → load `list_files()` lazily, show/filter `FilePanel`; else → hide panel, reset `_file_at_pos = -1`

`on_key`:
- `ChoiceBar` visible + left/right → `choice_bar.move_left()` / `choice_bar.move_right()`, `event.stop()`
- prompt not focused + printable → `prompt.focus()`, `prompt.insert_text_at_cursor()`, `event.stop()`

`action_navigate_up/down` (bound to Up/Down keys, priority):
- `FilePanel` visible → `file_panel.move_up/down()`
- `HistoryPanel` visible → `panel.move_up/down()`, set prompt value to `panel.selected_text`
- `CommandPalette` visible → `palette.move_up/down()`
- `ChoiceBar` visible → `choice_bar.move_left/right()`
- else → `conversation.scroll_up/down()`

## Streaming Worker (`_stream`)
Runs as exclusive Textual worker.
- picks random operative verb pair and accent color for the turn
- normalizes input via `agent.normalize`, classifies via `agent.classify`
- calls `agent.check_gate(segments)`; if rejected and `session.scope_gate` → mount `REJECTED` widget and return
- if `--debug`: mount `OPERATION` widget with `[classifier: ...]` labels in `#BA55D3`; mount second OPERATION with normalization result
- language detection: if `src_lang` differs from `_current_lang` → appends `<lang>` system message to session
- iterates `agent.process_stream`; `str` items → `answer_chunks`
- `SubAgentEvent` items dispatched to a `SubAgentRenderer` instance:
  - `SubAgentStartEvent` → construct renderer, call `renderer.start()`
  - `LogEvent` → `renderer.log(item.message, tool_name=item.tool_name)`; increments `query_tool_count` if renderer name is `"query"`
  - `InferEndEvent` → `renderer.accumulate_tokens(event)`
  - `ThinkingTokenEvent` → `renderer.thinking_chunk(item.text)`
  - `StatusUpdateEvent` → `renderer.status_update(event)`
  - `DoneEvent` → `renderer.done(item.thinking_chars)`
- updates `#context-bar` to `_fmt_context_pct(tokens, limit)` after stream
- on complete: mount ASSISTANT widget + OPERATION widget `"* {verb[1]} for {duration}"` (appends `" ({n} tools)"` if `query_tool_count > 0`), scroll end
- `finally`: `_stop_status_animation()`, `ws_renderer.stop_spinner()` if set, handle `_worker_cancelled` (mount INTERRUPTED widget), `_worker = None`, `_focus_prompt()`

## SubAgentRenderer
Helper class in `app.py`; one instance per subagent block within a turn.
- `start(name, description, color)` — mounts `assistant-spacer` Static then a `HEADER` MessageWidget with `"[bold #666666]Thinking...[/bold #666666]"` header text; starts braille-frame `_animate_dot()` asyncio task on `.header-dot`
- `thinking_chunk(text)` — buffers thinking tokens; updates `.header-text` to `"[bold #666666]Thinking({last_sentence})[/bold #666666]"`
- `log(message, tool_name)` — skips messages ending with `"..."`; in non-debug mode deduplicates consecutive calls from the same `tool_name` (updates existing widget to `"{kind} ({n} calls)"`); first item prefix `"  ⎿"`, subsequent `"   "`
- `status_update(event)` — lazily mounts a `ProgressBar` (40% width, no ETA, percentage shown) on first call; subsequent calls update progress/total
- `accumulate_tokens(event)` — sums `prompt_tokens + completion_tokens` from `InferEndEvent`; increments `_infer_count`
- `stop_spinner()` — cancels the `_animate_dot` task if running
- `done(thinking_chars)` — cancels spinner task; removes all log widgets; removes progress bar if present; updates header dot to `"[#666666]●[/#666666]"` and header text to `"[#666666]Thought ({tokens} tokens · {elapsed} · {n} calls)[/#666666]"`; elapsed uses `_fmt_duration_verbose` (ms / s / m s)

## Context Bar
`Static #context-bar`; sits below `#input-area` inside `#footer`; always visible.

Startup/clear displays full status bar: `_fmt_status_bar(model, working_dir.name, branch, tokens, limit)` → `"[dim][{model}][/dim] | 📁 {dir} [⎇ {branch}] | [dim]{X.X}% context[/dim]"`
After each `_stream` turn: updates to compact `_fmt_context_pct(tokens, limit)` → `"X.X% context"`

**Token estimate**: `sum(len(str(m.get("content") or "")) for m in session.messages) // 4`; includes persistent system messages (`[preference]`, `[artifact]`, `<lang>`)

**Context limit resolution** (`_context_limit`): checks `load_context_limit(working_dir)` (project then global settings) first; falls back to prefix matching against built-in table:
`gpt-4o` / `gpt-4-turbo` → 128k, `gpt-4` → 8192, `gpt-3.5` → 16385, `claude` → 200k, `gemini-1.5` / `gemini-2` → 1M, `deepseek-chat` → 128k; default 128k

**Update triggers**: `_init_session` (startup or session restore), `_clear_session` (after `/clear`), end of every `_stream` turn

No persistence — recomputed from message content each time

## Status / Spinner
`_tick_status(color)` cycles `["·", "•", "●", "•"]` frames; skipped if `_status_paused`; derives text via `_fmt_status(verb, elapsed)` → `"{Verb}... ({elapsed}s)"`, calls `_set_status`
`_set_status(text, color)`: `status.update(text)`, `status.styles.color = color`, sets both `#status-line` and `#status-spacer` to `display: True`
`_clear_status()`: clears text, sets both to `display: False`
`_start_status_animation(verb, color)` / `_stop_status_animation()`: asyncio task via `_animate_status()` that calls `_tick_status` every 150 ms
`_status_paused`: set `True` during `_ask_choice()` calls inside `_permission_callback`; elapsed time extended by pause duration so spinner clock is not inflated

## CommandPalette
`Static` subclass in `palette.py`; `display: none` by default; padding: `0 2`
`_max_name_len` computed once at init from all registered commands; `/{name}` padded to `_max_name_len + 3` chars; descriptions align to this column across filtered subsets
`filter(typed)`: prefix-match on `cmd.name.lower()`; resets `_selected = 0`; calls `_refresh_display()`
`_refresh_display()`: renders all matched items as a single Rich markup string; selected item `[bold cyan]❯ {padded}{desc}[/bold cyan]`, others `[dim]  {padded}{desc}[/dim]`; `display = True` if items, `False` if empty
`on_click(event)`: `idx = event.y`; dispatches `app.action_select_command(name)` for clicked row
Commands in `_COMMANDS_WITH_ARGS` (e.g. `config:gate`): when selected from palette, insert `/{cmd} ` into input and return without dispatching — user types arguments then submits

## ChoiceBar
`Static` subclass in `widgets.py`; `display: none` by default; replaces the former PermissionScreen modal
`show(question, options, default_index)`: sets `_question`, `_options`, `_selected`, calls `_refresh_display()`, sets `display = True`
`_refresh_display()`: renders question + options; selected option `[bold cyan]❯ {label}[/bold cyan]`, others plain
`move_left()` / `move_right()`: cycle `_selected` wrapping; bound to `left`/`right` arrow keys when ChoiceBar is visible
`selected_key` property: returns `_options[_selected][0]`
Used for: first-run permission selection, permission grant prompts during tool calls (`_permission_callback`)

## HistoryPanel
`Widget` subclass in `widgets.py`; `display: none` by default; height: 5; border-top solid `#3a3a3a`; mounted in `#footer` above `#input-area`
`_MAX_ENTRIES = 5`; `_MAX_TEXT_LEN = 60`; entries stored in `~/.gekai/workspaces/<dir>/history.jsonl` via `PromptHistory`

`action_toggle_history` (bound to `Ctrl+R`, priority): loads `_history.load()`, reverses list (newest first), calls `panel.show(entries, selected_index=0)`; if already visible, hides and refocuses prompt

Row format: `[bold cyan]❯ [dim]{ago:>6}[/dim]  {escaped}[/bold cyan]` for selected; `[dim]  [dim]{ago:>6}[/dim]  {escaped}[/dim]` for others; ago via `_fmt_ago` (`{n}s ago` / `{n}m ago` / `{n}h ago` / `{n}d ago`)
Scrollable window: `_window_start` centres the view around `_selected` when entries exceed 5

Up/Down (via `action_navigate_up/down`): when panel visible → `panel.move_up/down()`, set `#prompt` value to `panel.selected_text`
Enter (`action_confirm_or_submit`): sets `#prompt` value + cursor to end (`action_end()`), hides panel, refocuses prompt
Click (`on_history_panel_row_clicked`): selects clicked row, sets prompt value, hides panel, refocuses prompt
ESC (`action_cancel_stream`): hides panel (checked after `FilePanel`, before palette check)

## FilePanel
`Widget` subclass in `widgets.py`; `display: none` by default; height: 5; border-top solid `#3a3a3a`; mounted in `#footer` above `HistoryPanel`
`_MAX_ENTRIES = 5`; `_MAX_PATH_LEN = 60`; `_file_at_pos: int` on `GekaiApp` tracks `@` position in input for replacement

Trigger: `on_input_changed` finds last `@` via `value.rfind("@")`; extracts `query = value[at_pos+1:]`
- if `query` has no space: loads `list_files(working_dir)` lazily into `_file_paths` (flat sorted relative paths), calls `file_panel.show(paths, query)` or `file_panel.filter(query)` if already visible
- if space appears in `query` or no `@` found: hides panel, resets `_file_at_pos = -1`

Filter: contains-search `query.lower() in path.lower()` over full path list; resets `_selected = 0` on each filter

Row format: `[bold cyan]+ {escaped}[/bold cyan]` for selected; `[dim]  {escaped}[/dim]` for others; long paths truncated as `"…" + path[-(MAX_PATH_LEN - 1):]` (preserves last 59 chars)

Up/Down (via `action_navigate_up/down`, checked before HistoryPanel): when panel visible → `file_panel.move_up/down()`
Enter (`action_confirm_or_submit`): replaces input with `value[:at_pos] + "@{selected_path} "`, sets cursor to end, hides panel — does NOT submit
Click (`on_file_panel_row_clicked`): same replacement logic as Enter
ESC (`action_cancel_stream`): hides panel, resets `_file_at_pos = -1` (checked first, before HistoryPanel)

Post-processing: `_resolve_at_refs(text)` (called in `on_input_submitted` before passing to `_stream`) replaces every `@(\S+)` token with `` `\1` ``; user sees `@path` in conversation widget, model receives `` `path` ``

## Key Bindings
| Key          | Action                                                                                         |
|--------------|-----------------------------------------------------------------------------------------------|
| `esc`        | `action_cancel_stream`: hide FilePanel → hide HistoryPanel → dismiss pending ChoiceBar → hide palette+clear input → cancel worker+clear status → double-ESC to clear input |
| `ctrl+r`     | `action_toggle_history`: toggle HistoryPanel                                                  |
| `ctrl+c`     | `quit`                                                                                        |
| `ctrl+up`    | `action_scroll_to_top`: scroll conversation to home                                          |
| `ctrl+down`  | `action_scroll_to_end`: scroll conversation to end                                           |
| `pageup`     | `action_scroll_page_up`                                                                       |
| `pagedown`   | `action_scroll_page_down`                                                                     |
| `up`         | `action_navigate_up`: FilePanel → HistoryPanel → CommandPalette → ChoiceBar → scroll up      |
| `down`       | `action_navigate_down`: FilePanel → HistoryPanel → CommandPalette → ChoiceBar → scroll down  |
| `enter`      | `action_confirm_or_submit`: confirm FilePanel/HistoryPanel selection, or submit input         |
| `left/right` | when ChoiceBar visible: `choice_bar.move_left/right()`                                       |

## Accent Colors
Defined in `agent/ui.py`; used for turn color, status bar color, and debug labels:
yellow, ansi_bright_yellow, gold, orange, darkorange, goldenrod, ansi_bright_red, coral, salmon

Use `styles.color = color` (Textual CSS), not Rich inline style — CSS inherits override Rich markup color