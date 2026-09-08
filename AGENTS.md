# Project Overview & AI Instructions

## Project Brief
A precision-scoped AI coding agent designed for surgical interventions on codebases.
Checkpoint-oriented: built around frequent human validation, not long autonomous runs.

## Core Technologies
Python 3.11+, asyncio, PyPI

## Layers
- `agent/pipeline` — Router, FileLocator, blast-radius gate, PromptRewriter (the per-turn pipeline)
- `agent/harness` and `agent/subagents` — the tool-calling loop and its specialist delegation
- `agent/llm` — vendored provider adapter (retries, events, model caps)
- `agent/tools` — tool catalog and implementations (files, shell)
- `agent/workspace` — scanning, ignore rules, manifest parsing, symbol indexing
- `agent/tui` — Textual UI
- `agent/commands` — slash-command handling
- Cross-cutting root modules (`permissions.py`, `session.py`, `events.py`, `logging.py`, `diff.py`, `shell.py`, `compact.py`) — support all layers above

## Architecture
Textual TUI on top of a per-turn pipeline:
- `Router` — single LLM call, classifies the turn as `main` / `trivial` / `rejected` / `<subagent>`
- `FileLocator` — agentic file discovery for non-trivial turns, biased by a SQLite keyword cache (hints only, never a bypass)
- Blast-radius gate — for subagent turns only, blocks if too many code areas are touched at once
- `PromptRewriter` — weaves located files into the prompt before the model sees it
- `Harness` — the tool-calling agent loop; runs directly or spawned as a specialist subagent
- Permissions (read/write/exec) gate filesystem/shell tools, persisted to `.gekai/settings.local.json`
- Persistence: `session.jsonl` (visible, resumable chat history) + always-on `events-*.jsonl` telemetry; `--debug` adds `debug.jsonl` (internal-only)

### Session Storage
Sessions live at `~/.gekai/workspaces/{normalized-repo-path}/{session-id}.jsonl`, one JSON object per line, each with a `ts` and a `kind` (`turn`, `command`, or `event`; entries without `kind` are legacy `turn`). `{session-id}.debug.jsonl` holds internal-only plumbing (system prompts, tool calls) and exists only with `--debug`. Separately, `~/.gekai/logs/events-{YYYYMMDD}-{HHMMSS}-{run_id}.jsonl` holds always-on per-run telemetry, linked back to a session turn via the shared `turn` field. See `memory-bank/detailed-spec/logging.md` and `docs/architecture.md → Session Persistence` for full record shapes

## Maturity
- The system is currently under development
- Backward compatibility is not required when changing existing features

## Memory
The development documents are in the `memory-bank` dir — they primarily focus on specific feature implementation details

## Output
- Code: match the architectural and stylistic conventions of the existing codebase
- Language: use English for all generated artifacts and symbols by default. Content in another language is allowed only in user-facing strings, messages, and labels when the application has a single localization
- Quality: production-grade — every line will be reviewed
- Markdown: compact, no linting compliance, formatting identical to this file

# Python Code Standards
- Generate code for Python 3.12+, using the newest 3.x syntax/features you have reliable knowledge of, favoring clarity and expressive constructs over legacy patterns
- Prefer explicit named parameters; avoid `**kwargs` except for true pass-through scenarios (e.g., decorators/adapters). If used, document all consumed keys
- Never use mutable defaults (`list`, `dict`, `set`); use `None` and initialize inside the function
- Avoid mutating input arguments unless explicitly documented or clearly indicated by the function name; otherwise return a new object
- Signal errors with specific exceptions; do not use sentinel return values (`None`, `False`, `-1`) unless explicitly required and properly typed
- Require full type annotations on all functions; only use `Optional[T]` when `None` has explicit semantic meaning, not as a generic default

### Python Code Standards
- Generate code for Python 3.12+, using the newest 3.x syntax/features you have reliable knowledge of, favoring clarity and expressive constructs over legacy patterns
- Prefer explicit named parameters; avoid `**kwargs` except for true pass-through scenarios (e.g., decorators/adapters). If used, document all consumed keys
- Never use mutable defaults (`list`, `dict`, `set`); use `None` and initialize inside the function
- Avoid mutating input arguments unless explicitly documented or clearly indicated by the function name; otherwise return a new object
- Signal errors with specific exceptions; do not use sentinel return values (`None`, `False`, `-1`) unless explicitly required and properly typed
- Require full type annotations on all functions; only use `Optional[T]` when `None` has explicit semantic meaning, not as a generic default

## Operational Rules
- If you're Claude Code, you may have specialized subagents available for many use cases — check `.claude/agents/` and delegate to a matching one when appropriate; otherwise, handle the work directly
- Read files in `memory-bank` only when required by the current task; scan filenames first and read file contents only if they are relevant to the task
- Review/audit/report requests end at the report; fixing findings needs its own separate request — authorization never carries across turns
- NEVER update `AGENTS.md` without an explicit request
- NEVER modify `*.md` files in `memory-bank` (at any depth in the dir tree) without an explicit request
- NEVER initiate any codebase modifications without an explicit request
- NEVER commit changes to Git history without explicit authorization

## Guardrails
These are hard constraints, not suggestions, and they bias toward caution over speed — apply them proportionately on trivial or throwaway tasks. Where a request conflicts with a guardrail, follow the guardrail and say why.

### Before Implementing — Reason First
- Surface Uncertainty, Don't Guess Through It: When requirements are ambiguous, contradictory, or incomplete, stop and ask instead of assuming intent and proceeding silently. State assumptions explicitly; if multiple readings are viable, present them rather than picking one silently. Resolve intent up front — this is what makes autonomous execution safe afterward
- Plan Before Implementing: For non-trivial tasks, outline a brief approach before writing code so wrong directions surface early. For multi-step work, list the steps with a verification check for each

### Design & Scope — Code Minimally
- Simplicity Over Abstraction: Write the simplest solution that meets the requirements. Avoid speculative features, abstractions for single-use code, unrequested configurability, and error handling for impossible cases. If a construction could be materially shorter without losing correctness, rewrite it — ask whether a senior engineer would call it overcomplicated
- Surgical Scope: Every changed line should trace directly to the current task. Match the surrounding style even where you'd choose differently. Remove imports, variables, and comments that *your* changes made obsolete, but never modify, reformat, or delete code or comments orthogonal to the task. If you notice unrelated dead code, mention it — don't delete it

### Execution — Verify Against Goals
- Drive Toward Success Criteria: Turn the task into checkable goals and work until they're met — e.g. "add validation" → write tests for invalid inputs, then make them pass; "fix the bug" → write a failing test that reproduces it, then make it pass. When the goal is well-defined, loop and self-verify independently rather than pausing for confirmation the criteria already answer. (This is the counterpart to *Surface Uncertainty*: clarify the goal before starting; do not re-open a settled goal mid-execution)

### Collaboration — Communicate Honestly
- Honesty Over Agreement: Push back on questionable requests and defend sound technical choices instead of complying by default; avoid reflexive agreement
- Signal Confidence Level: Indicate when a solution is a best guess versus a well-established approach, so review effort can be calibrated. (Complements `Surface Uncertainty`: if you couldn't proceed at all, you ask; if you proceeded on a judgment call, you flag it)