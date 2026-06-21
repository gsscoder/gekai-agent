# AI guidance

## Project Brief
A precision-scoped AI coding agent designed for surgical interventions on codebases.
Checkpoint-oriented: built around frequent human validation, not long autonomous runs.

## Core Technologies
Python 3.11+, asyncio, PyPI

## Architecture
Textual TUI on top of a per-turn pipeline:
- `Router` — single LLM call, classifies the turn as `main` / `trivial` / `rejected` / `<subagent>`
- `FileLocator` — agentic file discovery for non-trivial turns, biased by a SQLite keyword cache (hints only, never a bypass)
- Blast-radius gate — for subagent turns only, blocks if too many code areas are touched at once
- `PromptRewriter` — weaves located files into the prompt before the model sees it
- `Harness` — the tool-calling agent loop; runs directly or spawned as a specialist subagent
- Permissions (read/write/exec) gate filesystem/shell tools, persisted to `.gekai/settings.local.json`
- Persistence: `session.jsonl` (visible, resumable chat history) + always-on `events-*.jsonl` telemetry; `--debug` adds `debug.jsonl` (internal-only)

## Memory
The development documents are organized in the `memory-bank` dir:
- `progress.md`: progress log
- `detailed-spec`: primarily focuses on specific feature implementation details
- `gen-directives`: content and code generation guidelines

## Output
- Code: match the architectural and stylistic conventions of the existing codebase
- Quality: production-grade — every line will be reviewed
- Markdown: compact, no linting compliance, formatting identical to this file

## Operational Rules:
- Read a language-specific file in `gen-directives` only when a coding task is requested
- Read files in `detailed-spec` only when required by the current task; scan filenames first and read file contents only if they are relevant to the task
- NEVER update this file
- NEVER modify `*.md` files in `memory-bank` (at any depth in the dir tree) without an explicit request
- NEVER initiate any codebase modifications without an explicit request
- NEVER commit changes to Git history without explicit authorization

## Guardrails
### Coding & Design
- Simplicity Over Abstraction: write the simplest solution that meets the requirements; avoid unnecessary layers, patterns, and bloated APIs unless explicitly justified
- Plan Before Implementing: outline a brief strategy or approach before writing code, especially for non-trivial tasks, to catch wrong directions early
- Scope Changes Precisely: remove dead code and stale comments made obsolete by your changes, but never modify, reformat, or delete code and comments orthogonal to the current task
### Reasoning & Collaboration 
- Surface Uncertainty, Don't Guess Through It: when requirements are ambiguous, contradictory, or incomplete, stop and ask for clarification instead of assuming intent and proceeding silently
- Honesty Over Agreement: push back on questionable requests and defend sound technical choices instead of immediately complying with every suggestion
- Signal Confidence Level: indicate when a solution is a best guess versus a well-established approach, so the user can calibrate their review effort
### Instruction Governance
- Respect User Instructions Strictly: treat directives in instruction files as hard constraints, not soft suggestions to be overridden by default tendencies

## Progress.md
- If `Recent Changes` reaches 6, merge them into 1 summary item
- Annotated items must be conceptual, expressed in 1 sentence
- NEVER update any other paragraph