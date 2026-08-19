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

## Maturity
- The system is currently under development
- Backward compatibility is not required when changing existing features

## Memory
The development documents are organized in the `memory-bank` dir:
- `detailed-spec`: primarily focuses on specific feature implementation details
- `gen-directives`: language-specific code generation directives (see Operational Rules)

## Output
- Code: match the architectural and stylistic conventions of the existing codebase
- Language: use English for all generated artifacts and symbols by default. Content in another language is allowed only in user-facing strings, messages, and labels when the application has a single localization
- Quality: production-grade — every line will be reviewed
- Markdown: compact, no linting compliance, formatting identical to this file

## Operational Rules:
- MUST read the matching language-specific file in `gen-directives` before writing code, if one exists for that language
- Read files in `detailed-spec` only when required by the current task; scan filenames first and read file contents only if they are relevant to the task
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