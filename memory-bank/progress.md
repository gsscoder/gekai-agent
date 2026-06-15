# Project Progress

## Status
Incomplete Alpha version

## Current Focus
Implementing missing harnesses and improving existing ones

## Recent Changes
- Pipeline matured: per-subagent tool allowlists drive a centrally generated `<tools>` prompt, subagent system prompts have a split identity, `MainAgent` was renamed to `Harness` with the source tree reorganized into `pipeline`/`tools`/`subagents`, and the TUI gained a runtime permission gate, history panel, and file-reference panel
- Router gains a conservative `TRIVIAL` route that lets the TUI skip `FileLocator`, `PromptRewriter`, and the Harness thinking budget for non-codebase turns like greetings
- Added a two-tier workspace ignore engine (hidden vs forbidden) enforced across the scanner, shell, and file tools
- Added an interactive allow_hidden grant flow with concurrency-safe prompting, wired end-to-end through the harness, explorer, agent, and TUI