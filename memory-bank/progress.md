# Project Progress

## Status
Active development

## Current Focus
Query handler intelligence: deterministic tool activation and workspace-aware routing

## Recent Changes
- Full Textual TUI with intent classifier, workspace-aware routing, WS-explorer Y/N activation, slash command palette, permission modal, token tracking, and color-randomized spinner
- History panel (Ctrl+R): navigable prompt history overlay in TUI footer, persisted as JSONL per workspace
- File reference panel (@): contains-search over repo files; selection inserts `@path` in input, post-processed to backtick notation before model send