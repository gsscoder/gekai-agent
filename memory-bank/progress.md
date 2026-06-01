# Project Progress

## Status
Active development

## Current Focus
Upgrade intent classifier and introduce strategist component

## Recent Changes
- Runtime permission gate (`PermissionGate`) wired into tool execution loop; tools declare required permission and are blocked or escalated via async TUI callback; `+plan` tagging and WsExplorer Y/N activation removed
- Full Textual TUI with intent classifier, workspace-aware routing, slash command palette, permission dialog, token tracking, and color-randomized spinner
- History panel (Ctrl+R): navigable prompt history overlay in TUI footer, persisted as JSONL per workspace
- File reference panel (@): contains-search over repo files; selection inserts `@path` in input, post-processed to backtick notation before model send