# Project Progress

## Status
Active development

## Current Focus
Query handler intelligence: deterministic tool activation and workspace-aware routing

## Recent Changes
- Reworked QueryHandler tool activation: balanced `_TOOL_INSTRUCTION` frames `<workspace>` metadata as authoritative for high-level questions while mandating tools for code-level queries; workspace context preamble reinforces this distinction
- WS-explorer activation confirmation: transient Y/N via `on_input_submitted`, animation paused before question, no chat trace, ws-explorer completes before LLM call
- Hardened TUI, routing, and session wiring: intent classifier, `update_workspace_context`, VS Code launch configs, token usage tracking, user input highlight
- Full Textual TUI with slash command palette, color-randomized spinner, and permission modal; removed prompt_toolkit dependency