# Project Progress

## Status
Active development

## Current Focus
Upgrade intent classifier and introduce strategist component

## Recent Changes
- Per-subagent tool allowlists now actively narrow the registered toolset, and the `<tools>`
  activation prompt is generated centrally and deterministically — `render_tool_instruction`
  in `agent/persona.py` builds it from a non-LLM fragment table (`_TOOL_GUIDANCE`) keyed on
  tool-name groups in the new `agent/tools/catalog.py` (`READ_TOOLS`/`EDIT_TOOLS`/`FS_TOOLS`/
  `SHELL_TOOLS`/`ALL_TOOLS`, mirrored against `make_tools()` by a drift-guard test). The block
  reflects the *effective* set — allowlist intersected with the live permission grant — computed
  in one place, `Harness._build_agent`, for both direct and spawn modes; `render_tool_instruction(ALL_TOOLS)`
  reproduces the old static instruction string verbatim (behavior-preserving)
- Subagent system-prompt identity split to stop stacking two competing "you are" claims:
  `agent/persona.py` now exposes `_IDENTITY_MAIN` ("you are Gekai…", main agent only),
  `_IDENTITY_SUB` ("you are part of Gekai…", tool-capability-neutral, subagent only), and
  `_SHARED_BODY` (behavior/file_handling/response_style/output_format, reused verbatim by both;
  `SYSTEM_PROMPT = _IDENTITY_MAIN + _SHARED_BODY`, byte-identical to the pre-split constant).
  `Subagent.build_system_base()` assembles `_IDENTITY_SUB` + a plain-prose role line
  (`mandate`, reframed "you act as a …" — the `<core_mandate>` wrapper tag is gone) + `_SHARED_BODY`
  + `<directives>`; the `<tools>` block is appended afterward by the harness once the effective
  tool set is known
- `MainAgent` renamed/relocated to `Harness` (`handlers/main_agent.py` → `agent/harness/core.py`);
  source tree reorganized into `pipeline/` (router, blast-radius gate, prompt rewriter), `tools/`
  (catalog + file/shell tools), `subagents/`
- Runtime permission gate (`PermissionGate`) wired into tool execution loop; tools declare required permission and are blocked or escalated via async TUI callback; `+plan` tagging and WsManager Y/N activation removed
- Full Textual TUI with intent classifier, workspace-aware routing, slash command palette, permission dialog, token tracking, and color-randomized spinner
- History panel (Ctrl+R): navigable prompt history overlay in TUI footer, persisted as JSONL per workspace
- File reference panel (@): contains-search over repo files; selection inserts `@path` in input, post-processed to backtick notation before model send