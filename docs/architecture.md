# Architecture

## Overview

Every turn passes through three layers: **classification → profile selection → handler dispatch**.

```
user input
    │
    ▼
IntentClassifier          [core model]    — decompose into Segments; emit namespace label; reject non-English
    │
    ├── chat / action/generic ─────────────────────────────────────────┐
    │                                                                  │
    └── action/coding | action/management                              │
              │                                                        │
              ▼                                                        │
    ProfileSelector       [support model] — pick specialist profile    │
              │                             within namespace;          │
              │                             fallback on none/parse-fail│
              │                                                        │
              └───────────────────────────────────────────────────────►┤
                                                                       ▼
                                                             Handler(s) [core model]
                                                               chat / action (+ profile directives)
```

---

## Session Message Structure

`Session.messages` is the live context passed to the core model each turn.

```
index  role      content
─────────────────────────────────────────────────────────
  0    system    SYSTEM_PROMPT
  1    system    <workspace> block  (TOON-encoded, injected fresh on startup)
  …    user      prior turns
  …    assistant prior turns
  N    user      current input      (appended before dispatch)
  N+1  assistant response           (appended after all segments complete)
```

**System messages** (`[0]`, `[1]`) are excluded from persistence;
they are re-injected fresh on every startup or resume.

---

## SYSTEM_PROMPT Blocks

```
SYSTEM_PROMPT
├── preamble          identity + capability statement
├── meta-rule         "follow user instructions literally"
├── <behavior>        focus, precision, no fabrication
├── <response_style>  terse when explaining; fragments OK; no filler
└── <file_handling>   show/print/display → full verbatim fenced block
```

`ActionHandler._build_system(profile)` composes the system prompt at call time:

```
SYSTEM_PROMPT
\n<directives>\n{profile.directives}   ← only when profile exists and profile.directives non-empty
\n<tools>\n{_TOOL_INSTRUCTION}
```

Tags are non-closing (no `</tag>`), matching `SYSTEM_PROMPT` convention.  
`profile.directives` block is omitted when no profile is selected (e.g. `action/generic`).  
`_TOOL_INSTRUCTION` tells the model when tools are mandatory: if the question requires file
contents, implementation details, logic, or architecture depth, use tools — do not guess or
rely on training knowledge.

---

## Sandbox / Isolation

Tool execution is **not** OS-sandboxed. The current boundary is:

- **Path jail** — `_resolve_in_ws` rejects any path that resolves outside the workspace root (symlinks included via `.resolve()`).
- **No shell** — file ops use `shutil` / `pathlib` directly; no subprocess or shell interpolation surface.
- **Permission gate** — `PermissionGate` enforces `read` / `write` / `exec` per tool call; `exec` permission is not granted by default.

This is proportionate for a single-user local prototype. Full isolation is **deferred** under one explicit assumption: **no exec or network tool exists yet**. The moment either lands, OS-level sandboxing becomes blocking — the path jail is meaningless once arbitrary code runs with user privileges.

---

## Intent Classification

`IntentClassifier.classify(user_input, history=None)` makes **one LLM call** on the **core model** (non-thinking).
Returns `list[Segment]`.

```python
@dataclass
class Segment:
    intent: Intent
    text: str
    namespace: str | None = None  # populated for Intent.ACTION only
```

**History context:** last 6 user/assistant turns prepended before the user message.

```
CLASSIFIER_PROMPT
├── output format     label: text  (one line per segment)
├── <labels>          chat | action/coding | action/management | action/generic
├── <rules>           prefer chat over action; be cagey — prefer action/generic when unsure
└── <examples>        few-shot including management (move parsers/), generic (describe project, README typo)
```

Parser: `kind, _, ns = label.partition("/")`. Unknown or missing namespace → `"generic"`.

Fallback on parse failure: `[Segment(Intent.CHAT, user_input)]`.

### Routing table

| Label                | Intent   | Namespace    | Notes                                              |
|----------------------|----------|--------------|----------------------------------------------------|
| `chat`               | `CHAT`   | —            | knowledge only, no tools                           |
| `action/coding`      | `ACTION` | `coding`     | code edits, refactors, tests → ProfileSelector     |
| `action/management`  | `ACTION` | `management` | file/dir reorganization → ProfileSelector          |
| `action/generic`     | `ACTION` | `generic`    | inspect, docs, prose edits → ProfileSelector skipped; innate behavior only |
| `REJECTED`           | `REJECTED`| —           | non-English input                                  |

---

## Profile Routing

Two-stage design: classifier picks a **namespace** (few, stable labels); `ProfileSelector` picks a **specialist profile** within that namespace (bounded prompt per family). Neither prompt grows with total catalog size.

### AgentProfile

Frozen dataclass defined in `agent/profiles/` package — not user-definable.

```python
@dataclass(frozen=True)
class AgentProfile:
    name: str
    namespace: str
    description: str
    directives: str          # injected after SYSTEM_PROMPT in ActionHandler
    tools: list[str] | None  # allowlist; None = all tools
    permissions: Permissions | None  # AND-restricted overlay
    is_fallback: bool
```

Package layout: `__init__.py` exports `AgentProfile`, `PROFILES`, `profiles_for`, `fallback_for`, `validate_registry`, and `_discover()` auto-discovery.
One file per profile: `code_expert.py`, `code_refactorer.py`, `code_simplifier.py`, `ws_manager.py`.
Namespace-level shared directives live in `_coding.py` (namespace = `"coding"`) — composed into `profile.directives` at import time via `dataclasses.replace`.
Adding a profile = drop one file; zero other changes required.

Registry helpers: `profiles_for(namespace)`, `fallback_for(namespace)`.  
`validate_registry()` called at startup — raises if any namespace in `NAMESPACES` has no fallback.

### ProfileSelector

`ProfileSelector.select(namespace, text, history)` — **one LLM call**, temperature 0, **support model**.

```
SELECTOR_PROMPT
├── menu              one line per candidate:  name — description
├── history context   last 6 user/assistant turns
└── output            profile name only (first whitespace-delimited token, lowercased)
```

**Cagey fallback policy:** `none` / empty / unparseable / unknown token → `fallback_for(namespace)` + warning log.
Bias: correct fallback beats wrong specialist.

### Permission overlay

`profile.permissions` is **ANDed** with `session.permissions` per field — a profile can restrict but never escalate beyond what the session granted.

```python
effective = Permissions(
    read  = session.read  and profile.read,
    write = session.write and profile.write,
    exec  = session.exec  and profile.exec,
)
```

### TUI status indicator

`#input-area` container `border_title` always shows current routing state:

| State                     | Label shown              | Border background color                        |
|---------------------------|--------------------------|------------------------------------------------|
| idle / between turns      | `default`                | `#3a3a3a`                                      |
| after classify            | `action/<namespace>`     | `gold1` (coding) · `cyan` (management) · `#3a3a3a` (others) |
| sub-agent active          | `<profile-name>`         | color from namespace phase — persists          |
| finally (any exit)        | `default`                | `#3a3a3a`                                      |

---

## ActionHandler — Tool Loop

### Transient incarnation / recency window

The action agent is a **transient incarnation** — it does not receive the full session history.
`_recency_turns(session.messages, _RECENCY_N)` extracts the last `_RECENCY_N=2` user/assistant
pairs (skipping system messages, excluding the current trailing user input); the current user
input is then appended explicitly. The agent receives: last 2 turns + current request.

When `--debug` is active, `stream()` calls `append_debug(session, {"content": system})` before
dispatching — the transient system string lands in `.debug.jsonl` alongside the workspace context block.

Uses `llmstitch.Agent` with `OpenAIAdapter`; tools registered from `make_tools(working_dir)`.

```mermaid
sequenceDiagram
    participant TUI
    participant ActionHandler
    participant llmstitch
    participant CoreModel

    TUI->>ActionHandler: stream(session, user_input)
    ActionHandler->>llmstitch: agent.run(prior_messages)
    loop tool-calling
        llmstitch->>CoreModel: messages + tools
        CoreModel-->>llmstitch: ToolUseBlock / TextBlock
        llmstitch-->>ActionHandler: ToolExecutionStarted event
        ActionHandler-->>TUI: LogEvent (e.g. "Edit src/main.py")
        llmstitch->>llmstitch: execute tool, append result
        llmstitch-->>ActionHandler: ToolExecutionCompleted event
        ActionHandler-->>TUI: DiffEvent (edit_file only; old_str vs new_str)
    end
    llmstitch-->>ActionHandler: final history
    ActionHandler-->>TUI: DoneEvent → text response
```

Events flow through `EventBus` → async queue → TUI stream.

---

## WsManager SubAgent [dead code]

> **Note:** `WsManager`, `scan_workspace`, `enrich_workspace`, and all scan activation paths are currently dead code. They are present in source (`ws_manager/` subpackage) but not called from any live path. `_run_ws_manager()`, `_maybe_rescan_workspace()`, and `_rebuild_workspace()` in `tui/app.py` carry `# [dead code]` markers. `/workspace:rebuild` is not registered in the command palette. No live code writes `workspace.json`.

`WsManager` (`ws_manager/subagent.py`) is a `SubAgent` that **enriches workspace metadata**.
It streams `SubAgentEvent` instances to the TUI for live progress display.

### Activation [dead code]

- **Startup** — was intended to run on first launch; on resume, any cached `workspace.json` would be loaded immediately while enrichment could refresh it
- **`/workspace:rebuild`** — not registered; handler is dead code

`.gekai/` directory excluded from workspace scan activation triggers.

### Scan phases (`scan_workspace` → `.gekai/workspace.json`) [dead code]

```
1. repo name (git remote) + branch
2. manifest scan  →  projects[]  (lang + path; skip hidden/vendor)
3. extension frequency  →  extensions{}  (top 15; fallback when no manifests)
4. AI instruction file detection  (root + one level deep)
```

**Cache read (still active):** `_init_session()` reads `.gekai/workspace.json` at startup if present and passes it to `agent.start_session()`. No live code writes the file; the `<workspace>` block in `Session.messages[1]` reflects whatever the file contains (or defaults to empty/unknown values if absent). Staleness enforcement and the enrichment write path are dead.

### Enrichment (`enrich_workspace`) [dead code]

**Two parallel** support-model calls — inactive:

```
┌─────────────────┐     ┌──────────────────┐
│   proj_brief    │     │   domain_map     │
│  (summary +     │     │  (feature areas  │
│   tech stack)   │     │   → file paths)  │
└─────────────────┘     └──────────────────┘
         └──────────────────┘
                 ▼
         workspace.json  (merged)  ← never reached; write path is dead
                 ▼
         injected into Session.messages[1]
         as <workspace> block      ← injection at startup is active; enrichment write is not
```

Async callbacks fired during enrichment: `on_file` · `on_infer_end`.

### Injected workspace fields

**Always:** `workspace_name`, `workspace_type`, `branch`, `primary_languages`, `projects`
**Conditional:** `extensions` (when `projects` empty) · `domain_map` (when present)
**Preamble:** `"verified repository metadata — treat as authoritative for high-level questions:"`

> The injection format above describes what `_format_workspace_context()` produces. This function is called by the active `agent.start_session()` path. The fields will reflect an existing `workspace.json` cache if one is present, otherwise default/empty values.
