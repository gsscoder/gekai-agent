# Architecture
Precision-scoped AI coding agent with checkpoint-oriented design and LLM-backed intent routing

## Package Layout
```
agent/
  main.py          — REPL entry point, CLI flags, command dispatch, streaming loop
  agent.py         — GekaiAgent: classifies intent, routes to handler
  router.py        — Intent enum, Session dataclass, IntentClassifier (LLM call)
  handlers/
    base.py        — Handler protocol (handle + stream)
    chat.py        — ChatHandler: LLM streaming, session message history
    query.py       — QueryHandler: stub, read-only tool use (future)
    action.py      — ActionHandler: stub, read/write tool use (future)
  commands/
    base.py        — Command protocol, CommandResult dataclass
    registry.py    — CommandRegistry: /name dispatch
    exit.py        — ExitCommand: /exit
```

## Session
`Session` lives in `router.py`; holds a GUID and a flat `messages: list[dict]` passed to LiteLLM on every call to maintain conversation context

## Intent Routing
`IntentClassifier` sends a single LLM call that decomposes the user message into one or more labeled segments, returning `list[tuple[Intent, str]]`; each tuple is `(intent, sub_prompt)`

Intents:
- `chat` — answerable from model knowledge, no external data needed
- `query` — needs external data: web search, docs lookup, repo file reads (read-only)
- `action` — modifies repository files (read + write)
- `clarify` — request too ambiguous to act on; sub_prompt states what is unclear

Classifier outputs one `{label}: {sub-prompt}` line per segment; falls back to `[(Intent.CHAT, user_input)]` on unparseable output; prefers `query` over `chat` when uncertain

`GekaiAgent.classify()` exposes classification as a separate async step; `process_stream()` accepts the resulting segments and streams handlers sequentially; a `clarify` segment short-circuits with questions, skipping all handlers

`GekaiAgent._handlers` is a `dict[Intent, Handler]`; `Intent.CLARIFY` has no handler entry

## LLM Integration
LiteLLM with OpenAI-compatible endpoint; env vars:
- `GEKAI_DEFAULT_MODEL` — model id, e.g. `openai/deepseek-v4-pro`
- `GEKAI_API_KEY`
- `GEKAI_BASE_URL` — e.g. `https://api.deepseek.com`

`api_key` and `api_base` passed directly to `acompletion()` (not via litellm global) so provider-specific routing works correctly

## Streaming UX
REPL uses `Rich.Live(transient=True)` during streaming:
- spinner `| / - \` + `Thinking... (↓ N tokens)` while receiving chunks
- on completion: spinner clears, blank line, `* Thought for Ns`, blank line, `● response`, blank line

Token counter shows raw integer below 1000, `k`-format above

## Commands
Slash-prefixed input (`/name args`) is intercepted before agent routing and dispatched via `CommandRegistry`
`CommandResult.exit_app=True` breaks the REPL loop
Current commands: `/exit`

## CLI Flags
`--debug` prints `debug: {intent}: {sub_prompt}` for each segment to console in grey50 after classification, before streaming
