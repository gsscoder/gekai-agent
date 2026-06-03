# Classifier Refactor
Demote IntentClassifier from planner to router + gatekeeper; move capability/permission decisions to tool-call-time gating

## Intent Enum (before -> after)

Before: `chat`, `action`, `action`, `memorize`, `clarify` (5 intents)
After: `chat`, `action`, `memorize` (3 intents)

Removals:
- `action` — merged into `action`; write-vs-read is a permission axis resolved at tool-call time by `PermissionGate`, not a semantic label. `ActionHandler` is an unimplemented stub
- `clarify` — was always routed to `ChatHandler` anyway; classifier should prefer `chat` over `clarify` per existing rules. Remove the indirection

## Classifier Prompt Changes

`CLASSIFIER_PROMPT` in `router.py` shrinks:
- Labels section: `chat | action | memorize` only
- `action` label description absorbs `action`: "needs to inspect or modify the repository"
- `clarify` removed; rule "prefer chat or action over clarify" becomes unnecessary
- Examples updated: `action:` lines become `action:` lines
- Rule "prefer least destructive: chat over action, action over action" simplifies to "prefer chat over action"

## Handler Changes

`_handlers` dict in `agent.py`:
- Remove `Intent.ACTION: ActionHandler()` entry
- Remove `ActionHandler` import
- Delete `agent/handlers/action.py`
- `Intent.ACTION` handler (`ActionHandler`) already has tools + `PermissionGate` — handles both read and write intents

No changes to `ChatHandler` or `ActionHandler` internals.

## process_stream Changes

Segment loop in `agent.py:161` simplifies:
- Remove `Intent.CLARIFY` branch (was routing to CHAT handler)
- Remove `Intent.ACTION` from dispatch (no longer exists)
- `Intent.MEMORIZE` branch unchanged
- All non-memorize segments dispatch to `_handlers[intent]` (either CHAT or ACTION)

The loop structure stays — classifier can still emit multiple segments (e.g. `memorize` + `action`). The single-order gate (`evaluate_single_order_gate`) ensures at most one non-memorize segment per turn.

## Single-Order Gate

`evaluate_single_order_gate` in `router.py` — one order per turn. An "order" is any non-`memorize` segment. If classifier output contains >1 order, the gate rejects with a message asking the user to split. Memorize directives ride along (setup, not an order). Toggled via `scope_gate` bool (`/config:gate on|off`). Replaces the old `evaluate_structural_gate` which used configurable `max_segments` and `big_prompt_min_words` thresholds.

## Permission Flow (unchanged)

`PermissionGate.check(tool)` already intercepts at tool-call time:
- `required_permission="read"` tools auto-approve when `permissions.read=True`
- `required_permission="write"` tools prompt via `permission_callback`
- `required_permission="none"` tools always pass

All current tools in `make_tools()` are `is_read_only=True, required_permission="read"`. When write tools are added, they will use `required_permission="write"` and the gate handles it — no classifier involvement needed.

## Read-Only Allowlist

Already in place via tool decorator: `@tool(is_read_only=True, required_permission="read")`. `PermissionGate` auto-approves these when session has `read=True`. Mirrors CC's `SAFE_YOLO_ALLOWLISTED_TOOLS` pattern without a separate allowlist — the permission is on the tool definition.

## Files Modified

- `agent/router.py` — `Intent` enum (remove ACTION, CLARIFY), `CLASSIFIER_PROMPT` (shrink labels/rules/examples)
- `agent/agent.py` — remove ActionHandler import, remove from `_handlers`, simplify `process_stream` branches
- `agent/handlers/action.py` — delete file
- `agent/handlers/base.py` — no changes (Protocol stays)

## Debug Output

`--debug` prints `[classifier: INTENT, ...]` in TUI. After refactor, output shows `chat`/`action`/`memorize` only. No structural change to debug rendering — just fewer label values.

## Fallback Behavior

Classifier parse failure still returns `[(Intent.CHAT, user_input)]` — unchanged. Unknown labels in LLM output still fall back to `Intent.CHAT` via the `except ValueError` catch.
