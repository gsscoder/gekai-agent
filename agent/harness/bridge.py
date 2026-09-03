"""The bus-to-queue bridge: one translation from the vendored `agent/llm`
event vocabulary into Gekai's own `AgentEvent` taxonomy.

Both harness paths (the single-agent turn and the task-graph interpreter)
share this, so tool/diff/thinking/delegation rendering has one implementation
rather than one per path. It is also the only place that inspects a tool
call's shape to derive a diff, which keeps that knowledge out of the stream
logic in `core.py`.
"""

from __future__ import annotations

import asyncio
import json

from ..diff import build_diff
from ..events import (
    BudgetExhaustedEvent,
    DelegationDoneEvent,
    DelegationStartEvent,
    DiffEvent,
    ForeignFileDetectedEvent,
    InferEndEvent,
    LogEvent,
    TextChunkEvent,
    ThinkingTokenEvent,
)
from ..llm.events import (
    AgentStopped,
    DelegationCompleted,
    DelegationStarted,
    Event as LlmEvent,
    TextChunkReceived,
    ThinkingChunkReceived,
    ToolExecutionCompleted,
    ToolExecutionStarted,
    UsageUpdated,
)
from ..llm.types import ToolUseBlock
from ..persistence import append_debug
from ..session import Session

# Tools that mutate exactly one path, named by `path`, and those that name two
# (`src`/`dst`) — both count toward the turn's mutation tally.
_SINGLE_PATH_TOOLS = ("write_file", "edit_file", "make_dir", "delete_file")
_DUAL_PATH_TOOLS = ("move_file", "copy_file")

_DEBUG_TRUNCATE_LIMIT = 1000
_DIFFS_BLOCK_CHAR_LIMIT = 8000


def fmt_tool_call(call: ToolUseBlock) -> str:
    inp = call.input or {}
    if call.name == "read_file":
        return f"Read {inp.get('path', '')}"
    if call.name == "list_files":
        return f"List {inp.get('pattern', '')}"
    if call.name == "grep":
        pat = inp.get("pattern", "")
        path = inp.get("path", "")
        return f"Grep {pat}" + (f" in {path}" if path else "")
    if call.name == "edit_file":
        return f"Edit {inp.get('path', '')}"
    if call.name == "write_file":
        return f"Write {inp.get('path', '')}"
    if call.name == "move_file":
        return f"Move {inp.get('src', '')} → {inp.get('dst', '')}"
    if call.name == "copy_file":
        return f"Copy {inp.get('src', '')} → {inp.get('dst', '')}"
    if call.name == "delete_file":
        return f"Delete {inp.get('path', '')}"
    if call.name == "make_dir":
        return f"Mkdir {inp.get('path', '')}"
    if call.name == "run_command":
        return f"Run {inp.get('command', '')[:60]}"
    return call.name.capitalize()


def truncate_debug_text(text: str) -> str:
    if len(text) <= _DEBUG_TRUNCATE_LIMIT:
        return text
    return text[:_DEBUG_TRUNCATE_LIMIT] + f"…+{len(text) - _DEBUG_TRUNCATE_LIMIT} more chars"


def format_diff_summary(event: DiffEvent) -> str:
    """Renders a DiffEvent as compact plain text (add/del lines only) for
    root's synthesis prompt, which needs a plain string, not the rich.Text
    that `render_diff` (agent/diff.py) produces for TUI display."""
    lines = [f"--- {event.path} ---"]
    for dl in event.diff_lines:
        if dl.kind == "add":
            lines.append(f"+{dl.text}")
        elif dl.kind == "del":
            lines.append(f"-{dl.text}")
    return "\n".join(lines)


def truncate_diffs_block(text: str) -> str:
    if len(text) <= _DIFFS_BLOCK_CHAR_LIMIT:
        return text
    return text[:_DIFFS_BLOCK_CHAR_LIMIT] + "... (truncated)"


def maybe_flag_foreign_instruction_file(event: LlmEvent, queue: asyncio.Queue) -> None:
    """The foreign-file trigger, hung off the same `ToolExecutionCompleted`
    moment `bridge_llm_event` already reacts to for `edit_file`/`write_file`
    — kept as its own function rather than a branch inside that one because
    `bridge_llm_event` is shared with the graph path's cold subagent steps,
    which must never reach here; the caller invokes this only under its own
    `subagent is None` guard.

    Markdown-only, no prefilter: every root-dispatched `.md` `read_file`
    result fires unconditionally; the one cheap LLM question
    (`agent.directive_audit.Auditor`) is the only judgment left, downstream.
    Whether the audit is even enabled is checked once, downstream, in
    `GekaiAgent.start_foreign_file_audit` — the same place GEKAI.md's own
    audit already checks it; not duplicated here.
    """
    if not isinstance(event, ToolExecutionCompleted):
        return
    if event.result.is_error or event.call.name != "read_file":
        return
    path = (event.call.input or {}).get("path", "")
    if not path.lower().endswith(".md"):
        return
    queue.put_nowait(ForeignFileDetectedEvent(rel_path=path, text=event.result.content))


def _fmt_debug_tool_input(call: ToolUseBlock) -> dict:
    """Debug-log representation of a tool call's input.

    write_file/edit_file inputs carry full file contents or old/new diff strings —
    these are logged as path-only (the path is never truncated; it's always short and
    is the only part of those inputs that's useful for debugging without bloating the
    debug log with entire file bodies). All other tools get their full input dict,
    JSON-serialized and truncated like any other debug text.
    """
    inp = call.input or {}
    if call.name in ("write_file", "edit_file"):
        return {"name": call.name, "path": inp.get("path", "")}
    return {"name": call.name, "input": truncate_debug_text(json.dumps(inp, default=str))}


def bridge_llm_event(
    event: LlmEvent,
    queue: asyncio.Queue,
    session: Session,
    files_touched: list[str],
    verbose_telemetry: bool,
    run_id: str | None,
    *,
    emit_text_chunks: bool = False,
    diff_sink: list[DiffEvent] | None = None,
    mutation_count: list[int] | None = None,
) -> None:
    """Shared bus->queue bridge for both the single-agent path and the
    interpreter path.

    `emit_text_chunks`: only the no-graph direct-dispatch caller passes
    `True`. The graph path leaves it `False` so a graph-routed step's streamed
    answer text — which would otherwise interleave with `LogEvent`/`DiffEvent`
    in a multi-step transcript — never reaches the TUI. Display-only either
    way: the persisted answer always comes from the assembled response, never
    from these chunks.

    `diff_sink`/`mutation_count` (verifier gate, graph path only): mirrors of
    `queue.put_nowait`/`files_touched.append` that a caller can read
    synchronously right after `await`-ing the dispatch that produced them —
    unlike `queue`, which a separate consumer drains at its own pace, and
    `files_touched`, which dedupes across the whole turn so a second touch of
    an already-known path is invisible.
    """
    if isinstance(event, ToolExecutionStarted):
        queue.put_nowait(LogEvent(message=fmt_tool_call(event.call), tool_name=event.call.name))
        if verbose_telemetry:
            append_debug(session, {"content": {"tool_call": _fmt_debug_tool_input(event.call)}})
    elif isinstance(event, ToolExecutionCompleted):
        _bridge_tool_completed(
            event, queue, session, files_touched, verbose_telemetry,
            diff_sink=diff_sink, mutation_count=mutation_count,
        )
    elif isinstance(event, UsageUpdated) and event.delta:
        queue.put_nowait(InferEndEvent(
            prompt_tokens=event.delta.get("input_tokens"),
            completion_tokens=event.delta.get("output_tokens"),
        ))
    elif isinstance(event, ThinkingChunkReceived):
        queue.put_nowait(ThinkingTokenEvent(text=event.text))
    elif isinstance(event, TextChunkReceived):
        if emit_text_chunks:
            queue.put_nowait(TextChunkEvent(text=event.text))
    elif isinstance(event, DelegationStarted):
        queue.put_nowait(DelegationStartEvent(agent_name=event.agent, task=event.task, mission=event.mission))
    elif isinstance(event, DelegationCompleted):
        queue.put_nowait(DelegationDoneEvent(agent_name=event.agent))
    elif isinstance(event, AgentStopped) and run_id is not None:
        if event.run_id != run_id:
            return  # a nested run's own completion — not ours
        if event.budget_exhausted:
            queue.put_nowait(BudgetExhaustedEvent())
        queue.put_nowait(None)


def _bridge_tool_completed(
    event: ToolExecutionCompleted,
    queue: asyncio.Queue,
    session: Session,
    files_touched: list[str],
    verbose_telemetry: bool,
    *,
    diff_sink: list[DiffEvent] | None,
    mutation_count: list[int] | None,
) -> None:
    # A tool can fail without raising — e.g. `edit_file`'s atomic `edits`
    # batch writes nothing and returns an "error: ..." string when any one
    # hunk doesn't apply — so `is_error` alone under-detects failure; every
    # failure path in agent/tools/files.py returns that prefix.
    succeeded = not event.result.is_error and not event.result.content.startswith("error:")
    inp = event.call.input or {}

    def emit_diff(path: str, old: str, new: str, via: str) -> None:
        if old == new:
            return
        diff_event = DiffEvent(path=path, diff_lines=build_diff(old, new), via=via)
        queue.put_nowait(diff_event)
        if diff_sink is not None:
            diff_sink.append(diff_event)

    def count_mutation(path: str) -> None:
        if not path:
            return
        if mutation_count is not None:
            mutation_count[0] += 1
        if path not in files_touched:
            files_touched.append(path)

    if succeeded:
        if event.call.name == "edit_file":
            path = inp.get("path", "")
            edits = inp.get("edits")
            hunks = edits if edits else [{"old_str": inp.get("old_str", ""), "new_str": inp.get("new_str", "")}]
            for hunk in hunks:
                emit_diff(path, hunk.get("old_str", ""), hunk.get("new_str", ""), "edit_file")
        elif event.call.name == "write_file":
            content = inp.get("content", "")
            if content:
                via = "overwrite" if event.result.content.startswith("ok: overwritten") else "write_file"
                emit_diff(inp.get("path", ""), "", content, via)

        if event.call.name in _SINGLE_PATH_TOOLS:
            count_mutation(inp.get("path", ""))
        elif event.call.name in _DUAL_PATH_TOOLS:
            count_mutation(inp.get("src", ""))
            count_mutation(inp.get("dst", ""))

    if verbose_telemetry:
        append_debug(session, {
            "content": {
                "tool_result": {
                    "name": event.call.name,
                    "duration_s": round(event.duration_s, 3),
                    "is_error": event.result.is_error,
                    "result": truncate_debug_text(event.result.content),
                }
            }
        })


__all__ = [
    "bridge_llm_event",
    "fmt_tool_call",
    "format_diff_summary",
    "maybe_flag_foreign_instruction_file",
    "truncate_debug_text",
    "truncate_diffs_block",
]
