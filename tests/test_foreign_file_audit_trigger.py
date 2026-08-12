"""Plan 35 v3: the foreign-file directive-audit trigger.

Two layers, matching the split `test_directive_audit_wiring.py` already
uses for GEKAI.md:

- `Harness.stream`'s `ToolExecutionCompleted` hook
  (`agent.harness.core._maybe_flag_foreign_instruction_file`): a root-only,
  markdown-only `read_file` branch that yields a `ForeignFileDetectedEvent`
  unconditionally — no prefilter (v3 deletes it), never a model call by
  itself.
- `agent.harness.turn.run_step`: re-emits that event into
  `agent.start_foreign_file_audit`, which is exactly
  `test_directive_audit_wiring.py`'s already-covered cache/call/swallow
  flow, just fed a transient file instead of `session.gekai_md`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.events import ForeignFileDetectedEvent
from agent.harness import core as harness_core
from agent.harness import turn as harness_turn
from agent.harness.core import Harness
from agent.llm.providers.base import ProviderAdapter
from agent.llm.resolve import ResolvedTier
from agent.llm.tiers import TierName, TierPolicy
from agent.llm.types import CompletionResponse, StreamDone, StreamEvent, TextBlock, ToolUseBlock
from agent.session import Session
from agent.settings import Permissions
from agent.subagents import Subagent


def run(coro):
    return asyncio.run(coro)


class _ScriptedAdapter(ProviderAdapter):
    responses: list[CompletionResponse] = []

    def __init__(self, *args, **kwargs) -> None:
        self._responses = list(type(self).responses)
        self.calls: list[dict] = []

    async def complete(self, **kwargs) -> CompletionResponse:
        raise NotImplementedError

    async def stream(self, **kwargs):
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        yield StreamDone(response=self._responses[index])


def _make_session(tmp_path: Path) -> Session:
    return Session(working_dir=tmp_path, permissions=Permissions(read=True, write=True, exec=True))


def _make_harness() -> Harness:
    tier = ResolvedTier(model="test-model", api_key="key", api_base="http://localhost", extra_params={})
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE))
    return Harness(
        resolve=lambda _tier, _touchpoint: tier,
        sequencer_policy=TierPolicy(default=TierName.CORE, allowed=(TierName.SUPP, TierName.CORE)),
        root_dispatch_policy=policy,
        subagent_dispatch_policy=policy,
    )


async def _drain(harness: Harness, session: Session, prompt: str, **kwargs) -> list:
    collected = []
    async for item in harness.stream(session, prompt, **kwargs):
        collected.append(item)
    return collected


def _only(events: list, cls: type) -> list:
    return [e for e in events if isinstance(e, cls)]


_INSTRUCTION_TEXT = (
    "# Agent Instructions\n\nYou are an AI coding assistant working in this repository.\n\n"
    "## Rules\n- Always run tests before committing\n"
)
_README_TEXT = "# MyProject\n\nA lightweight command-line tool for converting CSV files to JSON.\n"


def _read_file_response(path: str, call_id: str = "rf-1") -> CompletionResponse:
    return CompletionResponse(
        content=[ToolUseBlock(id=call_id, name="read_file", input={"path": path})],
        stop_reason="tool_use",
    )


_DONE = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")


# ---------------------------------------------------------------------------
# Harness.stream: the ToolExecutionCompleted hook — unconditional on .md
# ---------------------------------------------------------------------------


def test_markdown_read_on_root_fires_the_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text(_INSTRUCTION_TEXT, encoding="utf-8")
    _ScriptedAdapter.responses = [_read_file_response("AGENTS.md"), _DONE]
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    session = _make_session(tmp_path)
    collected = run(_drain(_make_harness(), session, "read AGENTS.md to get the project brief"))

    fired = _only(collected, ForeignFileDetectedEvent)
    assert len(fired) == 1
    assert fired[0].rel_path == "AGENTS.md"
    assert fired[0].text == _INSTRUCTION_TEXT


def test_plain_readme_still_fires_the_event_no_prefilter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """v3 deletes the regex prefilter entirely (decision 5) — every
    root-dispatched `.md` `read_file` fires, including an ordinary README.
    The one cheap LLM question is the only judgment left, downstream."""
    (tmp_path / "README.md").write_text(_README_TEXT, encoding="utf-8")
    _ScriptedAdapter.responses = [_read_file_response("README.md"), _DONE]
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    session = _make_session(tmp_path)
    collected = run(_drain(_make_harness(), session, "read README.md"))

    fired = _only(collected, ForeignFileDetectedEvent)
    assert len(fired) == 1
    assert fired[0].rel_path == "README.md"
    assert fired[0].text == _README_TEXT


def test_non_markdown_read_never_fires(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    (tmp_path / "notes.py").write_text(f'"""{_INSTRUCTION_TEXT}"""\n', encoding="utf-8")
    _ScriptedAdapter.responses = [_read_file_response("notes.py"), _DONE]
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    session = _make_session(tmp_path)
    collected = run(_drain(_make_harness(), session, "read notes.py"))

    assert _only(collected, ForeignFileDetectedEvent) == []


def test_subagent_read_of_a_markdown_file_fires_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text(_INSTRUCTION_TEXT, encoding="utf-8")
    _ScriptedAdapter.responses = [_read_file_response("AGENTS.md"), _DONE]
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    sub = Subagent(name="t", namespace="coding", description="d")
    session = _make_session(tmp_path)
    collected = run(_drain(_make_harness(), session, "read AGENTS.md", subagent=sub))

    assert _only(collected, ForeignFileDetectedEvent) == []


# ---------------------------------------------------------------------------
# harness.turn.run_step: re-emits ForeignFileDetectedEvent into
# agent.start_foreign_file_audit, threading the on_directive_verdict callback
# ---------------------------------------------------------------------------


class _FakeAgent:
    def __init__(self, items: list) -> None:
        self._items = items
        self.events = SimpleNamespace(emit=lambda *a, **kw: None)
        self.foreign_calls: list[tuple] = []

    async def process_stream(self, session, raw, **kwargs):
        for item in self._items:
            yield item

    def start_foreign_file_audit(self, rel_path: str, text: str, on_verdict=None) -> None:
        self.foreign_calls.append((rel_path, text, on_verdict))


def test_run_step_fires_start_foreign_file_audit_with_the_verdict_callback(tmp_path: Path) -> None:
    event = ForeignFileDetectedEvent(rel_path="AGENTS.md", text=_INSTRUCTION_TEXT)
    agent = _FakeAgent([event])
    session = _make_session(tmp_path)
    seen_verdicts: list = []

    async def _run():
        return await harness_turn.run_step(
            agent, session, "read AGENTS.md", None,
            turn_id="t1", session_id="s1",
            permission_callback=None, hidden_grant_callback=None,
            on_directive_verdict=lambda path, v: seen_verdicts.append((path, v)),
        )

    run(_run())

    assert agent.foreign_calls == [("AGENTS.md", _INSTRUCTION_TEXT, agent.foreign_calls[0][2])]
    # the exact callback passed through must be the one run_step was given
    assert agent.foreign_calls[0][2]("AGENTS.md", None) is None
    assert seen_verdicts == [("AGENTS.md", None)]
