from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent.rewriter import PromptRewriter, _format_entries


def run(coro):
    return asyncio.run(coro)


def _make_rewriter() -> PromptRewriter:
    with patch("agent.rewriter.AsyncOpenAI"):
        return PromptRewriter(model="core-model", api_key="key", api_base="http://localhost")


def _mock_response(text: str | None):
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice])


# ---------------------------------------------------------------------------
# _format_entries
# ---------------------------------------------------------------------------

def test_format_entries_path_with_keywords() -> None:
    out = _format_entries([("src/a.py", ["alpha", "beta"])])
    assert out == "src/a.py | alpha, beta"


def test_format_entries_path_without_keywords() -> None:
    out = _format_entries([("src/a.py", [])])
    assert out == "src/a.py"


def test_format_entries_multiple_lines() -> None:
    out = _format_entries([("src/a.py", ["x"]), ("src/b.py", ["y", "z"])])
    assert out == "src/a.py | x\nsrc/b.py | y, z"


# ---------------------------------------------------------------------------
# rewrite
# ---------------------------------------------------------------------------

def test_rewrite_returns_stripped_text() -> None:
    rw = _make_rewriter()
    rw._client.chat.completions.create = AsyncMock(
        return_value=_mock_response("  update 'src/a.py' to do x  ")
    )
    out = run(rw.rewrite("update the thing to do x", [("src/a.py", ["thing"])]))
    assert out == "update 'src/a.py' to do x"


def test_rewrite_raises_on_empty_output() -> None:
    rw = _make_rewriter()
    rw._client.chat.completions.create = AsyncMock(return_value=_mock_response(""))
    with pytest.raises(ValueError):
        run(rw.rewrite("do x", [("src/a.py", [])]))


def test_rewrite_raises_on_none_content() -> None:
    rw = _make_rewriter()
    rw._client.chat.completions.create = AsyncMock(return_value=_mock_response(None))
    with pytest.raises(ValueError):
        run(rw.rewrite("do x", [("src/a.py", [])]))


def test_rewrite_uses_core_model_and_zero_temp() -> None:
    rw = _make_rewriter()
    create = AsyncMock(return_value=_mock_response("ok"))
    rw._client.chat.completions.create = create
    run(rw.rewrite("do x", [("src/a.py", ["x"])]))
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "core-model"
    assert kwargs["temperature"] == 0


def test_rewrite_sends_no_thinking_params() -> None:
    rw = _make_rewriter()
    create = AsyncMock(return_value=_mock_response("ok"))
    rw._client.chat.completions.create = create
    run(rw.rewrite("do x", [("src/a.py", ["x"])]))
    kwargs = create.call_args.kwargs
    # non-thinking call: no reasoning/thinking params emitted
    assert "reasoning_effort" not in kwargs
    assert "extra_body" not in kwargs
    assert "thinking" not in kwargs


def test_rewrite_passes_request_and_files_to_model() -> None:
    rw = _make_rewriter()
    create = AsyncMock(return_value=_mock_response("ok"))
    rw._client.chat.completions.create = create
    run(rw.rewrite("update the passcode dialog", [("src/auth/pass.tsx", ["passcode", "dialog"])]))
    messages = create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "update the passcode dialog" in user
    assert "src/auth/pass.tsx | passcode, dialog" in user
