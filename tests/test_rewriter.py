from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent.pipeline.rewriter import PromptRewriter, _format_entries, _split_ui_label
from agent.pipeline._directives import PIPELINE_DIRECTIVES
from tests.conftest import mock_llm_response, run


def _make_rewriter() -> PromptRewriter:
    with patch("agent.openai_client.AsyncOpenAI"):
        return PromptRewriter(model="core-model", api_key="key", api_base="http://localhost")


# ---------------------------------------------------------------------------
# _format_entries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entries, expected",
    [
        ([("src/a.py", ["alpha", "beta"])], "src/a.py | alpha, beta"),
        ([("src/a.py", [])], "src/a.py"),
        ([("src/a.py", ["x"]), ("src/b.py", ["y", "z"])], "src/a.py | x\nsrc/b.py | y, z"),
    ],
)
def test_format_entries_path_with_keywords(entries, expected) -> None:
    out = _format_entries(entries)
    assert out == expected


# ---------------------------------------------------------------------------
# rewrite
# ---------------------------------------------------------------------------

def test_rewriter_system_prompt_contains_pipeline_directives() -> None:
    rw = _make_rewriter()
    create = AsyncMock(return_value=mock_llm_response("ok"))
    rw._client.chat.completions.create = create
    run(rw.rewrite("do x", [("src/a.py", [])]))
    system = create.call_args.kwargs["messages"][0]["content"]
    assert system.startswith(PIPELINE_DIRECTIVES)


def test_rewrite_returns_stripped_text_and_empty_label_when_absent() -> None:
    rw = _make_rewriter()
    rw._client.chat.completions.create = AsyncMock(
        return_value=mock_llm_response("  update 'src/a.py' to do x  ")
    )
    rewritten, ui_label = run(rw.rewrite("update the thing to do x", [("src/a.py", ["thing"])]))
    assert rewritten == "update 'src/a.py' to do x"
    assert ui_label == ""


def test_rewrite_splits_off_ui_label() -> None:
    rw = _make_rewriter()
    rw._client.chat.completions.create = AsyncMock(
        return_value=mock_llm_response("<ui_label>fix login bug</ui_label>\nupdate `src/a.py` to do x")
    )
    rewritten, ui_label = run(rw.rewrite("fix the login bug", [("src/a.py", ["login"])]))
    assert rewritten == "update `src/a.py` to do x"
    assert ui_label == "fix login bug"


@pytest.mark.parametrize("content", ["", None])
def test_rewrite_raises_on_empty_output(content) -> None:
    rw = _make_rewriter()
    rw._client.chat.completions.create = AsyncMock(return_value=mock_llm_response(content))
    with pytest.raises(ValueError):
        run(rw.rewrite("do x", [("src/a.py", [])]))


def test_rewrite_uses_core_model_and_zero_temp() -> None:
    rw = _make_rewriter()
    create = AsyncMock(return_value=mock_llm_response("ok"))
    rw._client.chat.completions.create = create
    run(rw.rewrite("do x", [("src/a.py", ["x"])]))
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "core-model"
    assert kwargs["temperature"] == 0


def test_rewrite_sends_no_thinking_params() -> None:
    rw = _make_rewriter()
    create = AsyncMock(return_value=mock_llm_response("ok"))
    rw._client.chat.completions.create = create
    run(rw.rewrite("do x", [("src/a.py", ["x"])]))
    kwargs = create.call_args.kwargs
    # non-thinking call: no reasoning/thinking params emitted
    assert "reasoning_effort" not in kwargs
    assert "extra_body" not in kwargs
    assert "thinking" not in kwargs


def test_rewrite_passes_request_and_files_to_model() -> None:
    rw = _make_rewriter()
    create = AsyncMock(return_value=mock_llm_response("ok"))
    rw._client.chat.completions.create = create
    run(rw.rewrite("update the passcode dialog", [("src/auth/pass.tsx", ["passcode", "dialog"])]))
    messages = create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    system = messages[0]["content"]
    assert "update the passcode dialog" in system
    user = messages[1]["content"]
    assert "src/auth/pass.tsx | passcode, dialog" in user


# ---------------------------------------------------------------------------
# _split_ui_label — fail-soft: a missing/malformed label must never propagate
# an error, only degrade to "" so the caller can fall back
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected_label, expected_rest",
    [
        (
            "<ui_label>fix login bug</ui_label>\nupdate `src/a.py`",
            "fix login bug",
            "update `src/a.py`",
        ),
        (
            "update `src/a.py` to do x",
            "",
            "update `src/a.py` to do x",
        ),
        (
            "<ui_label>only a label</ui_label>",
            "only a label",
            "<ui_label>only a label</ui_label>",
        ),
    ],
)
def test_split_ui_label_extracts_label_and_strips_it_from_body(text, expected_label, expected_rest) -> None:
    label, rest = _split_ui_label(text)
    assert label == expected_label
    assert rest == expected_rest
