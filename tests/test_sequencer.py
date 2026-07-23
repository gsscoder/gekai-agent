from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from agent.pipeline.sequencer import Sequencer
from tests.conftest import mock_llm_response, run


def _make_sequencer() -> Sequencer:
    with patch("openai.AsyncOpenAI"):
        return Sequencer(model="test-model", api_key="key")


def test_phase1_yields_code_then_test_expert_in_order() -> None:
    sequencer = _make_sequencer()
    raw = json.dumps({
        "summary": "build a library with tests",
        "steps": [
            {"agent": "code-expert", "instruction": "write the library", "mission": "write the library", "verify": "mechanical"},
            {"agent": "test-expert", "instruction": "test {{step_1}}", "mission": "test the library", "verify": "mechanical"},
        ],
    })
    sequencer._client.chat.completions.create = AsyncMock(return_value=mock_llm_response(raw))

    graph = run(sequencer.sequence("build a library with tests"))

    assert graph.summary == "build a library with tests"
    assert [s.agent for s in graph] == ["code-expert", "test-expert"]
    assert all(s.verify for s in graph)


def test_trivial_single_duty_yields_one_step_no_verify() -> None:
    sequencer = _make_sequencer()
    raw = json.dumps({
        "summary": "add minimal test coverage",
        "steps": [{"agent": "test-expert", "instruction": "add coverage to calcexpr", "mission": "add coverage", "verify": None}],
    })
    sequencer._client.chat.completions.create = AsyncMock(return_value=mock_llm_response(raw))

    graph = run(sequencer.sequence("add minimal test coverage"))

    assert len(graph) == 1
    assert graph[0].verify is None


def test_agent_x_seed_assigns_primary_step_to_seed_agent() -> None:
    sequencer = _make_sequencer()
    raw = json.dumps({
        "summary": "add coverage",
        "steps": [{"agent": "test-expert", "instruction": "add coverage", "mission": "add coverage", "verify": None}],
    })
    mock_create = AsyncMock(return_value=mock_llm_response(raw))
    sequencer._client.chat.completions.create = mock_create

    graph = run(sequencer.sequence("add coverage", seed="test-expert"))

    # the seed must reach the outbound prompt (mechanism)...
    sent_user_message = mock_create.call_args.kwargs["messages"][1]["content"]
    assert "test-expert" in sent_user_message
    # ...and the parsed graph's primary (first) step must actually carry it (outcome).
    assert graph[0].agent == "test-expert"


def test_post_planning_only_agent_in_phase1_raises() -> None:
    sequencer = _make_sequencer()
    raw = json.dumps({
        "summary": "review this",
        "steps": [{"agent": "code-refactorer", "instruction": "review the change", "mission": "review the change", "verify": None}],
    })
    sequencer._client.chat.completions.create = AsyncMock(return_value=mock_llm_response(raw))

    with pytest.raises(ValueError, match="code-refactorer"):
        run(sequencer.sequence("review this"))


def test_no_json_object_in_output_raises() -> None:
    sequencer = _make_sequencer()
    sequencer._client.chat.completions.create = AsyncMock(return_value=mock_llm_response("not json at all"))

    with pytest.raises(ValueError, match="no JSON object"):
        run(sequencer.sequence("do something"))
