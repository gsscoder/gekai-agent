from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.pipeline.router import Route, Router
from agent.pipeline._directives import PIPELINE_DIRECTIVES


def run(coro):
    return asyncio.run(coro)


def _make_router() -> Router:
    with patch("agent.pipeline.router.AsyncOpenAI"):
        return Router(model="test-model", api_key="key", api_base="http://localhost")


def _mock_response(text: str):
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice])


def test_router_system_prompt_contains_pipeline_directives() -> None:
    router = _make_router()
    assert router._prompt.startswith(PIPELINE_DIRECTIVES)


def test_route_main() -> None:
    router = _make_router()
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response("main"))
    route = run(router.route("how does async/await work in Python"))
    assert route.subagent is None
    assert not route.rejected


def test_route_rejected() -> None:
    router = _make_router()
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response("REJECTED"))
    route = run(router.route("¿cómo funciona esto?"))
    assert route.rejected
    assert route.subagent is None


def test_route_known_subagent() -> None:
    router = _make_router()
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response("code-expert"))
    route = run(router.route("add a new endpoint to the API"))
    assert not route.rejected
    assert route.subagent is not None
    assert route.subagent.name == "code-expert"
    assert route.subagent.namespace == "coding"


def test_route_unknown_token_falls_back_to_main() -> None:
    router = _make_router()
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response("nonsense-token"))
    route = run(router.route("do something"))
    assert not route.rejected
    assert route.subagent is None


def test_route_strips_extra_whitespace() -> None:
    router = _make_router()
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response("  main  "))
    route = run(router.route("explain closures"))
    assert route.subagent is None
    assert not route.rejected


def test_route_passes_history_to_model() -> None:
    router = _make_router()
    create_mock = AsyncMock(return_value=_mock_response("main"))
    router._client.chat.completions.create = create_mock
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "prev question"},
        {"role": "assistant", "content": "prev answer"},
    ]
    run(router.route("follow-up question", history=history))
    call_messages = create_mock.call_args.kwargs["messages"]
    # router system prompt + 1 user/assistant pair + current input = 4 messages
    assert len(call_messages) == 4
    assert call_messages[1]["role"] == "user"
    assert call_messages[1]["content"] == "prev question"
