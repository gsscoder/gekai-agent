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


def test_route_trivial() -> None:
    router = _make_router()
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response("TRIVIAL"))
    route = run(router.route("hi there"))
    assert route.trivial
    assert route.subagent is None


def test_route_default_trivial_is_false() -> None:
    assert Route().trivial is False


def test_route_explore() -> None:
    router = _make_router()
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response("EXPLORE"))
    route = run(router.route("where is the Router class defined"))
    assert route.explore
    assert route.subagent is None


def test_route_default_explore_is_false() -> None:
    assert Route().explore is False


def test_route_rejected_with_name() -> None:
    router = _make_router()
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response("REJECTED agent-xxx"))
    route = run(router.route("use agent-xxx to do this"))
    assert route.rejected is True
    assert route.reason == "agent-xxx"
    assert route.subagent is None
    assert route.trivial is False
    assert route.explore is False


def test_route_rejected_bare() -> None:
    router = _make_router()
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response("REJECTED"))
    route = run(router.route("use some agent to do this"))
    assert route.rejected is True
    assert route.reason == ""


def test_route_known_subagent() -> None:
    router = _make_router()
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response("code-expert"))
    route = run(router.route("add a new endpoint to the API"))
    assert route.subagent is not None
    assert route.subagent.name == "code-expert"
    assert route.subagent.namespace == "coding"


def test_route_unknown_token_falls_back_to_main() -> None:
    router = _make_router()
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response("nonsense-token"))
    route = run(router.route("do something"))
    assert route.subagent is None
    assert route.rejected is False


def test_route_strips_extra_whitespace() -> None:
    router = _make_router()
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response("  main  "))
    route = run(router.route("explain closures"))
    assert route.subagent is None


def test_route_plan_two_steps() -> None:
    router = _make_router()
    plan_text = (
        "<plan>\n"
        "code-expert:\n"
        "  add a /healthz endpoint to the API\n"
        "test-expert:\n"
        "  write tests for the new endpoint\n"
    )
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response(plan_text))
    route = run(router.route("add a /healthz endpoint to the API and write tests for it"))
    assert route.plan is not None
    assert len(route.plan) == 2
    assert route.plan[0].subagent.name == "code-expert"
    assert route.plan[0].raw == "add a /healthz endpoint to the API"
    assert route.plan[1].subagent.name == "test-expert"
    assert route.plan[1].raw == "write tests for the new endpoint"
    assert route.subagent is None


def test_route_plan_with_main_step() -> None:
    router = _make_router()
    plan_text = (
        "<plan>\n"
        "main:\n"
        "  explain the current architecture\n"
        "code-expert:\n"
        "  then refactor the router module\n"
    )
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response(plan_text))
    route = run(router.route("explain the architecture then refactor the router module"))
    assert route.plan is not None
    assert len(route.plan) == 2
    assert route.plan[0].subagent is None
    assert route.plan[0].raw == "explain the current architecture"
    assert route.plan[1].subagent.name == "code-expert"
    assert route.plan[1].raw == "then refactor the router module"


def test_route_plan_single_step_collapses_to_subagent() -> None:
    router = _make_router()
    plan_text = (
        "<plan>\n"
        "code-expert:\n"
        "  create sqrt.py and code the Quake version of the function inside\n"
    )
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response(plan_text))
    route = run(router.route("create sqrt.py and code the Quake version of the function inside"))
    assert route.plan is None
    assert route.subagent is not None
    assert route.subagent.name == "code-expert"


def test_route_plan_single_step_main_collapses_to_main() -> None:
    router = _make_router()
    plan_text = "<plan>\nmain:\n  explain how async/await works\n"
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response(plan_text))
    route = run(router.route("explain how async/await works"))
    assert route.plan is None
    assert route.subagent is None


def test_route_plan_unknown_agent_falls_back_to_main() -> None:
    router = _make_router()
    plan_text = (
        "<plan>\n"
        "code-expert:\n"
        "  do the first part\n"
        "nonsense-agent:\n"
        "  do the second part\n"
    )
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response(plan_text))
    route = run(router.route("do something with two parts"))
    assert route.plan is None
    assert route.subagent is None


def test_route_plan_zero_blocks_falls_back_to_main() -> None:
    router = _make_router()
    plan_text = "<plan>\nthis is not a valid block at all\n"
    router._client.chat.completions.create = AsyncMock(return_value=_mock_response(plan_text))
    route = run(router.route("do something"))
    assert route.plan is None
    assert route.subagent is None


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
