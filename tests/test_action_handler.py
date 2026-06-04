from __future__ import annotations

import pytest

from agent.handlers.action import _recency_turns, _build_agent, _RECENCY_N, _TOOL_INSTRUCTION
from agent.llm.types import Message
from agent.router import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# _recency_turns
# ---------------------------------------------------------------------------

def _msgs(*pairs: tuple[str, str]) -> list[dict]:
    """Build a session.messages list: system + n user/assistant pairs + trailing user."""
    out: list[dict] = [{"role": "system", "content": "workspace context"}]
    for user, assistant in pairs:
        out.append({"role": "user", "content": user})
        out.append({"role": "assistant", "content": assistant})
    out.append({"role": "user", "content": "current request"})
    return out


def test_recency_turns_returns_last_n_pairs():
    messages = _msgs(("u1", "a1"), ("u2", "a2"), ("u3", "a3"))
    result = _recency_turns(messages, 2)
    assert len(result) == 4
    assert [m.role for m in result] == ["user", "assistant", "user", "assistant"]
    assert result[0].content == "u2"
    assert result[3].content == "a3"


def test_recency_turns_excludes_current_user_input():
    messages = _msgs(("u1", "a1"))
    result = _recency_turns(messages, 2)
    # current request ("current request") must NOT appear
    contents = [m.content for m in result]
    assert "current request" not in contents


def test_recency_turns_no_system_messages():
    messages = _msgs(("u1", "a1"), ("u2", "a2"))
    result = _recency_turns(messages, 2)
    assert all(m.role != "system" for m in result)


def test_recency_turns_fewer_turns_than_n():
    messages = _msgs(("u1", "a1"))
    result = _recency_turns(messages, 2)
    assert len(result) == 2
    assert result[0].content == "u1"
    assert result[1].content == "a1"


def test_recency_turns_empty_history():
    # only system + current user input, no prior turns
    messages = [
        {"role": "system", "content": "ws"},
        {"role": "user", "content": "current"},
    ]
    result = _recency_turns(messages, 2)
    assert result == []


def test_recency_n_constant():
    assert _RECENCY_N == 2


# ---------------------------------------------------------------------------
# system prompt structure
# ---------------------------------------------------------------------------

def _fake_profile(directives: str):
    from agent.profiles import AgentProfile
    return AgentProfile(
        name="test-profile",
        namespace="coding",
        description="test",
        directives=directives,
    )


def _compose_system(profile=None) -> str:
    system = SYSTEM_PROMPT
    if profile and profile.directives:
        system += f"\n<directives>\n{profile.directives}"
    system += f"\n<tools>\n{_TOOL_INSTRUCTION}"
    return system


def test_system_prompt_has_tools_block_always():
    system = _compose_system(profile=None)
    assert "<tools>\n" in system


def test_system_prompt_has_directives_block_with_profile():
    profile = _fake_profile("do not invent features")
    system = _compose_system(profile=profile)
    assert "<directives>\n" in system
    assert "do not invent features" in system


def test_system_prompt_no_directives_block_without_profile():
    system = _compose_system(profile=None)
    assert "<directives>" not in system


def test_system_prompt_no_directives_block_empty_directives():
    profile = _fake_profile("")
    system = _compose_system(profile=profile)
    assert "<directives>" not in system


def test_system_prompt_no_closing_tags():
    profile = _fake_profile("some directive")
    system = _compose_system(profile=profile)
    assert "</" not in system
