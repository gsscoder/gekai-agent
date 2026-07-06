from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import cast

from agent.agent import GekaiAgent
from agent.harness import Harness
from agent.persistence import session_file
from agent.pipeline import Route
from agent.session import Session


def run(coro):
    return asyncio.run(coro)


class _FakeMain:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.last_extra_params: dict | None | str = "unset"  # sentinel — distinguishes "not passed" from None

    def stream(self, session, user_input, permission_callback=None, extra_params=None, hidden_grant_callback=None, seed=None):
        self.last_extra_params = extra_params

        async def _gen():
            for c in self._chunks:
                yield c
        return _gen()



def _make_session(tmp_path: Path) -> Session:
    return Session(id=str(uuid.uuid4()), working_dir=tmp_path)


def _stub_agent(chunks: list[str]) -> GekaiAgent:
    stub = object.__new__(GekaiAgent)
    stub._main = cast(Harness, _FakeMain(chunks))
    return stub



async def _drain(agen) -> None:
    async for _ in agen:
        pass


def _turns(session: Session) -> list[dict]:
    entries = [json.loads(line) for line in session_file(session).read_text().splitlines()]
    return [e for e in entries if e["kind"] == "turn"]


# ---------------------------------------------------------------------------
# process_stream owns the user-turn persist (sole owner — no double-write)
# ---------------------------------------------------------------------------

def test_process_stream_persists_user_turn_once(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    agent = _stub_agent(["reply"])
    run(_drain(agent.process_stream(session, "hi", Route())))

    user_turns = [t for t in _turns(session) if t["role"] == "user"]
    assert len(user_turns) == 1
    assert user_turns[0]["content"] == "hi"


def test_process_stream_persists_user_input_when_not_rewritten(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    agent = _stub_agent(["reply"])
    run(_drain(agent.process_stream(session, "plain request", Route())))

    user_turns = [t for t in _turns(session) if t["role"] == "user"]
    assert user_turns[0]["content"] == "plain request"


def test_process_stream_writes_user_turn_before_assistant(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    agent = _stub_agent(["reply"])
    run(_drain(agent.process_stream(session, "question", Route())))

    roles = [t["role"] for t in _turns(session)]
    assert roles == ["user", "assistant"]


# ---------------------------------------------------------------------------
# append_user=False (plan steps 2..N) skips the user-turn persist, keeps the
# assistant-turn persist
# ---------------------------------------------------------------------------

def test_process_stream_append_user_false_skips_user_turn(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    agent = _stub_agent(["reply"])
    run(_drain(agent.process_stream(session, "step 2 raw", Route(), append_user=False)))

    turns = _turns(session)
    assert [t["role"] for t in turns] == ["assistant"]


def test_process_stream_append_user_true_matches_default(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    agent = _stub_agent(["reply"])
    run(_drain(agent.process_stream(session, "hi", Route(), append_user=True)))

    roles = [t["role"] for t in _turns(session)]
    assert roles == ["user", "assistant"]


def test_process_stream_plan_two_steps_persist_one_user_two_assistant(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    turn_id = "shared-turn"

    agent_step1 = _stub_agent(["step1 reply"])
    run(_drain(agent_step1.process_stream(session, "original prompt", Route(), turn_id=turn_id)))

    agent_step2 = _stub_agent(["step2 reply"])
    run(_drain(agent_step2.process_stream(session, "step 2 raw", Route(), turn_id=turn_id, append_user=False)))

    turns = _turns(session)
    assert [t["role"] for t in turns] == ["user", "assistant", "assistant"]
    assert turns[0]["content"] == "original prompt"
    assert turns[1]["content"] == "step1 reply"
    assert turns[2]["content"] == "step2 reply"


# ---------------------------------------------------------------------------
# trivial routes run main with empty extra_params (no thinking)
# ---------------------------------------------------------------------------

def test_process_stream_trivial_route_passes_empty_extra_params(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    agent = _stub_agent(["reply"])
    run(_drain(agent.process_stream(session, "hi", Route(trivial=True))))

    main = cast(_FakeMain, agent._main)
    assert main.last_extra_params == {}


def test_process_stream_non_trivial_route_passes_no_extra_params_override(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    agent = _stub_agent(["reply"])
    run(_drain(agent.process_stream(session, "question", Route())))

    main = cast(_FakeMain, agent._main)
    assert main.last_extra_params is None
