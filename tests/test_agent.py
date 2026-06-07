from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import cast

from agent.agent import GekaiAgent
from agent.handlers.main_agent import MainAgent
from agent.persistence import session_file
from agent.router import Route, Session


def run(coro):
    return asyncio.run(coro)


class _FakeMain:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def stream(self, session, user_input, permission_callback=None, subagent=None):
        async def _gen():
            for c in self._chunks:
                yield c
        return _gen()


def _make_session(tmp_path: Path) -> Session:
    return Session(id=str(uuid.uuid4()), working_dir=tmp_path)


def _stub_agent(chunks: list[str]) -> GekaiAgent:
    stub = object.__new__(GekaiAgent)
    stub._main = cast(MainAgent, _FakeMain(chunks))
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


def test_process_stream_persists_original_input_when_rewritten(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    agent = _stub_agent(["reply"])
    run(_drain(agent.process_stream(
        session, "rewritten phrasing", Route(), original_input="verbatim user text",
    )))

    user_turns = [t for t in _turns(session) if t["role"] == "user"]
    assert len(user_turns) == 1
    assert user_turns[0]["content"] == "verbatim user text"


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
