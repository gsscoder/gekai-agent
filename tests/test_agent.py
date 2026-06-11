from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import cast

from agent.agent import GekaiAgent
from agent.harness import FileLocator, Harness
from agent.persistence import session_file
from agent.pipeline import Route
from agent.session import Session
from agent.workspace import db as workspace_db


def run(coro):
    return asyncio.run(coro)


class _FakeMain:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.last_extra_params: dict | None | str = "unset"  # sentinel — distinguishes "not passed" from None

    def stream(self, session, user_input, permission_callback=None, subagent=None, extra_params=None):
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


# ---------------------------------------------------------------------------
# locate() consults the workspace.db cache and feeds it to FileLocator as hints
# ---------------------------------------------------------------------------

class _FakeLocator:
    def __init__(self, entries: list[tuple[str, list[str]]]) -> None:
        self._entries = entries
        self.last_hint_paths: list[str] | None = "unset"  # type: ignore[assignment]

    async def locate(self, working_dir, request, hint_paths=None):
        self.last_hint_paths = hint_paths
        return self._entries


def _stub_agent_with_locator(locator: _FakeLocator) -> GekaiAgent:
    stub = object.__new__(GekaiAgent)
    stub._locator = cast(FileLocator, locator)
    return stub


def test_locate_returns_entries_and_no_hints_when_cache_empty(tmp_path: Path) -> None:
    locator = _FakeLocator([("src/a.py", ["alpha"])])
    agent = _stub_agent_with_locator(locator)

    entries, hint_paths = run(agent.locate(tmp_path, "find alpha"))

    assert entries == [("src/a.py", ["alpha"])]
    assert hint_paths == []
    assert locator.last_hint_paths is None


def test_locate_passes_cached_candidates_as_hints(tmp_path: Path) -> None:
    f = tmp_path / "src" / "prompt_builder.py"
    f.parent.mkdir(parents=True)
    f.write_text("class PromptBuilder: ...")

    conn = workspace_db.ensure(tmp_path)
    workspace_db.save_findings(
        conn, tmp_path, [("src/prompt_builder.py", ["promptbuilder", "prompt_builder"])]
    )
    conn.close()

    locator = _FakeLocator([])
    agent = _stub_agent_with_locator(locator)

    entries, hint_paths = run(agent.locate(tmp_path, "update the prompt builder"))

    assert hint_paths == ["src/prompt_builder.py"]
    assert locator.last_hint_paths == hint_paths
    assert entries == []
