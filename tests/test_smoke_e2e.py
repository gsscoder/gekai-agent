"""End-to-end smoke test (plan 28 Phase 0 verification): confirms a real turn
survives the Plan->TaskGraph rename and the tui/app.py -> agent.harness.turn
orchestration extraction, start to answer, unchanged.

Drives the same sequence `GekaiApp._stream` drives (harness.turn.run_step,
which derives its no-graph/graph split from the Estimator's scope rung inside
`Harness.stream()`) against a real model, exercising both branches the
rename+extraction touched: the trivial single-agent path and the mutate
sequencer+interpreter path (which must actually produce a file on disk via
the interpreter's dispatch).

Real LLM calls — requires GEKAI_CORE_MODEL_NAME/KEY (+ SUPPORT). Run 3 trials
(plan 28 verification policy). Unlike test_planner_probe.py's majority-pass
threshold (that one measures model *quality*, which is inherently variable),
this measures mechanical pipeline *correctness* post-refactor, so every trial
must pass — any failure is a regression signal, not model variance.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from agent.agent import GekaiAgent
from agent.harness.turn import run_step
from agent.session import Session
from agent.settings import Permissions

_TRIALS = 3


def _make_agent(working_dir: Path) -> GekaiAgent:
    return GekaiAgent(
        working_dir=working_dir,
        permissions=Permissions(read=True, write=True, exec=True),
    )


async def _run_trial(working_dir: Path) -> str | None:
    """Returns None on success, else a failure reason."""
    agent = _make_agent(working_dir)
    session = Session(working_dir=working_dir, permissions=agent.permissions)

    # trivial path: estimator (inside Harness.stream) -> main solo
    trivial_result = await run_step(
        agent, session, "what is 7 plus 5? reply with just the number, nothing else.", None,
        turn_id="smoke-trivial", session_id=session.id,
        permission_callback=None, hidden_grant_callback=None,
    )
    if trivial_result.outcome != "ok":
        return f"trivial path outcome={trivial_result.outcome!r}"
    if not trivial_result.answer.strip():
        return "trivial path produced an empty answer"

    # mutate path: estimator:mutate -> sequencer -> interpreter -> main/subagent dispatch
    mutate_prompt = "create a file named smoke.txt in the workspace root containing exactly the text: smoke ok"
    mutate_result = await run_step(
        agent, session, mutate_prompt, None,
        turn_id="smoke-mutate", session_id=session.id,
        permission_callback=None, hidden_grant_callback=None,
    )
    if mutate_result.outcome != "ok":
        return f"mutate path outcome={mutate_result.outcome!r}"
    if not (working_dir / "smoke.txt").exists():
        return "mutate path did not create smoke.txt on disk"

    return None


@pytest.mark.llm
def test_turn_runs_start_to_answer_unchanged(tmp_path_factory: pytest.TempPathFactory) -> None:
    model_name = os.environ.get("GEKAI_CORE_MODEL_NAME")
    api_key = os.environ.get("GEKAI_CORE_MODEL_KEY")
    if not model_name or not api_key:
        pytest.skip("GEKAI_CORE_MODEL_NAME and GEKAI_CORE_MODEL_KEY must be set")

    failures: list[str] = []
    for trial in range(_TRIALS):
        working_dir = tmp_path_factory.mktemp(f"smoke_trial_{trial}")
        reason = asyncio.run(_run_trial(working_dir))
        if reason is not None:
            failures.append(f"trial {trial}: {reason}")

    assert not failures, f"{len(failures)}/{_TRIALS} trials failed: {failures}"
