from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from agent.pipeline.verifier import _VERIFIER_PROMPT, Verdict, Verifier
from tests.conftest import ONESHOT_TIER, mock_llm_response, run


def _make_verifier() -> Verifier:
    return Verifier(ONESHOT_TIER)


def _verify(verifier: Verifier) -> Verdict:
    return run(
        verifier.verify(
            user_input="add a widget endpoint",
            step_instruction="add a POST /widgets endpoint",
            diff_text="+def create_widget(): ...",
            output="added the endpoint",
        )
    )


def test_pass_response_yields_ok_verdict(llm_create) -> None:
    verifier = _make_verifier()
    raw = json.dumps({"verdict": "pass", "violations": []})
    llm_create.return_value = mock_llm_response(raw)

    verdict = _verify(verifier)

    assert verdict == Verdict(ok=True, violations=[])


def test_fail_response_yields_violations(llm_create) -> None:
    verifier = _make_verifier()
    raw = json.dumps({"verdict": "fail", "violations": ["field renamed on one side only"]})
    llm_create.return_value = mock_llm_response(raw)

    verdict = _verify(verifier)

    assert verdict == Verdict(ok=False, violations=["field renamed on one side only"])


def test_malformed_json_fails_open(llm_create) -> None:
    verifier = _make_verifier()
    llm_create.return_value = mock_llm_response("not json at all")

    verdict = _verify(verifier)

    assert verdict == Verdict(ok=True, violations=[])


def test_client_exception_fails_open(llm_create) -> None:
    verifier = _make_verifier()
    llm_create.side_effect = RuntimeError("network error")

    verdict = _verify(verifier)

    assert verdict == Verdict(ok=True, violations=[])


def test_fail_verdict_with_malformed_violations_key_does_not_crash(llm_create) -> None:
    verifier = _make_verifier()
    raw = json.dumps({"verdict": "fail", "violations": "not a list"})
    llm_create.return_value = mock_llm_response(raw)

    verdict = _verify(verifier)

    assert verdict.ok is False
    assert verdict.violations == []


def test_prompt_includes_lifecycle_invalidation_check(llm_create) -> None:
    assert "lifecycle-invalidation gap" in _VERIFIER_PROMPT


def test_fail_verdict_with_missing_violations_key_does_not_crash(llm_create) -> None:
    verifier = _make_verifier()
    raw = json.dumps({"verdict": "fail"})
    llm_create.return_value = mock_llm_response(raw)

    verdict = _verify(verifier)

    assert verdict.ok is False
    assert verdict.violations == []
