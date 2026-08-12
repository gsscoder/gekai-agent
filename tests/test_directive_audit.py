"""Coverage for `agent/directive_audit.py` (plan 35 v3): the auditor's
one-shot call/parse shape (mirroring `test_estimate.py`) plus the
`file_sha`-only cache.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from agent.directive_audit import (
    Auditor,
    AuditVerdict,
    file_sha,
    load_cached_verdict,
    save_cached_verdict,
)
from tests.conftest import mock_llm_response, run


def _make_auditor() -> Auditor:
    with patch("agent.directive_audit.AsyncOpenAI"):
        return Auditor(model="test-model", api_key="key", api_base="http://localhost")


# ---------------------------------------------------------------------------
# parsing — YES, NO, and malformed output
# ---------------------------------------------------------------------------

def test_audit_yes() -> None:
    a = _make_auditor()
    a._client.chat.completions.create = AsyncMock(return_value=mock_llm_response("YES"))
    result = run(a.audit("# Agent Instructions\nyou must always run tests before committing"))
    assert result == AuditVerdict(has_directives=True, raw="YES")


def test_audit_no() -> None:
    a = _make_auditor()
    a._client.chat.completions.create = AsyncMock(return_value=mock_llm_response("NO"))
    result = run(a.audit("# Changelog\n..."))
    assert result == AuditVerdict(has_directives=False, raw="NO")


def test_audit_lowercase_is_accepted() -> None:
    a = _make_auditor()
    a._client.chat.completions.create = AsyncMock(return_value=mock_llm_response("yes"))
    result = run(a.audit("file text"))
    assert result.has_directives is True


def test_audit_malformed_output_falls_back_to_no() -> None:
    a = _make_auditor()
    a._client.chat.completions.create = AsyncMock(
        return_value=mock_llm_response("uh, this file looks fine I guess"),
    )
    result = run(a.audit("file text"))
    assert result.has_directives is False


def test_audit_exception_falls_back_to_no() -> None:
    a = _make_auditor()
    a._client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    result = run(a.audit("file text"))
    assert result.has_directives is False


def test_audit_system_prompt_carries_the_question_and_file_is_the_user_message() -> None:
    a = _make_auditor()
    mock_create = AsyncMock(return_value=mock_llm_response("NO"))
    a._client.chat.completions.create = mock_create

    run(a.audit("my file text"))

    messages = mock_create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "YES or NO" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "my file text"}
    assert mock_create.call_args.kwargs["temperature"] == 0


# ---------------------------------------------------------------------------
# cache — keyed by file_sha alone
# ---------------------------------------------------------------------------

def test_cache_miss_when_no_cache_file(tmp_path) -> None:
    assert load_cached_verdict(tmp_path, "GEKAI.md", "abc123") is None


def test_cache_hit_when_file_sha_matches(tmp_path) -> None:
    verdict = AuditVerdict(has_directives=True, raw="YES")
    save_cached_verdict(tmp_path, "GEKAI.md", "filesha1", verdict)

    hit = load_cached_verdict(tmp_path, "GEKAI.md", "filesha1")
    assert hit == verdict


def test_cache_miss_when_file_sha_differs(tmp_path) -> None:
    verdict = AuditVerdict(has_directives=False, raw="NO")
    save_cached_verdict(tmp_path, "GEKAI.md", "filesha1", verdict)

    miss = load_cached_verdict(tmp_path, "GEKAI.md", "filesha-changed")
    assert miss is None


def test_cache_keyed_by_rel_path(tmp_path) -> None:
    save_cached_verdict(tmp_path, "GEKAI.md", "sha1", AuditVerdict(has_directives=False, raw="NO"))
    assert load_cached_verdict(tmp_path, "AGENTS.md", "sha1") is None


def test_file_sha_stable_and_content_sensitive() -> None:
    assert file_sha("hello") == file_sha("hello")
    assert file_sha("hello") != file_sha("hello!")
