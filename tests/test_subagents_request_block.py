"""Tests for `_subagents_request_block`'s parametrized intro (plan 26
Improvement 3): the same block shape, two reasons — "named" (today's
explicit-mention sentence) and "implies" (the Improvement 1 scope-estimate
sentence) — and an unknown reason must raise.
"""

from __future__ import annotations

import pytest

from agent.harness.core import _subagents_request_block


def test_named_reason_contains_explicitly_named() -> None:
    block = _subagents_request_block(["code-expert"], reason="named")
    assert "explicitly named" in block
    assert "code-expert" in block


def test_implies_reason_contains_implies() -> None:
    block = _subagents_request_block(["code-expert"], reason="implies")
    assert "implies" in block
    assert "code-expert" in block


def test_both_reasons_contain_full_roster() -> None:
    names = ["code-expert", "test-expert"]
    named_block = _subagents_request_block(names, reason="named")
    implies_block = _subagents_request_block(names, reason="implies")
    for block in (named_block, implies_block):
        assert "code-expert" in block
        assert "test-expert" in block


def test_unknown_reason_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _subagents_request_block(["code-expert"], reason="bogus")
