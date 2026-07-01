"""Coverage for `detect_subagent_mentions` in agent/harness/core.py.

This heuristic is a follow-up fix to a directive-only approach that failed
twice in production: a plain-English "if the user names an agent, delegate"
sentence in the system prompt was ignored because the model's own "does this
specialist actually fit" reasoning overrode it. Detecting the mention in code
and injecting a structured `<subagents_request>` block makes delegation
mandatory rather than a judgment call — this file tests the detection in
isolation, before the block is even built.

Roster (agent/subagents/__init__.py): code-expert, code-refactorer,
test-expert, test-fixer are user_invocable=True; ws-manager is
user_invocable=False (system-managed worker) and must never match, even when
the trigger verb + name pattern is present in the text.
"""

from __future__ import annotations

from agent.harness.core import detect_subagent_mentions


def test_use_trigger_matches() -> None:
    assert detect_subagent_mentions("use code-expert to fix this bug") == ["code-expert"]


def test_have_trigger_matches() -> None:
    assert detect_subagent_mentions("have test-fixer solve the tests") == ["test-fixer"]


def test_case_insensitive() -> None:
    assert detect_subagent_mentions("USE CODE-EXPERT now") == ["code-expert"]


def test_negation_guard_blocks_match() -> None:
    assert detect_subagent_mentions("don't use test-fixer for this") == []


def test_bare_mention_without_trigger_verb_does_not_match() -> None:
    assert detect_subagent_mentions("test-fixer is broken right now") == []


def test_multiple_mentions_returned() -> None:
    result = detect_subagent_mentions(
        "use code-expert for the fix and ask test-expert to write tests"
    )
    assert set(result) == {"code-expert", "test-expert"}


def test_non_invocable_agent_never_matches() -> None:
    assert detect_subagent_mentions("use ws-manager to reindex") == []


def test_no_mentions_returns_empty() -> None:
    assert detect_subagent_mentions("list the files in this repo") == []
