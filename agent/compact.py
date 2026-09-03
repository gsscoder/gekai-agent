"""Conversation-summarization core for `/compact` (Phase 1 of 6 — no TUI, no
persistence here; later phases wire this into the command and session
storage). Threshold math, the summarization prompt, and the two LLM/session
operations (`summarize`, `apply_summary`) live here so downstream phases have
a stable contract to build against."""

from __future__ import annotations

from typing import Literal

from .tiers.resolve import ResolvedTier
from .oneshot import complete
from .persona import ROOT_SYSTEM_PROMPT
from .session import Session

WARN_PCT = 0.75
AUTO_PCT = 0.80

SUMMARY_PREFIX = (
    "This session is being continued from a previous conversation that ran "
    "low on context. Here is the summary:\n\n"
)

_SUMMARIZE_PROMPT = (
    "you summarize a coding-agent conversation transcript so work can "
    "continue in a fresh context window with no loss of continuity\n"
    "produce a concise, structured summary with exactly these sections:\n\n"
    "## Primary Request/Intent\n"
    "the user's overall goal(s) for this session\n\n"
    "## Key Technical Concepts\n"
    "technologies, patterns, and design decisions in play\n\n"
    "## Files and Code Sections\n"
    "files touched and what changed in each, specific enough to resume "
    "editing without re-reading everything\n\n"
    "## Errors encountered and fixes\n"
    "problems hit and how they were resolved (or why they remain open)\n\n"
    "## Pending Tasks\n"
    "work requested but not yet done\n\n"
    "## Current Work\n"
    "exactly what was being worked on immediately before this summary\n\n"
    "## Next Step\n"
    "the single next action to take, directly continuing the current work\n\n"
    "be concise — this summary replaces the transcript, it does not "
    "document it"
)


def context_state(tokens: int, limit: int) -> Literal["ok", "warn", "auto"]:
    """Classify context usage against the warn/auto compaction thresholds."""
    ratio = tokens / limit
    if ratio >= AUTO_PCT:
        return "auto"
    if ratio >= WARN_PCT:
        return "warn"
    return "ok"


async def summarize(messages: list[dict], tier: ResolvedTier, instructions: str = "") -> str:
    """Summarize `messages` into a structured handoff via `tier`'s model.
    The original leading system message, if any, is dropped and replaced by
    the summarization prompt — the model summarizes the transcript, it does
    not continue it. Exceptions propagate; callers handle failure."""
    prompt = _SUMMARIZE_PROMPT
    if instructions:
        prompt += f"\n\nAdditional Instructions:\n{instructions}"

    transcript = messages[1:] if messages and messages[0]["role"] == "system" else messages

    return (await complete(
        tier,
        system=prompt,
        user="Produce the summary now.",
        context=transcript,
        temperature=0,
    )).strip()


def apply_summary(session: Session, summary: str) -> None:
    """Replace `session.messages` in place with a fresh system prompt and a
    single user message carrying `summary`, discarding the prior transcript."""
    session.messages[:] = [
        {"role": "system", "content": ROOT_SYSTEM_PROMPT},
        {"role": "user", "content": SUMMARY_PREFIX + summary},
    ]
