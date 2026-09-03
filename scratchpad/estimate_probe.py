# Manual live-LLM probe — NOT a pytest test. Nondeterministic and costs real
# tokens; run by hand after touching agent/pipeline/estimate.py's prompt.
"""Runs the real Estimator (configured FAST tier) against a fixed set of
phrases and reports whether each one lands on the expected side of the
CHAT / non-CHAT line."""

from __future__ import annotations

import asyncio

from agent.pipeline.estimate import Estimator
from agent.tiers.catalog import TierName
from agent.tiers.resolve import resolve_tier
from agent.tiers.store import load_model_catalog, load_tier_bindings

CASES: list[tuple[str, bool]] = [
    # (phrase, expect_chat)
    ("hi", True),
    ("who are you", True),
    ("thanks", True),
    ("what is Python", True),
    ("briefly explain this repo", False),
    ("what does this app do", False),
    ("quick summary of the code", False),
    ("what's in main.py", False),
    ("in short, describe this codebase", False),
]


async def main() -> None:
    catalog = load_model_catalog()
    bindings = load_tier_bindings()
    tier = resolve_tier(TierName.FAST, catalog, bindings)
    estimator = Estimator(tier)

    results: list[tuple[str, bool, str, bool]] = []
    for phrase, expect_chat in CASES:
        estimate = await estimator.estimate(phrase)
        is_chat = estimate.scope == "chat"
        passed = is_chat == expect_chat
        results.append((phrase, expect_chat, estimate.scope, passed))
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {phrase!r} -> {estimate.scope} (expected chat={expect_chat})")

    failures = [r for r in results if not r[3]]
    print()
    if failures:
        print(f"SUMMARY: FAIL — {len(failures)}/{len(results)} misclassified")
    else:
        print(f"SUMMARY: PASS — all {len(results)} phrases classified as expected")


if __name__ == "__main__":
    asyncio.run(main())
