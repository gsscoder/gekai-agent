"""Tests for ToolRegistry.run's per-tool timeout override.

Background: `Agent.tool_timeout` defaults to a flat 30s applied to every tool
call via `asyncio.wait_for` in `ToolRegistry.run`. That ceiling is sane for
fast tools (read_file) but wrong for tools that legitimately run long (e.g.
`delegate`, which runs an entire nested specialist agent to completion). This
covers the `Tool.timeout` field / `tool(..., timeout=...)` override that lets
a tool opt out of (or into a different) timeout than the registry default.
"""

from __future__ import annotations

import asyncio

import pytest

from agent.llm.tools import ToolRegistry, tool
from agent.llm.types import ToolUseBlock


def run(coro):
    return asyncio.run(coro)


async def _slow(seconds: float) -> str:
    await asyncio.sleep(seconds)
    return "done"


def _use(name: str = "slow") -> ToolUseBlock:
    return ToolUseBlock(id="call-1", name=name, input={"seconds": 0.2})


def test_unset_timeout_uses_registry_default() -> None:
    """A tool with no `timeout` set (the sentinel default) must still be
    cancelled by the registry-wide default passed to `.run(...)` — the
    sentinel must not accidentally disable timeouts for everyone."""
    registry = ToolRegistry()
    registry.register(tool(_slow, name="slow"))

    results = run(registry.run([_use()], timeout=0.05))

    assert len(results) == 1
    assert results[0].is_error is True
    assert "timed out" in results[0].content


def test_explicit_none_timeout_is_never_cancelled() -> None:
    """A tool with timeout=None is NOT cancelled even when it runs longer
    than the registry-wide default passed to `.run(...)`."""
    registry = ToolRegistry()
    registry.register(tool(_slow, name="slow", timeout=None))

    results = run(registry.run([_use()], timeout=0.05))

    assert len(results) == 1
    assert results[0].is_error is False
    assert results[0].content == "done"


def test_explicit_numeric_timeout_overrides_registry_default_smaller() -> None:
    """A tool with an explicit numeric timeout gets cancelled at ITS OWN
    value even when the registry-wide default is larger (proving the
    per-tool override wins, not just that None is special-cased)."""
    registry = ToolRegistry()
    registry.register(tool(_slow, name="slow", timeout=0.05))

    results = run(registry.run([_use()], timeout=5.0))

    assert len(results) == 1
    assert results[0].is_error is True
    assert "timed out after 0.05s" in results[0].content


def test_explicit_numeric_timeout_overrides_registry_default_larger() -> None:
    """A tool with an explicit numeric timeout that is LARGER than the
    registry-wide default is not cancelled early — proving the per-tool
    value, not the registry default, governs."""
    registry = ToolRegistry()
    registry.register(tool(_slow, name="slow", timeout=5.0))

    results = run(registry.run([_use()], timeout=0.05))

    assert len(results) == 1
    assert results[0].is_error is False
    assert results[0].content == "done"
