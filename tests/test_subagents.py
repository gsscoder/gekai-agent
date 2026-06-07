from __future__ import annotations

from agent.persona import SYSTEM_PROMPT, TOOL_INSTRUCTION
from agent.subagents import Subagent


def _subagent(mandate: str = "", directives: str = "") -> Subagent:
    return Subagent(
        name="test-subagent",
        namespace="coding",
        description="test",
        mandate=mandate,
        directives=directives,
    )


def test_build_system_has_tools_block_always():
    system = _subagent().build_system()
    assert "<tools>\n" in system
    assert TOOL_INSTRUCTION in system


def test_build_system_starts_with_base_persona():
    system = _subagent(mandate="your specialization is X", directives="do Y").build_system()
    assert system.startswith(SYSTEM_PROMPT)


def test_build_system_has_mandate_block_when_present():
    system = _subagent(mandate="your specialization is refactors").build_system()
    assert "<core_mandate>\n" in system
    assert "your specialization is refactors" in system


def test_build_system_no_mandate_block_when_empty():
    system = _subagent(mandate="").build_system()
    assert "<core_mandate>" not in system


def test_build_system_has_directives_block_when_present():
    system = _subagent(directives="do not invent features").build_system()
    assert "<directives>\n" in system
    assert "do not invent features" in system


def test_build_system_no_directives_block_when_empty():
    system = _subagent(directives="").build_system()
    assert "<directives>" not in system


def test_build_system_block_order_mandate_then_directives_then_tools():
    system = _subagent(mandate="your specialization is X", directives="do Y").build_system()
    mandate_idx = system.index("<core_mandate>")
    directives_idx = system.index("<directives>")
    tools_idx = system.index("<tools>")
    assert mandate_idx < directives_idx < tools_idx


def test_build_system_no_closing_tags():
    system = _subagent(mandate="your specialization is X", directives="do Y").build_system()
    assert "</" not in system
