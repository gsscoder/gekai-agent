from __future__ import annotations

from agent.persona import SYSTEM_PROMPT
from agent.subagents import Subagent


def _subagent(mandate: str = "", directives: str = "") -> Subagent:
    return Subagent(
        name="test-subagent",
        namespace="coding",
        description="test",
        mandate=mandate,
        directives=directives,
    )


def test_build_system_base_has_no_tools_block():
    # <tools> is appended by the harness once the effective tool set is known
    system = _subagent().build_system_base()
    assert "<tools>" not in system


def test_build_system_base_starts_with_base_persona():
    system = _subagent(mandate="your specialization is X", directives="do Y").build_system_base()
    assert system.startswith(SYSTEM_PROMPT)


def test_build_system_base_has_mandate_block_when_present():
    system = _subagent(mandate="your specialization is refactors").build_system_base()
    assert "<core_mandate>\n" in system
    assert "your specialization is refactors" in system


def test_build_system_base_no_mandate_block_when_empty():
    system = _subagent(mandate="").build_system_base()
    assert "<core_mandate>" not in system


def test_build_system_base_has_directives_block_when_present():
    system = _subagent(directives="do not invent features").build_system_base()
    assert "<directives>\n" in system
    assert "do not invent features" in system


def test_build_system_base_no_directives_block_when_empty():
    system = _subagent(directives="").build_system_base()
    assert "<directives>" not in system


def test_build_system_base_block_order_mandate_then_directives():
    system = _subagent(mandate="your specialization is X", directives="do Y").build_system_base()
    mandate_idx = system.index("<core_mandate>")
    directives_idx = system.index("<directives>")
    assert mandate_idx < directives_idx


def test_build_system_base_no_closing_tags():
    system = _subagent(mandate="your specialization is X", directives="do Y").build_system_base()
    assert "</" not in system
