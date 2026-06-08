from __future__ import annotations

from agent.persona import _SHARED_BODY
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


def test_build_system_base_opens_with_member_identity():
    # a subagent is a scoped role played within Gekai, not Gekai itself —
    # it must never see the main agent's top-level "you are Gekai" assertion
    system = _subagent(mandate="you act as X", directives="do Y").build_system_base()
    assert system.startswith("you are part of Gekai")
    assert "you are Gekai" not in system


def test_build_system_base_contains_shared_body():
    system = _subagent(mandate="you act as X", directives="do Y").build_system_base()
    assert _SHARED_BODY in system


def test_build_system_base_includes_role_line_when_mandate_present():
    system = _subagent(mandate="you act as a refactoring specialist").build_system_base()
    assert "<core_mandate>" not in system
    assert "you act as a refactoring specialist" in system


def test_build_system_base_no_role_line_when_mandate_empty():
    system = _subagent(mandate="").build_system_base()
    assert "<core_mandate>" not in system
    assert "you act as" not in system


def test_build_system_base_has_directives_block_when_present():
    system = _subagent(directives="do not invent features").build_system_base()
    assert "<directives>\n" in system
    assert "do not invent features" in system


def test_build_system_base_no_directives_block_when_empty():
    system = _subagent(directives="").build_system_base()
    assert "<directives>" not in system


def test_build_system_base_order_identity_role_body_directives():
    system = _subagent(mandate="you act as X", directives="do Y").build_system_base()
    identity_idx = system.index("you are part of Gekai")
    role_idx = system.index("you act as X")
    body_idx = system.index(_SHARED_BODY)
    directives_idx = system.index("<directives>")
    assert identity_idx < role_idx < body_idx < directives_idx


def test_build_system_base_no_closing_tags():
    system = _subagent(mandate="you act as X", directives="do Y").build_system_base()
    assert "</" not in system
