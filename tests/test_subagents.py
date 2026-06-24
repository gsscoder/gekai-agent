from __future__ import annotations

import pytest

from agent import subagents as subagents_module
from agent.persona import _SHARED_BODY
from agent.subagents import NAMESPACE_COLORS, NAMESPACES, Subagent, validate_registry
from agent.subagents.worker.ws_manager import subagent as ws_manager_subagent
from agent.tools.catalog import SHELL_TOOLS


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


# ---------------------------------------------------------------------------
# Namespace badge colors — co-located with the namespace, no fallback at
# render time: a namespace cannot exist without a color
# ---------------------------------------------------------------------------

def test_namespaces_derive_from_namespace_colors():
    assert NAMESPACES == tuple(NAMESPACE_COLORS)


def test_every_declared_namespace_has_a_non_empty_color():
    for ns in NAMESPACES:
        assert NAMESPACE_COLORS.get(ns)


def test_validate_registry_raises_when_a_namespace_has_no_color(monkeypatch):
    monkeypatch.setattr(subagents_module, "NAMESPACES", (*NAMESPACES, "ghost"))
    monkeypatch.setattr(subagents_module, "NAMESPACE_COLORS", {**NAMESPACE_COLORS, "ghost": ""})
    with pytest.raises(ValueError, match="ghost"):
        validate_registry()


# ---------------------------------------------------------------------------
# ws-manager — the system-managed worker/onboard subagent (plan 17b context):
# user-invocable so the router/menu can surface it, but it must never gain
# shell access, and it must satisfy the worker namespace's fallback invariant.
# ---------------------------------------------------------------------------

def test_ws_manager_is_user_invocable():
    assert ws_manager_subagent.user_invocable is True


def test_ws_manager_is_the_worker_namespace_fallback():
    assert ws_manager_subagent.is_fallback is True


def test_ws_manager_namespace_is_worker():
    assert ws_manager_subagent.namespace == "worker"


def test_ws_manager_has_no_shell_tools():
    # scaffolding-only specialist — never gets run_command (or any shell tool)
    assert ws_manager_subagent.tools is not None
    for shell_tool in SHELL_TOOLS:
        assert shell_tool not in ws_manager_subagent.tools


def test_ws_manager_is_discoverable_and_registry_stays_valid():
    # ws-manager is picked up by _discover() and does not break the
    # "exactly one fallback per namespace with invocable members" invariant
    # for the real, unmodified registry.
    assert ws_manager_subagent in subagents_module.SUBAGENTS
    validate_registry()  # must not raise
