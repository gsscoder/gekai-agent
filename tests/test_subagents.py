from __future__ import annotations

import pytest

import dataclasses

from agent import subagents as subagents_module
from agent.persona import _SHARED_BODY
from agent.subagents import NAMESPACE_COLORS, NAMESPACES, Subagent, ToolPolicy, validate_registry
from agent.tools.catalog import RUNGS


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
# Tool policy ceiling — validated both at construction (__post_init__) and
# at startup (validate_registry), matching its existing style
# ---------------------------------------------------------------------------

def test_validate_registry_rejects_out_of_range_ceiling(monkeypatch):
    # ToolPolicy.__post_init__ already forbids constructing an out-of-range
    # ceiling directly, so bypass __init__ to simulate a registry entry that
    # slipped past construction-time validation, matching what
    # validate_registry()'s redundant check is meant to catch.
    bad_policy = object.__new__(ToolPolicy)
    object.__setattr__(bad_policy, "ceiling", len(RUNGS))
    bad = dataclasses.replace(_subagent(), tool_policy=bad_policy)
    monkeypatch.setattr(subagents_module, "SUBAGENTS", [*subagents_module.SUBAGENTS, bad])
    with pytest.raises(ValueError, match="out-of-range tool policy ceiling"):
        validate_registry()
