from __future__ import annotations

import pytest

import dataclasses

from agent import subagents as subagents_module
from agent.persona import _SHARED_BODY
from agent.subagents import (
    NAMESPACE_COLORS,
    NAMESPACES,
    Subagent,
    ToolPolicy,
    _compose_directives,
    validate_registry,
)
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


# ---------------------------------------------------------------------------
# _compose_directives — cross-domain directive composition (directive_domains)
# own namespace is always included unconditionally; directive_domains lists
# ADDITIONAL domains; "*" pulls in every other domain, ranked
# ---------------------------------------------------------------------------

def test_compose_directives_with_no_extra_domains_uses_own_namespace_and_own_directives_only():
    p = _subagent(directives="own mandate text")
    ns_directives = {"coding": "coding domain text", "testing": "testing domain text"}
    result = _compose_directives(p, ns_directives, {})
    assert result == "coding domain text\nown mandate text"
    assert "testing domain text" not in result


def test_compose_directives_with_explicit_extra_domains_preserves_array_order():
    p = dataclasses.replace(
        _subagent(directives="own mandate text"),
        directive_domains=("beta_domain", "alpha_domain"),
    )
    ns_directives = {
        "coding": "coding domain text",
        "alpha_domain": "alpha domain text",
        "beta_domain": "beta domain text",
    }
    result = _compose_directives(p, ns_directives, {})
    # array order (beta, then alpha) must be preserved, not alphabetically re-sorted
    assert result == "coding domain text\nbeta domain text\nalpha domain text\nown mandate text"


def test_compose_directives_star_pulls_every_other_domain_ordered_by_rank_not_alphabet():
    p = dataclasses.replace(_subagent(directives="own mandate text"), directive_domains=("*",))
    ns_directives = {
        "coding": "coding domain text",
        "alpha_domain": "alpha domain text",
        "zeta_domain": "zeta domain text",
    }
    # zeta_domain alphabetically last, but ranked first — proves ordering is
    # rank-driven, not alphabetical
    ns_rank = {"zeta_domain": 1, "alpha_domain": 2}
    result = _compose_directives(p, ns_directives, ns_rank)
    assert result == "coding domain text\nzeta domain text\nalpha domain text\nown mandate text"


def test_compose_directives_domain_with_no_entry_contributes_nothing_and_no_stray_whitespace():
    # simulates "generic": a domain that defines no namespace_directives
    p = dataclasses.replace(
        _subagent(directives="own mandate text"),
        directive_domains=("generic_domain",),
    )
    ns_directives = {"coding": "coding domain text"}
    result = _compose_directives(p, ns_directives, {})
    assert result == "coding domain text\nown mandate text"
    assert "\n\n" not in result


def test_validate_registry_raises_on_unknown_directive_domain(monkeypatch):
    bad = dataclasses.replace(_subagent(), directive_domains=("no_such_domain",))
    monkeypatch.setattr(subagents_module, "SUBAGENTS", [*subagents_module.SUBAGENTS, bad])
    with pytest.raises(ValueError, match="no_such_domain"):
        validate_registry()


def test_validate_registry_raises_when_star_mixed_with_explicit_domain(monkeypatch):
    bad = dataclasses.replace(_subagent(), directive_domains=("*", "testing"))
    monkeypatch.setattr(subagents_module, "SUBAGENTS", [*subagents_module.SUBAGENTS, bad])
    with pytest.raises(ValueError, match="directive_domains"):
        validate_registry()


# ---------------------------------------------------------------------------
# omni-worker (plan 32 Phase 1) — the generic namespace's first member;
# directive_domains=("*",) means its own namespace (generic) composes first,
# then every other domain ranked, ahead of its own mandate-level directives
# ---------------------------------------------------------------------------

def test_omni_worker_registered_in_generic_namespace():
    omni_worker = next(p for p in subagents_module.SUBAGENTS if p.name == "omni-worker")
    assert omni_worker.namespace == "generic"
    assert omni_worker.directive_domains == ("*",)


def test_omni_worker_directives_lead_with_generic_block_ahead_of_coding_and_testing():
    omni_worker = next(p for p in subagents_module.SUBAGENTS if p.name == "omni-worker")
    generic_idx = omni_worker.directives.index(subagents_module.NAMESPACE_DIRECTIVES["generic"])
    coding_idx = omni_worker.directives.index(subagents_module.NAMESPACE_DIRECTIVES["coding"])
    testing_idx = omni_worker.directives.index(subagents_module.NAMESPACE_DIRECTIVES["testing"])
    assert generic_idx == 0
    assert generic_idx < coding_idx < testing_idx


def test_validate_registry_passes_with_omni_worker_registered():
    # omni-worker is part of the real, discovered registry (not injected via
    # monkeypatch) — this asserts the actual startup state is valid
    validate_registry()


# ---------------------------------------------------------------------------
# Cold-dispatch specialists inherit the generic contract (plan 32 Phase 4) —
# code-expert, code-fixer, code-refactorer, test-expert, test-fixer each
# list "generic" in directive_domains so their composed directives carry
# the generic namespace's cold-run contract alongside their own craft directives
# (test-expert/test-fixer additionally list "coding" for its reuse-before-write trait)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    ["code-expert", "code-fixer", "code-refactorer", "test-expert", "test-fixer"],
)
def test_cold_dispatch_specialist_composes_generic_block(name):
    p = next(s for s in subagents_module.SUBAGENTS if s.name == name)
    assert "generic" in p.directive_domains
    assert subagents_module.NAMESPACE_DIRECTIVES["generic"] in p.directives


def test_code_refactorer_no_longer_instructs_asking_before_proceeding():
    # generic's cold contract says "never ask a clarifying question" — the
    # refactorer's own line on untraceable references must not contradict it
    p = next(s for s in subagents_module.SUBAGENTS if s.name == "code-refactorer")
    assert "ask before proceeding" not in p.directives
    assert "state the limitation as the blocker and stop" in p.directives
