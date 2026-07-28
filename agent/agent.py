from __future__ import annotations

import platform
from collections.abc import AsyncIterator
from pathlib import Path

from . import __version__
from .llm.resolve import ResolvedTier, TierResolutionError, resolve_tier, resolve_touchpoint
from .llm.tiers import TierName
from .harness import Harness, HiddenGrantCallback
from .harness.touchpoints import touchpoint
from .permissions import PermissionCallback
from .pipeline import Gate, Route
from .session import Session
from .settings import Permissions, load_model_catalog, load_tier_bindings
from .logging import EventLogger

from .events import MaxIterationsEvent, AgentEvent
from .persistence import append_message, append_debug, append_event


class GekaiAgent:
    def __init__(self, *, working_dir: Path, permissions: Permissions, debug: bool = False) -> None:
        self.working_dir = working_dir
        self.permissions = permissions
        self.debug = debug

        # Resolution must NOT raise here: this constructor runs in
        # agent/main.py before the TUI (and so before `/tiers`) exists —
        # raising would permanently lock an unconfigured install out of the
        # only place that can fix it. `_configure_touchpoints` stashes
        # `self._tier_error` instead and `gate()`/`process_stream()` retry it
        # lazily on every call while still unconfigured (not just once — see
        # `_configure_touchpoints`'s own docstring for why a single attempt
        # isn't enough), raising it once there's a live chat (`_stream`'s
        # existing try/except) to show it in. The startup/per-prompt nudge
        # (`_maybe_warn_tiers_unconfigured`) covers the "haven't configured
        # yet" case before the user even tries to chat.
        self._tier_error: str | None = None
        self._gate: Gate | None = None
        self._main: Harness | None = None
        self.model: str = "unconfigured"
        self.effort: str | None = None
        self._api_key: str | None = None
        self._api_base: str | None = None
        gate_model, estimator_model, sequencer_model, main_dispatch_model, subagent_dispatch_model, responder_model = (
            self._configure_touchpoints()
        )
        self.events = EventLogger()
        self.events.emit(
            "run.start",
            version=__version__,
            platform=platform.system(),
            gate_model=gate_model,
            estimator_model=estimator_model,
            sequencer_model=sequencer_model,
            main_dispatch_model=main_dispatch_model,
            subagent_dispatch_model=subagent_dispatch_model,
            responder_model=responder_model,
            tiers_configured=self._tier_error is None,
            permissions={"read": permissions.read, "write": permissions.write, "exec": permissions.exec},
            debug=self.debug,
        )

    def _configure_touchpoints(self) -> tuple[str | None, str | None, str | None, str | None, str | None, str | None]:
        """(Re)resolve every touchpoint against the *current* on-disk tier
        catalog+bindings, updating `self._gate`/`self._main`/`self.model`/
        `self._api_key`/`self._api_base` in place. A single resolve-once-at-
        construction attempt isn't enough: `/tiers` runs inside the same
        already-constructed `GekaiAgent` and only touches disk, so without a
        retry here every touchpoint stays permanently stuck on whatever
        failed at process startup — the exact bug this fixes (config saved
        mid-session, next prompt still reports the pre-`/tiers` error).
        Called once at construction and again lazily from `gate()`/
        `process_stream()` on every call while `self._tier_error` is set.
        Returns the six touchpoints' resolved model names (or all-`None` on
        failure) purely for the `run.start` telemetry emit.
        """
        catalog = load_model_catalog()
        bindings = load_tier_bindings()
        resolved: dict[str, ResolvedTier] = {}
        try:
            for name in ("gate", "estimator", "sequencer", "main-dispatch", "subagent-dispatch", "responder"):
                resolved[name] = resolve_touchpoint(name, catalog, bindings)
        except TierResolutionError:
            # The specific failure (which tier, why) is deliberately not
            # surfaced here: it's an artifact of touchpoint iteration order,
            # not a meaningful "this is the one broken thing" signal (e.g.
            # "tier 'fast' is not configured" reads as "you never configured
            # this" even when FAST *is* bound and it's actually SUPP that's
            # missing a stored credential). A `/tiers` grid UI shows per-tier
            # detail in its own status column instead.
            self._tier_error = "tier configuration is incomplete — run /tiers"
            return (None, None, None, None, None, None)

        self._tier_error = None
        sequencer_cfg = resolved["sequencer"]
        self.model = sequencer_cfg.model
        self.effort = bindings[TierName.CORE].default_effort  # sequencer's nominal tier is CORE
        self._api_key = sequencer_cfg.api_key
        self._api_base = sequencer_cfg.api_base
        # gate runs on FAST — intent classification is pattern-matching, not
        # reasoning, and is the high-volume common path (one cheap call per turn)
        self._gate = Gate(
            model=resolved["gate"].model,
            api_key=resolved["gate"].api_key,
            api_base=resolved["gate"].api_base,
        )

        # The 3 scaled touchpoints (plan 28 Phase 2) don't get a frozen
        # ResolvedTier baked into the Harness — they get this resolver
        # closure plus each touchpoint's TierPolicy, so the harness can
        # re-resolve at a scaled tier per dispatch instead of the one
        # resolved above at `policy.default` (kept only for `run.start`
        # telemetry, below). Rebuilt fresh every call so a `/tiers` save
        # mid-session is picked up the same way the frozen values used to be.
        def _resolve(tier: TierName) -> ResolvedTier:
            return resolve_tier(tier, catalog, bindings)

        self._main = Harness(
            resolve=_resolve,
            sequencer_policy=touchpoint("sequencer").policy,
            main_dispatch_policy=touchpoint("main-dispatch").policy,
            subagent_dispatch_policy=touchpoint("subagent-dispatch").policy,
            estimator=resolved["estimator"],
            responder=resolved["responder"],
            debug=self.debug,
        )
        return (
            resolved["gate"].model,
            resolved["estimator"].model,
            sequencer_cfg.model,
            resolved["main-dispatch"].model,
            resolved["subagent-dispatch"].model,
            resolved["responder"].model,
        )

    def start_session(
        self,
        restored_messages: list[dict] | None = None,
        session_id: str | None = None,
    ) -> Session:
        session = Session(working_dir=self.working_dir, permissions=self.permissions)
        if session_id:
            session.id = session_id
        if self.debug:
            append_debug(session, session.messages[0])
        if restored_messages:
            session.messages.extend(restored_messages)
        return session

    async def gate(self, user_input: str, history: list[dict] | None = None) -> Route:
        if self._gate is None:
            self._configure_touchpoints()
        if self._gate is None:
            raise RuntimeError(self._tier_error or "tiers are not configured — run /tiers")
        return await self._gate.gate(user_input, history=history)

    async def process_stream(
        self,
        session: Session,
        user_input: str,
        route: Route,
        permission_callback: PermissionCallback | None = None,
        turn_id: str | None = None,
        hidden_grant_callback: HiddenGrantCallback | None = None,
        append_user: bool = True,
        seed: str | None = None,
    ) -> AsyncIterator[str | AgentEvent]:
        if self._main is None:
            self._configure_touchpoints()
        if self._main is None:
            raise RuntimeError(self._tier_error or "tiers are not configured — run /tiers")
        if append_user:
            session.messages.append({"role": "user", "content": user_input})
            append_message(session, session.messages[-1], turn=turn_id)

        all_chunks: list[str] = []
        max_iter_hit = False
        completed = False
        stream_iter = self._main.stream(
            session, user_input,
            permission_callback=permission_callback,
            extra_params={} if route.trivial else None,
            hidden_grant_callback=hidden_grant_callback,
            seed=seed,
        )
        try:
            async for item in stream_iter:
                if isinstance(item, str):
                    all_chunks.append(item)
                elif isinstance(item, MaxIterationsEvent):
                    max_iter_hit = True
                yield item

            if max_iter_hit and not all_chunks:
                append_event(session, "agent hit iteration limit without producing a response", source="max_iterations")
            else:
                session.messages.append({"role": "assistant", "content": "".join(all_chunks)})
                append_message(session, session.messages[-1], turn=turn_id)
            completed = True
        finally:
            if not completed and session.messages and session.messages[-1].get("role") == "user":
                session.messages.pop()
