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
        # `self._tier_error` instead and `process_stream()` retries it
        # lazily on every call while still unconfigured (not just once — see
        # `_configure_touchpoints`'s own docstring for why a single attempt
        # isn't enough), raising it once there's a live chat (`_stream`'s
        # existing try/except) to show it in. The startup/per-prompt nudge
        # (`_maybe_warn_tiers_unconfigured`) covers the "haven't configured
        # yet" case before the user even tries to chat.
        self._tier_error: str | None = None
        self._main: Harness | None = None
        self.model: str = "unconfigured"
        self.effort: str | None = None
        self._api_key: str | None = None
        self._api_base: str | None = None
        estimator_model, sequencer_model, root_dispatch_model, subagent_dispatch_model = (
            self._configure_touchpoints()
        )
        self.events = EventLogger()
        self.events.emit(
            "run.start",
            version=__version__,
            platform=platform.system(),
            estimator_model=estimator_model,
            sequencer_model=sequencer_model,
            root_dispatch_model=root_dispatch_model,
            subagent_dispatch_model=subagent_dispatch_model,
            tiers_configured=self._tier_error is None,
            permissions={"read": permissions.read, "write": permissions.write, "exec": permissions.exec},
            debug=self.debug,
        )

    def _configure_touchpoints(self) -> tuple[str | None, str | None, str | None, str | None]:
        """(Re)resolve every touchpoint against the *current* on-disk tier
        catalog+bindings, updating `self._main`/`self.model`/
        `self._api_key`/`self._api_base` in place. A single resolve-once-at-
        construction attempt isn't enough: `/tiers` runs inside the same
        already-constructed `GekaiAgent` and only touches disk, so without a
        retry here every touchpoint stays permanently stuck on whatever
        failed at process startup — the exact bug this fixes (config saved
        mid-session, next prompt still reports the pre-`/tiers` error).
        Called once at construction and again lazily from
        `process_stream()` on every call while `self._tier_error` is set.
        Returns the four touchpoints' resolved model names (or all-`None` on
        failure) purely for the `run.start` telemetry emit.
        """
        catalog = load_model_catalog()
        bindings = load_tier_bindings()
        resolved: dict[str, ResolvedTier] = {}
        try:
            for name in ("estimator", "sequencer", "root-dispatch", "subagent-dispatch"):
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
            return (None, None, None, None)

        self._tier_error = None
        sequencer_cfg = resolved["sequencer"]
        self.model = sequencer_cfg.model
        self.effort = bindings[TierName.CORE].default_effort  # sequencer's nominal tier is CORE
        self._api_key = sequencer_cfg.api_key
        self._api_base = sequencer_cfg.api_base

        # The 3 scaled touchpoints (plan 28 Phase 2) don't get a frozen
        # ResolvedTier baked into the Harness — they get this resolver
        # closure plus each touchpoint's TierPolicy, so the harness can
        # re-resolve at a scaled tier per dispatch instead of the one
        # resolved above at `policy.default` (kept only for `run.start`
        # telemetry, below). Rebuilt fresh every call so a `/tiers` save
        # mid-session is picked up the same way the frozen values used to be.
        # The touchpoint name rides along so a touchpoint's own operating
        # point (effort/thinking) still applies at whatever tier `scale()`
        # picked — the tier alone no longer says who is being resolved.
        def _resolve(tier: TierName, touchpoint_name: str) -> ResolvedTier:
            return resolve_tier(tier, catalog, bindings, touchpoint_name)

        self._main = Harness(
            resolve=_resolve,
            sequencer_policy=touchpoint("sequencer").policy,
            root_dispatch_policy=touchpoint("root-dispatch").policy,
            subagent_dispatch_policy=touchpoint("subagent-dispatch").policy,
            estimator=resolved["estimator"],
            debug=self.debug,
        )
        return (
            resolved["estimator"].model,
            sequencer_cfg.model,
            resolved["root-dispatch"].model,
            resolved["subagent-dispatch"].model,
        )

    def reconfigure_touchpoints(self) -> None:
        """Force an immediate re-resolve of every touchpoint against the
        current on-disk tier catalog+bindings. `process_stream()`
        only retries `_configure_touchpoints()` lazily while resolution is
        still *failing* (`self._main` is `None`) — a `/tiers`
        commit that changes an already-working tier's model/effort/thinking
        would otherwise sit stale (including `self.model`/`self.effort`,
        which the TUI status bar reads directly) until the next process
        restart. Called by the TUI right after a `/tiers` commit."""
        self._configure_touchpoints()

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

    async def process_stream(
        self,
        session: Session,
        user_input: str,
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
