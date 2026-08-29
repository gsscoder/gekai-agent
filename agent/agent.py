from __future__ import annotations

import asyncio
import logging
import platform
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from . import __version__
from .directive_audit import Auditor, AuditVerdict, file_sha, load_cached_verdict, save_cached_verdict
from .llm.resolve import ResolvedTier, TierResolutionError, resolve_tier, resolve_touchpoint
from .llm.tiers import TierName
from .harness import Harness, HiddenGrantCallback
from .harness.touchpoints import touchpoint
from .permissions import PermissionCallback
from .session import IngestedFile, Session
from .settings import Permissions, load_directive_audit_enabled, load_model_catalog, load_tier_bindings
from .logging import EventLogger

from .events import DirectiveAuditEvent, MaxIterationsEvent, AgentEvent
from .persistence import append_message, append_debug, append_event

_log = logging.getLogger(__name__)

# Called with (rel_path, verdict) whenever the directive audit's TUI-visible
# state changes: `verdict=None` means "in flight" (a cache miss just started
# the LLM call), any `AuditVerdict` means the audit finished (cache hit or a
# real call) and is the final word until the next audit replaces it.
DirectiveVerdictCallback = Callable[[str, "AuditVerdict | None"], None]


class GekaiAgent:
    def __init__(self, *, working_dir: Path, permissions: Permissions, debug: bool = False) -> None:
        self.working_dir = working_dir
        self.permissions = permissions
        self.debug = debug

        # Resolution must NOT raise here: this constructor runs in
        # agent/main.py before the TUI (and so before `/models`/`/tier`) exists —
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
        self._root: Harness | None = None
        self.model: str = "unconfigured"
        self.effort: str | None = None
        self._api_key: str | None = None
        self._api_base: str | None = None
        # Strong references to fire-and-forget background tasks (plan 35
        # concept 5's directive audit is the first one) — asyncio only holds
        # a task alive via a live reference elsewhere; without this set a
        # task can be garbage-collected mid-flight and silently vanish.
        self._background_tasks: set[asyncio.Task] = set()
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
        catalog+bindings, updating `self._root`/`self.model`/
        `self._api_key`/`self._api_base` in place. A single resolve-once-at-
        construction attempt isn't enough: `/models`/`/tier` run inside the same
        already-constructed `GekaiAgent` and only touches disk, so without a
        retry here every touchpoint stays permanently stuck on whatever
        failed at process startup — the exact bug this fixes (config saved
        mid-session, next prompt still reports the pre-command error).
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
            # missing a stored credential). `/tier` with no args shows per-tier
            # detail in its own status column instead.
            self._tier_error = "tier configuration is incomplete — run /models, then /tier"
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
        # telemetry, below). Rebuilt fresh every call so a `/tier` save
        # mid-session is picked up the same way the frozen values used to be.
        # The touchpoint name rides along so a touchpoint's own operating
        # point (effort/thinking) still applies at whatever tier `scale()`
        # picked — the tier alone no longer says who is being resolved.
        def _resolve(tier: TierName, touchpoint_name: str) -> ResolvedTier:
            return resolve_tier(tier, catalog, bindings, touchpoint_name)

        self._root = Harness(
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
        still *failing* (`self._root` is `None`) — a `/tier`
        commit that changes an already-working tier's model/effort/thinking
        would otherwise sit stale (including `self.model`/`self.effort`,
        which the TUI status bar reads directly) until the next process
        restart. Called by the TUI right after a `/tier` or `/models` commit."""
        self._configure_touchpoints()

    def start_session(
        self,
        restored_messages: list[dict] | None = None,
        session_id: str | None = None,
    ) -> Session:
        session = Session(working_dir=self.working_dir, permissions=self.permissions)
        if session_id:
            session.id = session_id
        session.gekai_md = self._read_gekai_md()
        if self.debug:
            append_debug(session, session.messages[0])
        if restored_messages:
            session.messages.extend(restored_messages)
        return session

    def _read_gekai_md(self) -> IngestedFile | None:
        """Auto-read of `GEKAI.md` at the workspace root (plan 35 decision
        6: filename auto-discovery stops here, nothing else is scanned).
        A read failure — missing file, permissions, bad encoding, anything
        — is telemetry and a skip, never a crash: a broken GEKAI.md must
        not prevent a session from starting."""
        path = self.working_dir / "GEKAI.md"
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            self.events.emit(
                "gekai_md.read_failed", level="warning",
                path=str(path), error_type=type(exc).__name__, message=str(exc),
            )
            return None
        return IngestedFile(rel_path="GEKAI.md", text=text, sha=file_sha(text))

    def start_directive_audit(
        self, session: Session, on_verdict: DirectiveVerdictCallback | None = None,
    ) -> None:
        """Fires plan 35 Phase 2's directive audit for `session.gekai_md`, if
        any. Deliberately a separate call from `start_session()` rather than
        folded into it: `start_session()` must keep working for callers with
        no running event loop (e.g. `tests/test_gekai_md.py`'s plain sync
        tests), while firing a background task needs one. Both TUI call
        sites (`_init_session`, `_clear_session`) are already `async def`, so
        calling this immediately after `start_session()` there still counts
        as "after `_read_gekai_md()` sets `session.gekai_md`" — there is no
        await between the two, nothing else can observe the gap.

        Thin wrapper over `_start_audit`, fixed to `session.gekai_md` — the
        generalized entry point (plan 35 Phase 3's `start_foreign_file_audit`
        below shares the same cache/resolve/call/swallow flow)."""
        gekai_md = session.gekai_md
        if gekai_md is None:
            return
        self._start_audit(gekai_md, on_verdict)

    def start_foreign_file_audit(
        self, rel_path: str, text: str, on_verdict: DirectiveVerdictCallback | None = None,
    ) -> None:
        """Plan 35 Phase 3: the foreign-file counterpart of
        `start_directive_audit` above — same cache-check-then-fire, resolve,
        call, swallow-all-failures, and `DirectiveAuditEvent` telemetry,
        just built from a transient `IngestedFile` rather than
        `session.gekai_md`. Nothing is stored on `Session` for this (concept
        2 / decision 5 — no ingestion command, no per-file state): the
        `IngestedFile` built here exists only for the duration of this
        call, exactly like a foreign file read is a transient event in
        message history rather than accumulated session state."""
        ingested = IngestedFile(rel_path=rel_path, text=text, sha=file_sha(text))
        self._start_audit(ingested, on_verdict)

    def _start_audit(
        self, ingested: IngestedFile, on_verdict: DirectiveVerdictCallback | None,
    ) -> None:
        """The cache check happens synchronously, inline, before any task is
        created: it's a local file read, not a model call, so a hit can
        answer `on_verdict` immediately with no "in flight" flash and no
        task to track. Only a genuine miss pays for the background LLM call
        (concept 5 — never awaited, never blocks the caller)."""
        if not load_directive_audit_enabled(self.working_dir):
            return
        try:
            cached = load_cached_verdict(self.working_dir, ingested.rel_path, ingested.sha)
        except Exception:
            _log.warning("directive audit cache lookup failed; skipping", exc_info=True)
            return
        if cached is not None:
            self._emit_directive_audit_event(ingested, cached, cached=True, duration_ms=0)
            if on_verdict is not None:
                on_verdict(ingested.rel_path, cached)
            return

        if on_verdict is not None:
            on_verdict(ingested.rel_path, None)  # None = in flight
        task = asyncio.create_task(self._run_directive_audit(ingested, on_verdict))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _run_directive_audit(
        self, ingested: IngestedFile, on_verdict: DirectiveVerdictCallback | None,
    ) -> None:
        # Every failure here — bad/incomplete tier config, a network error,
        # anything the Auditor itself didn't already swallow — degrades to
        # the safe NO verdict, never a crash and never a stuck "in flight"
        # line (concept 5: the verdict has exactly one consumer, the human,
        # and a broken audit must never become the session's problem).
        t0 = time.monotonic()
        try:
            resolved = resolve_touchpoint("directive-audit", load_model_catalog(), load_tier_bindings())
            auditor = Auditor(
                model=resolved.model, api_key=resolved.api_key,
                api_base=resolved.api_base, extra_params=resolved.extra_params,
            )
            verdict = await auditor.audit(ingested.text)
        except Exception:
            _log.warning("directive audit failed; falling back to NO", exc_info=True)
            if on_verdict is not None:
                on_verdict(ingested.rel_path, AuditVerdict())
            return
        duration_ms = round((time.monotonic() - t0) * 1000)
        try:
            save_cached_verdict(self.working_dir, ingested.rel_path, ingested.sha, verdict)
        except Exception:
            _log.warning("directive audit cache save failed", exc_info=True)
        self._emit_directive_audit_event(ingested, verdict, cached=False, duration_ms=duration_ms)
        if on_verdict is not None:
            on_verdict(ingested.rel_path, verdict)

    def _emit_directive_audit_event(
        self, ingested: IngestedFile, verdict: AuditVerdict, *, cached: bool, duration_ms: int,
    ) -> None:
        event = DirectiveAuditEvent(
            path=ingested.rel_path, has_directives=verdict.has_directives,
            cached=cached, duration_ms=duration_ms,
            file_bytes=len(ingested.text.encode("utf-8")),
        )
        self.events.emit(
            "directive_audit",
            path=event.path, has_directives=event.has_directives,
            cached=event.cached, duration_ms=event.duration_ms,
            file_bytes=event.file_bytes,
        )

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
        if self._root is None:
            self._configure_touchpoints()
        if self._root is None:
            raise RuntimeError(self._tier_error or "tiers are not configured — run /models, then /tier")
        if append_user:
            session.messages.append({"role": "user", "content": user_input})
            append_message(session, session.messages[-1], turn=turn_id)

        all_chunks: list[str] = []
        max_iter_hit = False
        completed = False
        stream_iter = self._root.stream(
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
