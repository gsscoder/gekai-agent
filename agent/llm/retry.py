"""Retry policy and backoff helper for provider calls."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryAttempt:
    """Per-attempt observability payload passed to `RetryPolicy.on_retry`."""

    attempt: int
    delay: float
    exc: BaseException


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential-backoff retry policy for provider calls."""

    max_attempts: int = 3
    initial_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.2
    retry_on: tuple[type[BaseException], ...] = ()
    respect_retry_after: bool = True
    on_retry: Callable[[RetryAttempt], None] | None = None


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    value: Any = None
    if headers is not None:
        try:
            value = headers.get("retry-after")
        except Exception:
            value = None
    if value is None:
        value = getattr(exc, "retry_after", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_delay(policy: RetryPolicy, attempt: int, exc: BaseException) -> float:
    base = policy.initial_delay * (policy.multiplier ** (attempt - 1))
    base = min(base, policy.max_delay)
    if policy.jitter:
        spread = base * policy.jitter
        base = max(0.0, base + random.uniform(-spread, spread))
    if policy.respect_retry_after:
        hint = _retry_after_seconds(exc)
        if hint is not None:
            return min(policy.max_delay, max(base, hint))
    return base


async def retry_call(
    policy: RetryPolicy | None,
    factory: Callable[[], Awaitable[T]],
) -> T:
    if policy is None or policy.max_attempts <= 1 or not policy.retry_on:
        return await factory()

    last_exc: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await factory()
        except policy.retry_on as exc:
            last_exc = exc
            if attempt >= policy.max_attempts:
                raise
            delay = _compute_delay(policy, attempt, exc)
            if policy.on_retry is not None:
                policy.on_retry(RetryAttempt(attempt=attempt, delay=delay, exc=exc))
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc
