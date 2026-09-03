"""One-shot model calls: the harness's non-agentic touchpoints (estimator,
sequencer, verifier, directive auditor, `/compact`) all want the same thing —
a single chat completion from a resolved tier, with the same connect/write/
pool timeout policy and only the read timeout varying by how long the call is
expected to take.

Each caller keeps its own prompt and its own parsing; everything below the
prompt is here, once. Callers pass a `ResolvedTier` whole rather than
unpacking it into four loose parameters.
"""

from __future__ import annotations

from functools import lru_cache

import httpx
from openai import AsyncOpenAI

from .tiers.resolve import ResolvedTier

_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 30.0
_WRITE_TIMEOUT = 30.0
_POOL_TIMEOUT = 30.0


@lru_cache(maxsize=None)
def build_client(
    api_key: str | None,
    api_base: str | None,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
) -> AsyncOpenAI:
    """Cached per (credential, endpoint, read timeout): a client owns an
    httpx connection pool, so building a fresh one per call would leak a pool
    on every turn."""
    return AsyncOpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=httpx.Timeout(
            connect=_CONNECT_TIMEOUT,
            read=read_timeout,
            write=_WRITE_TIMEOUT,
            pool=_POOL_TIMEOUT,
        ),
    )


async def complete(
    tier: ResolvedTier,
    *,
    system: str,
    user: str,
    context: list[dict] | None = None,
    temperature: float | None = None,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
) -> str:
    """One chat completion against `tier`, returning the assistant's text.

    `context` is inserted between the system prompt and `user` for the callers
    that need prior turns. `temperature` is sent only when given — the
    reasoning-model touchpoints leave it unset rather than forcing a value the
    provider may reject. Exceptions propagate; each caller decides whether its
    own failure mode is fail-open, fail-loud, or a safe default.
    """
    client = build_client(tier.api_key, tier.api_base, read_timeout)
    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = await client.chat.completions.create(
        model=tier.model,
        messages=[
            {"role": "system", "content": system},
            *(context or []),
            {"role": "user", "content": user},
        ],
        **kwargs,
        **tier.extra_params,
    )
    return response.choices[0].message.content or ""


__all__ = ["build_client", "complete", "DEFAULT_READ_TIMEOUT"]
