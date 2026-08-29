"""Shared `AsyncOpenAI` client construction: every in-process call site wants
the same connect/write/pool timeout policy, only the read timeout varies —
one-shot classification calls use the default, longer planning calls pass a
larger value explicitly.
"""

from __future__ import annotations

import httpx
from openai import AsyncOpenAI

_CONNECT_TIMEOUT = 5.0
_DEFAULT_READ_TIMEOUT = 30.0
_WRITE_TIMEOUT = 30.0
_POOL_TIMEOUT = 30.0


def build_openai_client(
    api_key: str | None,
    api_base: str | None,
    *,
    read_timeout: float = _DEFAULT_READ_TIMEOUT,
) -> AsyncOpenAI:
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


__all__ = ["build_openai_client"]
