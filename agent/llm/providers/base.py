"""Provider adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from ..types import CompletionResponse, Message, StreamEvent, TokenCount, ToolDefinition


class ProviderAdapter(ABC):
    """Abstract interface every provider implements."""

    @abstractmethod
    async def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Single-shot completion. Returns the model's response."""

    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError(f"{type(self).__name__} does not implement streaming.")
        yield  # pragma: no cover

    async def count_tokens(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> TokenCount:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement count_tokens."
        )

    @classmethod
    def default_retryable(cls) -> tuple[type[BaseException], ...]:
        return ()
