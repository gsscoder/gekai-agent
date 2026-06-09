from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agent.pipeline.blast_radius import BlastRadiusLocator
from agent.pipeline._directives import PIPELINE_DIRECTIVES
from agent.llm.types import Message


def _make_locator() -> BlastRadiusLocator:
    with patch("agent.pipeline.blast_radius.OpenAIAdapter"):
        return BlastRadiusLocator(model="test-model", api_key="key", api_base="http://localhost")


def test_locator_system_prompt_contains_pipeline_directives() -> None:
    locator = _make_locator()
    with patch("agent.pipeline.blast_radius.Agent") as MockAgent:
        instance = MagicMock()
        instance.tools = MagicMock()
        instance.run = AsyncMock(return_value=[Message(role="assistant", content="")])
        MockAgent.return_value = instance

        import asyncio
        asyncio.run(locator.locate(Path("."), "add a new feature"))

        _, kwargs = MockAgent.call_args
        assert kwargs["system"].startswith(PIPELINE_DIRECTIVES)
