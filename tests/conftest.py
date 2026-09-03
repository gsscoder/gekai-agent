from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import load_dotenv

from agent.tiers.catalog import ModelCatalogEntry, TierBinding, TierName
from agent.tiers.resolve import ResolvedTier

load_dotenv(Path(__file__).parent / ".env.test", override=True)


def run(coro):
    return asyncio.run(coro)


def mock_llm_response(text: str | None) -> SimpleNamespace:
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def llm_create():
    """Stubs the one client `agent.oneshot.complete` builds, yielding the
    `chat.completions.create` mock so a test can set its return value or
    side effect and then assert on the call it received."""
    create = AsyncMock(return_value=mock_llm_response(""))
    with patch("agent.oneshot.build_client") as mock_build:
        mock_build.return_value.chat.completions.create = create
        yield create


ONESHOT_TIER = ResolvedTier(
    model="test-model", api_key="key", api_base="http://localhost", extra_params={},
)


TIER_CATALOG = {
    "fast-model": ModelCatalogEntry(name="fast-model", base_url="https://fast.example.com", efforts=("low", "medium"), thinking=False),
    "supp-model": ModelCatalogEntry(name="supp-model", base_url="https://supp.example.com", efforts=("low", "medium"), thinking=False),
    "core-model": ModelCatalogEntry(name="core-model", base_url="https://core.example.com", efforts=("high", "xhigh"), thinking=True),
}

TIER_BINDINGS = {
    TierName.FAST: TierBinding(model="fast-model", default_effort="low"),
    TierName.SUPP: TierBinding(model="supp-model", default_effort="low"),
    TierName.CORE: TierBinding(model="core-model", default_effort="high", thinking=True),
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--llm-harness", action="store_true", default=False, help="run LLM harness integration tests")
    parser.addoption("--llm-compact", action="store_true", default=False, help="run LLM /compact integration tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gated = {
        "llm_harness": config.getoption("--llm-harness"),
        "llm_compact": config.getoption("--llm-compact"),
    }
    for item in items:
        for marker, enabled in gated.items():
            if marker in item.keywords and not enabled:
                item.add_marker(pytest.mark.skip(reason=f"pass --{marker.replace('_', '-')} to run"))
