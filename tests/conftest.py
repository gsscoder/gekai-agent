from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv

from agent.llm.tiers import ModelCatalogEntry, TierBinding, TierName

load_dotenv(Path(__file__).parent / ".env.test", override=True)


def run(coro):
    return asyncio.run(coro)


def mock_llm_response(text: str | None) -> SimpleNamespace:
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice])


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
    parser.addoption("--llm", action="store_true", default=False, help="run LLM integration tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--llm"):
        skip = pytest.mark.skip(reason="pass --llm to run")
        for item in items:
            if "llm" in item.keywords:
                item.add_marker(skip)
