from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env.test", override=True)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--llm", action="store_true", default=False, help="run LLM integration tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--llm"):
        skip = pytest.mark.skip(reason="pass --llm to run")
        for item in items:
            if "llm" in item.keywords:
                item.add_marker(skip)
