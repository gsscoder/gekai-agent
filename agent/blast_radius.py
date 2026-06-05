from __future__ import annotations

from pathlib import Path

from openai import AsyncOpenAI

from .llm import Agent
from .llm.providers.openai import OpenAIAdapter
from .llm.types import Message, TextBlock
from .tools import make_tools

_SYSTEM = (
    "you locate files in a repository that are relevant to a requested change\n"
    "you have read-only tools — use them to verify files exist before listing them\n"
    "when done, output ONLY a plain list: one line per file\n"
    "format each line exactly as:\n"
    "  <relative/path/to/file> | keyword1, keyword2, keyword3\n"
    "keywords must be expanded and normalized — include synonyms and related terms "
    "the original request may not have named explicitly\n"
    "no prose, no explanation, no markdown, no extra lines\n"
    "do not edit or create any files"
)

_MAX_ITERATIONS = 5


def _parse_blast_radius_output(text: str) -> list[tuple[str, list[str]]]:
    results: list[tuple[str, list[str]]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "|" not in line:
            continue
        path_part, _, kw_part = line.partition("|")
        path = path_part.strip()
        if not path:
            continue
        keywords = [k.strip().lower() for k in kw_part.split(",") if k.strip()]
        keywords = list(dict.fromkeys(keywords))  # dedup, preserve order
        results.append((path, keywords))
    return results


class BlastRadiusLocator:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base

    async def locate(
        self,
        working_dir: Path,
        request: str,
    ) -> list[tuple[str, list[str]]]:
        adapter = OpenAIAdapter(api_key=self._api_key, base_url=self._api_base)
        agent = Agent(
            provider=adapter,
            model=self._model,
            system=_SYSTEM,
            max_iterations=_MAX_ITERATIONS,
        )
        for t in make_tools(working_dir):
            if t.is_read_only:
                agent.tools.register(t)

        history = await agent.run([Message(role="user", content=request)])

        last = history[-1]
        if isinstance(last.content, list):
            text = "\n".join(b.text for b in last.content if isinstance(b, TextBlock))
        else:
            text = last.content or ""

        return _parse_blast_radius_output(text)
