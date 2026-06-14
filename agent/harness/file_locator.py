from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from ..llm import Agent, MaxIterationsExceeded
from ..llm.providers.openai import OpenAIAdapter
from ..llm.types import Message, TextBlock
from ..pipeline import PIPELINE_DIRECTIVES
from ..tools import make_tools

_SYSTEM_TEMPLATE = (
    "you locate the SMALLEST verified set of files relevant to a user request\n"
    "you DISCOVER — you do not plan, explain, edit, or perform the request\n"
    "<user_request>\n"
    "{request}\n"
    "{hint_section}"
    "<strategy>\n"
    "the request itself is your richest clue — drain every drop from it:\n"
    "1. mine it for concrete signals: identifiers, class/function/symbol names, "
    "module or file names, domain nouns and verbs\n"
    "2. expand each signal into search variants across casings and joins — "
    "e.g. 'prompt builder' -> prompt, builder, promptbuilder, PromptBuilder, prompt_builder, build_prompt\n"
    "3. grep those variants to find where they live; confirm hits with read_file / symbols\n"
    "4. use list_files ONLY with a targeted glob once a name hint points at it\n"
    "   NEVER enumerate the repo root or walk broad trees blindly — that is not your job\n"
    "<scope>\n"
    "stay minimal but complete: the files the request directly concerns — "
    "for changes, where work lands; for questions, where the answer lives — "
    "not the whole subsystem around them\n"
    "verify every file exists before listing it\n"
    "if the request needs no codebase context (greetings, acknowledgments, "
    "general knowledge unrelated to the codebase), output nothing immediately, "
    "without using tools\n"
    "<output>\n"
    "output ONLY a plain list: one line per file\n"
    "format each line exactly as:\n"
    "  `relative/path/to/file` | keyword1, keyword2, keyword3\n"
    "keywords must be expanded and normalized — include synonyms and related terms "
    "the original request may not have named explicitly\n"
    "no prose, no explanation, no markdown, no extra lines\n"
    "do not edit or create any files"
)


def _format_hint_section(hint_paths: list[str] | None) -> str:
    if not hint_paths:
        return ""
    paths = ", ".join(f"`{p}`" for p in hint_paths)
    return (
        "<hints>\n"
        f"previously-relevant for these signals: {paths}\n"
        "verify each still exists and is relevant before listing it; do not assume\n"
    )


def _parse_locator_output(text: str) -> list[tuple[str, list[str]]]:
    results: list[tuple[str, list[str]]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "|" not in line:
            continue
        path_part, _, kw_part = line.partition("|")
        path = path_part.strip().strip("`")
        if not path:
            continue
        keywords = [k.strip().lower() for k in kw_part.split(",") if k.strip()]
        keywords = list(dict.fromkeys(keywords))  # dedup, preserve order
        results.append((path, keywords))
    return results


class FileLocator:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        max_iterations: int = 4,
        wall_clock_timeout: float = 25.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._max_iterations = max_iterations
        self._wall_clock_timeout = wall_clock_timeout

    async def locate(
        self,
        working_dir: Path,
        request: str,
        hint_paths: list[str] | None = None,
    ) -> tuple[list[tuple[str, list[str]]], bool]:
        adapter = OpenAIAdapter(
            api_key=self._api_key,
            base_url=self._api_base,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=30.0),
        )
        system = PIPELINE_DIRECTIVES + _SYSTEM_TEMPLATE.format(
            request=request, hint_section=_format_hint_section(hint_paths)
        )
        agent = Agent(
            provider=adapter,
            model=self._model,
            system=system,
            max_iterations=self._max_iterations,
            wall_clock_timeout=self._wall_clock_timeout,
        )
        for t in make_tools(working_dir):
            if t.is_read_only:
                agent.tools.register(t)

        try:
            history = await agent.run([
                Message(role="user", content="locate the files"),
            ])
        except (MaxIterationsExceeded, asyncio.TimeoutError):
            return [], True

        last = history[-1]
        if isinstance(last.content, list):
            text = "\n".join(b.text for b in last.content if isinstance(b, TextBlock))
        else:
            text = last.content or ""

        return _parse_locator_output(text), False
