from __future__ import annotations

from pathlib import Path

from openai import AsyncOpenAI

from .llm import Agent
from .llm.providers.openai import OpenAIAdapter
from .llm.types import Message, TextBlock
from .tools import make_tools

# Extensions that count toward the blast-radius area metric.
# Broader than _EXT_TO_LANG (symbol-parse support) — gate coverage ≠ AST coverage.
# Manifests, configs, docs, lockfiles: inspected by locator but never counted.
_CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".pyi",                          # Python
    ".js", ".jsx", ".mjs", ".cjs",          # JavaScript
    ".ts", ".tsx",                          # TypeScript
    ".vue", ".svelte",                      # component frameworks
    ".go",                                  # Go
    ".java",                                # Java
    ".cs",                                  # C#
    ".kt", ".kts",                          # Kotlin
    ".swift",                               # Swift
    ".rs",                                  # Rust
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",  # C / C++
    ".rb",                                  # Ruby
    ".php",                                 # PHP
    ".scala",                               # Scala
    ".dart",                                # Dart
    ".ex", ".exs",                          # Elixir
    ".lua",                                 # Lua
    ".hs",                                  # Haskell
    ".r",                                   # R
    ".ipynb",                               # Jupyter notebooks
})

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


def _blast_area_survivors(paths: list[str]) -> list[Path]:
    parents: set[Path] = set()
    for p in paths:
        if Path(p).suffix.lower() in _CODE_EXTENSIONS:
            parents.add(Path(p).parent)
    return sorted(
        d for d in parents
        if not any(ancestor != d and d.is_relative_to(ancestor) for ancestor in parents)
    )


def count_blast_areas(paths: list[str]) -> int:
    return len(_blast_area_survivors(paths))


def evaluate_blast_radius_gate(
    entries: list[tuple[str, list[str]]],
    limit: int,
) -> tuple[bool, str | None]:
    paths = [path for path, _ in entries]
    survivors = _blast_area_survivors(paths)
    count = len(survivors)
    if count > limit:
        labels = ", ".join(str(d) for d in survivors)
        return (True, f"change spans {count} areas (limit {limit}): {labels}")
    return (False, None)


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
