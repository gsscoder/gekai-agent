from __future__ import annotations

from pathlib import Path

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