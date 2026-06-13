from __future__ import annotations

import importlib
from pathlib import Path

_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
}

# (module_name, function_name) to obtain the language capsule
_LANG_TO_MODULE: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
}

_LANG_QUERIES: dict[str, dict[str, str]] = {
    "python": {
        "function": "(function_definition name: (identifier) @name)",
        "class": "(class_definition name: (identifier) @name)",
    },
    "javascript": {
        "function": "(function_declaration name: (identifier) @name)",
        "class": "(class_declaration name: (identifier) @name)",
        "method": "(method_definition name: (property_identifier) @name)",
    },
    "typescript": {
        "function": "(function_declaration name: (identifier) @name)",
        "class": "(class_declaration name: (type_identifier) @name)",
        "method": "(method_definition name: (property_identifier) @name)",
        "interface": "(interface_declaration name: (type_identifier) @name)",
    },
    "go": {
        "function": "(function_declaration name: (identifier) @name)",
        "method": "(method_declaration name: (field_identifier) @name)",
        "type": "(type_spec name: (type_identifier) @name)",
    },
}
_LANG_QUERIES["tsx"] = _LANG_QUERIES["typescript"]


def extract_symbol_names(path: Path, lang_name: str) -> list[str]:
    """Extract function/class/method/etc. names from a source file via tree-sitter.

    Synchronous and batch-friendly (unlike the `_symbols` tool wrapper in
    `agent.tools.files`, which is async and returns a formatted string).
    Returns an empty list on any error: missing tree-sitter, unsupported
    language, unreadable file, or parse failure — non-fatal by design since
    callers use this for best-effort keyword enrichment.
    """
    try:
        from tree_sitter import Language, Parser, Query, QueryCursor
    except ImportError:
        return []

    queries = _LANG_QUERIES.get(lang_name)
    if queries is None:
        return []

    try:
        source = path.read_bytes()
    except OSError:
        return []

    mod_name, fn_name = _LANG_TO_MODULE[lang_name]
    try:
        mod = importlib.import_module(mod_name)
        language = Language(getattr(mod, fn_name)())
    except Exception:
        return []

    parser = Parser(language)
    tree = parser.parse(source)

    names: list[str] = []
    for query_str in queries.values():
        try:
            cursor = QueryCursor(Query(language, query_str))
            caps: dict[str, list] = cursor.captures(tree.root_node)
        except Exception:
            continue
        for node in caps.get("name", []):
            names.append(node.text.decode("utf-8", errors="replace"))

    return names
