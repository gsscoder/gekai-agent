from __future__ import annotations

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
