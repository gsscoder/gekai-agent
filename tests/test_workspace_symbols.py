from __future__ import annotations

from pathlib import Path

from agent.workspace.symbols import extract_symbol_names


def test_extract_symbol_names_python(tmp_path: Path) -> None:
    src = tmp_path / "module.py"
    src.write_text("def foo():\n    pass\n\n\nclass Bar:\n    pass\n")

    names = extract_symbol_names(src, "python")

    assert "foo" in names
    assert "Bar" in names


def test_extract_symbol_names_unsupported_lang_returns_empty(tmp_path: Path) -> None:
    src = tmp_path / "notes.md"
    src.write_text("# hello\n")

    assert extract_symbol_names(src, "markdown") == []


def test_extract_symbol_names_missing_file_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "missing.py"

    assert extract_symbol_names(missing, "python") == []


def test_extract_symbol_names_invalid_syntax_returns_empty_or_partial(tmp_path: Path) -> None:
    # tree-sitter is error-tolerant: invalid syntax yields a partial parse
    # tree rather than raising. extract_symbol_names must not crash either way.
    src = tmp_path / "broken.py"
    src.write_text("def (((( not valid python\n")

    names = extract_symbol_names(src, "python")

    assert isinstance(names, list)
