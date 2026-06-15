from __future__ import annotations

from pathlib import Path

from agent.workspace.ignore import IgnoreRules


def test_builtin_floor_with_no_gitignore_or_aiignore(tmp_path: Path) -> None:
    rules = IgnoreRules(tmp_path)

    assert rules.is_hidden(".git/") is True
    assert rules.is_hidden(".venv/") is True
    assert rules.is_hidden("__pycache__/") is True
    assert rules.is_hidden("node_modules/") is True
    assert rules.is_hidden(".gekai/") is True

    assert rules.is_hidden("src/main.py") is False


def test_gitignore_patterns_are_hidden(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("build\n*.log\n")

    rules = IgnoreRules(tmp_path)

    assert rules.is_hidden("build/") is True
    assert rules.is_hidden("build") is True
    assert rules.is_hidden("app.log") is True
    assert rules.is_hidden("logs/app.log") is True
    assert rules.is_hidden("src/main.py") is False


def test_aiignore_patterns_are_hidden_and_forbidden(tmp_path: Path) -> None:
    (tmp_path / ".aiignore").write_text("secrets/\ncreds.json\n")

    rules = IgnoreRules(tmp_path)

    assert rules.is_hidden("secrets/") is True
    assert rules.is_forbidden("secrets/") is True
    assert rules.is_hidden("creds.json") is True
    assert rules.is_forbidden("creds.json") is True


def test_gitignore_only_path_is_hidden_but_not_forbidden(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("build\n")

    rules = IgnoreRules(tmp_path)

    assert rules.is_hidden("build/") is True
    assert rules.is_forbidden("build/") is False


def test_no_aiignore_means_nothing_is_forbidden(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("build\n*.log\n")

    rules = IgnoreRules(tmp_path)

    assert rules.is_forbidden("build/") is False
    assert rules.is_forbidden("app.log") is False
    assert rules.is_forbidden("src/main.py") is False
    assert rules.is_forbidden(".git/") is False


def test_nested_dotfiles_and_dotdirs_are_hidden(tmp_path: Path) -> None:
    rules = IgnoreRules(tmp_path)

    assert rules.is_hidden("src/.hidden") is True
    assert rules.is_hidden("a/b/.git/") is True


def test_gitignore_with_invalid_utf8_bytes_does_not_crash(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_bytes(b"build\n\xff\xfe invalid\n*.log\n")

    rules = IgnoreRules(tmp_path)

    assert rules.is_hidden("build/") is True
    assert rules.is_hidden("app.log") is True
