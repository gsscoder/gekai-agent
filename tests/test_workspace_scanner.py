from __future__ import annotations

from pathlib import Path

from agent.workspace import scanner


def test_builtin_floor_excludes_skip_dirs_with_no_ignore_files(tmp_path: Path) -> None:
    for skip_dir in (".git", ".venv", "__pycache__", "node_modules"):
        d = tmp_path / skip_dir
        d.mkdir()
        (d / "file.txt").write_text("noise\n")

    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("x = 1\n")

    files = scanner.list_files(tmp_path)
    dirs = scanner.list_dirs(tmp_path)

    assert "src/main.py" in files
    for skip_dir in (".git", ".venv", "__pycache__", "node_modules"):
        assert f"{skip_dir}/file.txt" not in files
        assert f"{skip_dir}/" not in dirs


def test_gitignore_patterns_excluded_from_listing(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("build\n*.log\n")

    build = tmp_path / "build"
    build.mkdir()
    (build / "output.bin").write_text("binary\n")

    (tmp_path / "app.log").write_text("log\n")

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "app.log").write_text("log\n")

    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("x = 1\n")

    files = scanner.list_files(tmp_path)
    dirs = scanner.list_dirs(tmp_path)

    assert "build/output.bin" not in files
    assert "app.log" not in files
    assert "logs/app.log" not in files
    assert "src/main.py" in files

    assert "build/" not in dirs


def test_walk_subdir_respects_root_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("secrets/\n")

    sub = tmp_path / "sub"
    secrets = sub / "secrets"
    secrets.mkdir(parents=True)
    (secrets / "key.txt").write_text("top secret\n")
    (sub / "normal.txt").write_text("hello\n")

    all_dirnames: list[str] = []
    all_filenames: list[str] = []
    for _, dirnames, filenames in scanner._walk(tmp_path / "sub", root=tmp_path):
        all_dirnames.extend(dirnames)
        all_filenames.extend(filenames)

    assert "normal.txt" in all_filenames
    assert "secrets" not in all_dirnames
    assert "key.txt" not in all_filenames
