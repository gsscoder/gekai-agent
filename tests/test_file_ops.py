import asyncio
from pathlib import Path

from agent.tools import _move_file, _copy_file, _delete_file, _make_dir


def run(coro):
    return asyncio.run(coro)


def _write(p: Path, text: str = "content") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


# ---------------------------------------------------------------------------
# _move_file
# ---------------------------------------------------------------------------

class TestMoveFile:
    def test_happy_path(self, tmp_path):
        _write(tmp_path / "a.txt", "hello")
        result = run(_move_file("a.txt", "b.txt", working_dir=tmp_path))
        assert result == "ok"
        assert not (tmp_path / "a.txt").exists()
        assert (tmp_path / "b.txt").read_text() == "hello"

    def test_creates_parent_dirs(self, tmp_path):
        _write(tmp_path / "a.txt")
        result = run(_move_file("a.txt", "sub/dir/b.txt", working_dir=tmp_path))
        assert result == "ok"
        assert (tmp_path / "sub" / "dir" / "b.txt").exists()

    def test_refuses_missing_src(self, tmp_path):
        result = run(_move_file("missing.txt", "b.txt", working_dir=tmp_path))
        assert result.startswith("error:")

    def test_refuses_existing_dst(self, tmp_path):
        _write(tmp_path / "a.txt")
        _write(tmp_path / "b.txt")
        result = run(_move_file("a.txt", "b.txt", working_dir=tmp_path))
        assert result.startswith("error:")
        assert (tmp_path / "a.txt").exists()

    def test_jail_src(self, tmp_path):
        result = run(_move_file("../outside.txt", "b.txt", working_dir=tmp_path))
        assert result == "error: path outside working directory"

    def test_jail_dst(self, tmp_path):
        _write(tmp_path / "a.txt")
        result = run(_move_file("a.txt", "../outside.txt", working_dir=tmp_path))
        assert result == "error: path outside working directory"


# ---------------------------------------------------------------------------
# _copy_file
# ---------------------------------------------------------------------------

class TestCopyFile:
    def test_happy_path(self, tmp_path):
        _write(tmp_path / "a.txt", "data")
        result = run(_copy_file("a.txt", "b.txt", working_dir=tmp_path))
        assert result == "ok"
        assert (tmp_path / "a.txt").exists()
        assert (tmp_path / "b.txt").read_text() == "data"

    def test_creates_parent_dirs(self, tmp_path):
        _write(tmp_path / "a.txt")
        result = run(_copy_file("a.txt", "sub/b.txt", working_dir=tmp_path))
        assert result == "ok"
        assert (tmp_path / "sub" / "b.txt").exists()

    def test_refuses_existing_dst(self, tmp_path):
        _write(tmp_path / "a.txt")
        _write(tmp_path / "b.txt")
        result = run(_copy_file("a.txt", "b.txt", working_dir=tmp_path))
        assert result.startswith("error:")

    def test_refuses_directory_src(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        result = run(_copy_file("subdir", "b.txt", working_dir=tmp_path))
        assert result.startswith("error:")

    def test_jail_src(self, tmp_path):
        result = run(_copy_file("../outside.txt", "b.txt", working_dir=tmp_path))
        assert result == "error: path outside working directory"

    def test_jail_dst(self, tmp_path):
        _write(tmp_path / "a.txt")
        result = run(_copy_file("a.txt", "../outside.txt", working_dir=tmp_path))
        assert result == "error: path outside working directory"


# ---------------------------------------------------------------------------
# _delete_file
# ---------------------------------------------------------------------------

class TestDeleteFile:
    def test_happy_path(self, tmp_path):
        _write(tmp_path / "a.txt")
        result = run(_delete_file("a.txt", working_dir=tmp_path))
        assert result == "ok"
        assert not (tmp_path / "a.txt").exists()

    def test_refuses_missing_file(self, tmp_path):
        result = run(_delete_file("missing.txt", working_dir=tmp_path))
        assert result.startswith("error:")

    def test_refuses_directory(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        result = run(_delete_file("subdir", working_dir=tmp_path))
        assert result.startswith("error:")

    def test_jail(self, tmp_path):
        result = run(_delete_file("../outside.txt", working_dir=tmp_path))
        assert result == "error: path outside working directory"


# ---------------------------------------------------------------------------
# _make_dir
# ---------------------------------------------------------------------------

class TestMakeDir:
    def test_happy_path(self, tmp_path):
        result = run(_make_dir("newdir", working_dir=tmp_path))
        assert result == "ok"
        assert (tmp_path / "newdir").is_dir()

    def test_nested(self, tmp_path):
        result = run(_make_dir("a/b/c", working_dir=tmp_path))
        assert result == "ok"
        assert (tmp_path / "a" / "b" / "c").is_dir()

    def test_idempotent(self, tmp_path):
        (tmp_path / "existing").mkdir()
        result = run(_make_dir("existing", working_dir=tmp_path))
        assert result == "ok"

    def test_jail(self, tmp_path):
        result = run(_make_dir("../outside", working_dir=tmp_path))
        assert result == "error: path outside working directory"
