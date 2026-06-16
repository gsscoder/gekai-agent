import asyncio
from pathlib import Path

from agent.settings import load_allow_hidden
from agent.tools import _move_file, _copy_file, _delete_file, _make_dir
from agent.tools.files import _grep, _read_file, _edit_file, _write_file, _list_files, _file_info


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


class TestGrep:
    def test_finds_match_in_source(self, tmp_path):
        (tmp_path / "app.py").write_text("def api_home():\n    pass\n")
        result = run(_grep("api_home", working_dir=tmp_path))
        assert "app.py:1: def api_home():" in result

    def test_skips_ignored_dirs(self, tmp_path):
        # a match buried in .git / .venv / __pycache__ / .gekai must NOT be read —
        # walking those is what stalled grep for minutes (reads VCS internals,
        # virtualenvs, build output).
        (tmp_path / "real.py").write_text("needle here\n")
        for junk in (".git", ".venv", "__pycache__", ".gekai", "node_modules"):
            d = tmp_path / junk
            d.mkdir()
            (d / "buried.py").write_text("needle here\n")

        result = run(_grep("needle", working_dir=tmp_path))

        assert "real.py:1:" in result
        for junk in (".git", ".venv", "__pycache__", ".gekai", "node_modules"):
            assert junk not in result

    def test_no_matches(self, tmp_path):
        (tmp_path / "a.py").write_text("nothing relevant\n")
        result = run(_grep("zzz_absent", working_dir=tmp_path))
        assert result == "(no matches)"


# ---------------------------------------------------------------------------
# .aiignore red zone (forbidden, even via explicit path)
# ---------------------------------------------------------------------------

class TestAiignoreForbidden:
    def test_read_file_denied(self, tmp_path):
        _write(tmp_path / ".aiignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "top secret")
        result = run(_read_file("secret.txt", working_dir=tmp_path))
        assert result == "error: path outside working directory"

    def test_edit_file_denied(self, tmp_path):
        _write(tmp_path / ".aiignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "top secret")
        result = run(_edit_file("secret.txt", "x", "y", working_dir=tmp_path))
        assert result == "error: path outside working directory"

    def test_write_file_denied(self, tmp_path):
        _write(tmp_path / ".aiignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "top secret")
        result = run(_write_file("secret.txt", "data", working_dir=tmp_path))
        assert result == "error: path outside working directory"

    def test_grep_with_path_denied(self, tmp_path):
        _write(tmp_path / ".aiignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "top secret")
        result = run(_grep("anything", path="secret.txt", working_dir=tmp_path))
        assert result == "error: path outside working directory"

    def test_non_listed_file_still_readable(self, tmp_path):
        _write(tmp_path / ".aiignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "top secret")
        _write(tmp_path / "normal.txt", "nothing special")
        result = run(_read_file("normal.txt", working_dir=tmp_path))
        assert result == "nothing special"


# ---------------------------------------------------------------------------
# .gitignore hidden (Tier 1: hidden from discovery, still explicitly readable)
# ---------------------------------------------------------------------------

class TestGitignoreHidden:
    def test_list_files_excludes_gitignored_and_dotfiles(self, tmp_path):
        _write(tmp_path / ".gitignore", "ignored.txt\n")
        _write(tmp_path / "ignored.txt", "should be hidden")
        _write(tmp_path / "visible.txt", "should be visible")
        result = run(_list_files("*", working_dir=tmp_path))
        assert "visible.txt" in result
        assert "ignored.txt" not in result
        assert ".gitignore" not in result

    def test_grep_skips_gitignored_file(self, tmp_path):
        _write(tmp_path / ".gitignore", "ignored.txt\n")
        _write(tmp_path / "ignored.txt", "xyz needle\n")
        _write(tmp_path / "visible.txt", "nothing here\n")
        result = run(_grep("x", working_dir=tmp_path))
        assert "ignored.txt" not in result

    def test_read_file_denied_without_grant_on_gitignored_file(self, tmp_path):
        _write(tmp_path / ".gitignore", "ignored.txt\n")
        _write(tmp_path / "ignored.txt", "xyz needle\n")
        result = run(_read_file("ignored.txt", working_dir=tmp_path))
        assert result == "error: access to hidden path denied: ignored.txt"


# ---------------------------------------------------------------------------
# Hidden-path grant flow (grant_cb / allow_hidden)
# ---------------------------------------------------------------------------

class TestHiddenGrant:
    def test_grant_yes_allows_read_and_persists(self, tmp_path):
        _write(tmp_path / ".gitignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "top secret")

        calls = []

        async def cb(rel, mode):
            calls.append((rel, mode))
            return True

        result = run(_read_file("secret.txt", working_dir=tmp_path, allow_hidden=set(), grant_cb=cb))

        assert result == "top secret"
        assert calls == [("secret.txt", "read")]
        assert load_allow_hidden(tmp_path) == {"secret.txt"}

    def test_grant_no_denies(self, tmp_path):
        _write(tmp_path / ".gitignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "top secret")

        async def cb(rel, mode):
            return False

        result = run(_read_file("secret.txt", working_dir=tmp_path, allow_hidden=set(), grant_cb=cb))

        assert result == "error: access to hidden path denied: secret.txt"
        assert load_allow_hidden(tmp_path) == set()

    def test_granted_path_not_reprompted(self, tmp_path):
        _write(tmp_path / ".gitignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "top secret")

        allow_hidden: set[str] = set()
        count = 0

        async def cb(rel, mode):
            nonlocal count
            count += 1
            return True

        result1 = run(_read_file("secret.txt", working_dir=tmp_path, allow_hidden=allow_hidden, grant_cb=cb))
        result2 = run(_read_file("secret.txt", working_dir=tmp_path, allow_hidden=allow_hidden, grant_cb=cb))

        assert result1 == "top secret"
        assert result2 == "top secret"
        assert count == 1

    def test_forbidden_never_prompts(self, tmp_path):
        _write(tmp_path / ".aiignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "top secret")

        count = 0

        async def cb(rel, mode):
            nonlocal count
            count += 1
            return True

        result = run(_read_file("secret.txt", working_dir=tmp_path, allow_hidden=set(), grant_cb=cb))

        assert result == "error: path outside working directory"
        assert count == 0

    def test_granted_path_still_hidden_from_discovery(self, tmp_path):
        _write(tmp_path / ".gitignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "secret content")
        _write(tmp_path / "visible.txt", "nothing here")

        allow_hidden: set[str] = set()

        async def cb(rel, mode):
            return True

        result = run(_read_file("secret.txt", working_dir=tmp_path, allow_hidden=allow_hidden, grant_cb=cb))
        assert result == "secret content"
        assert allow_hidden == {"secret.txt"}

        list_result = run(_list_files("*", working_dir=tmp_path))
        assert "secret.txt" not in list_result
        assert "visible.txt" in list_result

        grep_result = run(_grep("secret", working_dir=tmp_path))
        assert "secret.txt" not in grep_result

    def test_write_mode_passed_for_edit(self, tmp_path):
        _write(tmp_path / ".gitignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "x")

        modes = []

        async def cb(rel, mode):
            modes.append(mode)
            return True

        result = run(_edit_file("secret.txt", "x", "y", working_dir=tmp_path, allow_hidden=set(), grant_cb=cb))

        assert result == "ok"
        assert modes == ["write"]

    def test_workspace_root_path_never_prompts(self, tmp_path):
        _write(tmp_path / "visible.txt", "hello")

        count = 0

        async def cb(rel, mode):
            nonlocal count
            count += 1
            return True

        result = run(_grep("hello", path=".", working_dir=tmp_path, allow_hidden=set(), grant_cb=cb))

        assert count == 0
        assert "visible.txt" in result
        assert load_allow_hidden(tmp_path) == set()


class TestHiddenGrantConcurrency:
    def test_concurrent_double_call_not_double_prompted(self, tmp_path):
        _write(tmp_path / ".gitignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "top secret")

        allow_hidden: set[str] = set()
        pending: set[str] = set()
        count = 0

        async def cb(rel, mode):
            nonlocal count
            count += 1
            await asyncio.sleep(0)
            return True

        async def both():
            return await asyncio.gather(
                _read_file("secret.txt", working_dir=tmp_path, allow_hidden=allow_hidden, grant_cb=cb, pending=pending),
                _read_file("secret.txt", working_dir=tmp_path, allow_hidden=allow_hidden, grant_cb=cb, pending=pending),
            )

        result1, result2 = run(both())

        assert count == 1
        results = {result1, result2}
        assert "top secret" in results
        assert "error: access to hidden path denied: secret.txt" in results
        assert pending == set()
        assert allow_hidden == {"secret.txt"}

    def test_concurrent_double_call_read_and_grep_not_double_prompted(self, tmp_path):
        _write(tmp_path / ".gitignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "top secret")

        allow_hidden: set[str] = set()
        pending: set[str] = set()
        count = 0

        async def cb(rel, mode):
            nonlocal count
            count += 1
            await asyncio.sleep(0)
            return True

        async def both():
            return await asyncio.gather(
                _read_file("secret.txt", working_dir=tmp_path, allow_hidden=allow_hidden, grant_cb=cb, pending=pending),
                _grep("secret", path="secret.txt", working_dir=tmp_path, allow_hidden=allow_hidden, grant_cb=cb, pending=pending),
            )

        result1, result2 = run(both())

        assert count == 1
        assert pending == set()
        assert allow_hidden == {"secret.txt"}
        denied = "error: access to hidden path denied: secret.txt"
        if result1 == denied:
            assert result2 != denied
        else:
            assert result1 == "top secret"
            assert result2 == denied

    def test_followup_call_after_race_not_reprompted(self, tmp_path):
        _write(tmp_path / ".gitignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "top secret")

        allow_hidden: set[str] = set()
        pending: set[str] = set()
        count = 0

        async def cb(rel, mode):
            nonlocal count
            count += 1
            await asyncio.sleep(0)
            return True

        async def both():
            return await asyncio.gather(
                _read_file("secret.txt", working_dir=tmp_path, allow_hidden=allow_hidden, grant_cb=cb, pending=pending),
                _read_file("secret.txt", working_dir=tmp_path, allow_hidden=allow_hidden, grant_cb=cb, pending=pending),
            )

        run(both())
        assert count == 1

        result = run(_read_file("secret.txt", working_dir=tmp_path, allow_hidden=allow_hidden, grant_cb=cb, pending=pending))

        assert result == "top secret"
        assert count == 1


# ---------------------------------------------------------------------------
# Directory-as-path guard (file-only tools must reject directory targets)
# ---------------------------------------------------------------------------

class TestDirectoryPathGuard:
    """Passing '.' or any directory path to file-only tools must return a clear
    'is a directory' error rather than an OS error or a grant prompt."""

    def _no_grant(self):
        async def cb(rel, mode):
            raise AssertionError("grant_cb must not be called for directory paths")
        return cb

    def test_read_file_dot_returns_directory_error(self, tmp_path):
        _write(tmp_path / "visible.txt", "hello")
        result = run(_read_file(".", working_dir=tmp_path))
        assert "is a directory" in result
        assert result.startswith("error:")

    def test_read_file_subdir_returns_directory_error(self, tmp_path):
        (tmp_path / "sub").mkdir()
        result = run(_read_file("sub", working_dir=tmp_path))
        assert "is a directory" in result
        assert result.startswith("error:")

    def test_file_info_dot_returns_directory_error(self, tmp_path):
        _write(tmp_path / "visible.txt", "hello")
        result = run(_file_info(".", working_dir=tmp_path))
        assert "is a directory" in result
        assert result.startswith("error:")

    def test_edit_file_dot_returns_directory_error(self, tmp_path):
        result = run(_edit_file(".", "old", "new", working_dir=tmp_path))
        assert "is a directory" in result
        assert result.startswith("error:")

    def test_write_file_dot_returns_directory_error(self, tmp_path):
        result = run(_write_file(".", "content", working_dir=tmp_path))
        assert "is a directory" in result
        assert result.startswith("error:")

    def test_move_file_src_dir_returns_directory_error(self, tmp_path):
        (tmp_path / "srcdir").mkdir()
        result = run(_move_file("srcdir", "dst.txt", working_dir=tmp_path))
        assert "is a directory" in result
        assert result.startswith("error:")

    def test_read_file_dot_does_not_prompt_grant(self, tmp_path):
        _write(tmp_path / "visible.txt", "hello")
        count = 0

        async def cb(rel, mode):
            nonlocal count
            count += 1
            return True

        result = run(_read_file(".", working_dir=tmp_path, allow_hidden=set(), grant_cb=cb))
        assert "is a directory" in result
        assert count == 0
