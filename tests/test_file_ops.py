import asyncio
from pathlib import Path

import pytest

from agent.settings import load_allow_hidden
from agent.tools import _move_file, _copy_file, _delete_file, _make_dir
from agent.tools.files import FileToolContext, _grep, _read_file, _edit_file, _write_file, _list_files, _file_info


def run(coro):
    return asyncio.run(coro)


def _ctx(
    working_dir: Path,
    allow_hidden: set[str] | None = None,
    grant_cb=None,
    pending: set[str] | None = None,
) -> FileToolContext:
    return FileToolContext(working_dir=working_dir, allow_hidden=allow_hidden, grant_cb=grant_cb, pending=pending)


def _write(p: Path, text: str = "content") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture
def secret_aiignore(tmp_path: Path) -> Path:
    _write(tmp_path / ".aiignore", "secret.txt\n")
    _write(tmp_path / "secret.txt", "top secret")
    return tmp_path


@pytest.fixture
def secret_gitignore(tmp_path: Path) -> Path:
    _write(tmp_path / ".gitignore", "secret.txt\n")
    _write(tmp_path / "secret.txt", "top secret")
    return tmp_path


# ---------------------------------------------------------------------------
# _move_file
# ---------------------------------------------------------------------------

class TestMoveFile:
    def test_happy_path(self, tmp_path):
        _write(tmp_path / "a.txt", "hello")
        result = run(_move_file("a.txt", "b.txt", ctx=_ctx(tmp_path)))
        assert result == "ok"
        assert not (tmp_path / "a.txt").exists()
        assert (tmp_path / "b.txt").read_text() == "hello"

    def test_creates_parent_dirs(self, tmp_path):
        _write(tmp_path / "a.txt")
        result = run(_move_file("a.txt", "sub/dir/b.txt", ctx=_ctx(tmp_path)))
        assert result == "ok"
        assert (tmp_path / "sub" / "dir" / "b.txt").exists()

    def test_refuses_missing_src(self, tmp_path):
        result = run(_move_file("missing.txt", "b.txt", ctx=_ctx(tmp_path)))
        assert result.startswith("error:")

    def test_refuses_existing_dst(self, tmp_path):
        _write(tmp_path / "a.txt")
        _write(tmp_path / "b.txt")
        result = run(_move_file("a.txt", "b.txt", ctx=_ctx(tmp_path)))
        assert result.startswith("error:")
        assert (tmp_path / "a.txt").exists()

    def test_jail_src(self, tmp_path):
        result = run(_move_file("../outside.txt", "b.txt", ctx=_ctx(tmp_path)))
        assert result == "error: path outside working directory"

    def test_jail_dst(self, tmp_path):
        _write(tmp_path / "a.txt")
        result = run(_move_file("a.txt", "../outside.txt", ctx=_ctx(tmp_path)))
        assert result == "error: path outside working directory"


# ---------------------------------------------------------------------------
# _copy_file
# ---------------------------------------------------------------------------

class TestCopyFile:
    def test_happy_path(self, tmp_path):
        _write(tmp_path / "a.txt", "data")
        result = run(_copy_file("a.txt", "b.txt", ctx=_ctx(tmp_path)))
        assert result == "ok"
        assert (tmp_path / "a.txt").exists()
        assert (tmp_path / "b.txt").read_text() == "data"

    def test_creates_parent_dirs(self, tmp_path):
        _write(tmp_path / "a.txt")
        result = run(_copy_file("a.txt", "sub/b.txt", ctx=_ctx(tmp_path)))
        assert result == "ok"
        assert (tmp_path / "sub" / "b.txt").exists()

    def test_refuses_existing_dst(self, tmp_path):
        _write(tmp_path / "a.txt")
        _write(tmp_path / "b.txt")
        result = run(_copy_file("a.txt", "b.txt", ctx=_ctx(tmp_path)))
        assert result.startswith("error:")

    def test_refuses_directory_src(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        result = run(_copy_file("subdir", "b.txt", ctx=_ctx(tmp_path)))
        assert result.startswith("error:")

    def test_jail_src(self, tmp_path):
        result = run(_copy_file("../outside.txt", "b.txt", ctx=_ctx(tmp_path)))
        assert result == "error: path outside working directory"

    def test_jail_dst(self, tmp_path):
        _write(tmp_path / "a.txt")
        result = run(_copy_file("a.txt", "../outside.txt", ctx=_ctx(tmp_path)))
        assert result == "error: path outside working directory"


# ---------------------------------------------------------------------------
# _delete_file
# ---------------------------------------------------------------------------

class TestDeleteFile:
    def test_happy_path(self, tmp_path):
        _write(tmp_path / "a.txt")
        result = run(_delete_file("a.txt", ctx=_ctx(tmp_path)))
        assert result == "ok"
        assert not (tmp_path / "a.txt").exists()

    def test_refuses_missing_file(self, tmp_path):
        result = run(_delete_file("missing.txt", ctx=_ctx(tmp_path)))
        assert result.startswith("error:")

    def test_refuses_directory(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        result = run(_delete_file("subdir", ctx=_ctx(tmp_path)))
        assert result.startswith("error:")

    def test_jail(self, tmp_path):
        result = run(_delete_file("../outside.txt", ctx=_ctx(tmp_path)))
        assert result == "error: path outside working directory"


# ---------------------------------------------------------------------------
# _make_dir
# ---------------------------------------------------------------------------

class TestMakeDir:
    def test_happy_path(self, tmp_path):
        result = run(_make_dir("newdir", ctx=_ctx(tmp_path)))
        assert result == "ok"
        assert (tmp_path / "newdir").is_dir()

    def test_nested(self, tmp_path):
        result = run(_make_dir("a/b/c", ctx=_ctx(tmp_path)))
        assert result == "ok"
        assert (tmp_path / "a" / "b" / "c").is_dir()

    def test_idempotent(self, tmp_path):
        (tmp_path / "existing").mkdir()
        result = run(_make_dir("existing", ctx=_ctx(tmp_path)))
        assert result == "ok"

    def test_jail(self, tmp_path):
        result = run(_make_dir("../outside", ctx=_ctx(tmp_path)))
        assert result == "error: path outside working directory"


class TestGrep:
    def test_finds_match_in_source(self, tmp_path):
        (tmp_path / "app.py").write_text("def api_home():\n    pass\n")
        result = run(_grep("api_home", ctx=_ctx(tmp_path)))
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

        result = run(_grep("needle", ctx=_ctx(tmp_path)))

        assert "real.py:1:" in result
        for junk in (".git", ".venv", "__pycache__", ".gekai", "node_modules"):
            assert junk not in result

    def test_no_matches(self, tmp_path):
        (tmp_path / "a.py").write_text("nothing relevant\n")
        result = run(_grep("zzz_absent", ctx=_ctx(tmp_path)))
        assert result == "(no matches)"

    def test_skips_unreadable_file_without_crashing(self, tmp_path, monkeypatch):
        # the narrowed `except (OSError, UnicodeDecodeError)` must still swallow a
        # per-file read failure and keep matching the rest of the candidates.
        (tmp_path / "good.py").write_text("needle here\n")
        bad = tmp_path / "bad.py"
        bad.write_text("needle here too\n")

        original_read_text = Path.read_text

        def _flaky_read_text(self: Path, *args: object, **kwargs: object) -> str:
            if self.name == "bad.py":
                raise OSError("permission denied")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", _flaky_read_text)

        result = run(_grep("needle", ctx=_ctx(tmp_path)))

        assert "good.py:1: needle here" in result
        assert "bad.py" not in result


# ---------------------------------------------------------------------------
# _edit_file — batch `edits` form
# ---------------------------------------------------------------------------

class TestEditFileBatch:
    def test_single_hunk_call_unchanged(self, tmp_path):
        _write(tmp_path / "a.txt", "hello world")
        result = run(_edit_file("a.txt", "hello", "goodbye", ctx=_ctx(tmp_path)))
        assert result == "ok"
        assert (tmp_path / "a.txt").read_text() == "goodbye world"

    def test_batch_applies_all_hunks_in_order(self, tmp_path):
        _write(tmp_path / "a.txt", "one two three")
        edits = [
            {"old_str": "one", "new_str": "1"},
            {"old_str": "two", "new_str": "2"},
            {"old_str": "three", "new_str": "3"},
        ]
        result = run(_edit_file("a.txt", edits=edits, ctx=_ctx(tmp_path)))
        assert result == "ok"
        assert (tmp_path / "a.txt").read_text() == "1 2 3"

    def test_batch_hunk_sees_prior_hunks_edits(self, tmp_path):
        _write(tmp_path / "a.txt", "foo")
        edits = [
            {"old_str": "foo", "new_str": "bar"},
            {"old_str": "bar", "new_str": "baz"},
        ]
        result = run(_edit_file("a.txt", edits=edits, ctx=_ctx(tmp_path)))
        assert result == "ok"
        assert (tmp_path / "a.txt").read_text() == "baz"

    def test_batch_fails_atomically_when_hunk_not_found(self, tmp_path):
        original = "one two three"
        _write(tmp_path / "a.txt", original)
        edits = [
            {"old_str": "one", "new_str": "1"},
            {"old_str": "missing", "new_str": "x"},
            {"old_str": "three", "new_str": "3"},
        ]
        result = run(_edit_file("a.txt", edits=edits, ctx=_ctx(tmp_path)))
        assert result == "error: edits[1].old_str not found in a.txt"
        assert (tmp_path / "a.txt").read_text() == original

    def test_batch_fails_atomically_when_prior_hunk_consumed_text(self, tmp_path):
        original = "foo bar"
        _write(tmp_path / "a.txt", original)
        edits = [
            {"old_str": "foo", "new_str": "baz"},
            {"old_str": "foo", "new_str": "qux"},
        ]
        result = run(_edit_file("a.txt", edits=edits, ctx=_ctx(tmp_path)))
        assert result == "error: edits[1].old_str not found in a.txt"
        assert (tmp_path / "a.txt").read_text() == original

    def test_both_old_str_and_edits_rejected(self, tmp_path):
        original = "hello"
        _write(tmp_path / "a.txt", original)
        result = run(_edit_file(
            "a.txt", "hello", "goodbye", edits=[{"old_str": "hello", "new_str": "x"}],
            ctx=_ctx(tmp_path),
        ))
        assert result == "error: pass either old_str/new_str or edits, not both"
        assert (tmp_path / "a.txt").read_text() == original

    def test_neither_old_str_nor_edits_rejected(self, tmp_path):
        original = "hello"
        _write(tmp_path / "a.txt", original)
        result = run(_edit_file("a.txt", ctx=_ctx(tmp_path)))
        assert result == "error: old_str/new_str required when edits is not given"
        assert (tmp_path / "a.txt").read_text() == original


# ---------------------------------------------------------------------------
# .aiignore red zone (forbidden, even via explicit path)
# ---------------------------------------------------------------------------

class TestAiignoreForbidden:
    def test_read_file_denied(self, secret_aiignore: Path):
        result = run(_read_file("secret.txt", ctx=_ctx(secret_aiignore)))
        assert result == "error: path outside working directory"

    def test_edit_file_denied(self, secret_aiignore: Path):
        result = run(_edit_file("secret.txt", "x", "y", ctx=_ctx(secret_aiignore)))
        assert result == "error: path outside working directory"

    def test_write_file_denied(self, secret_aiignore: Path):
        result = run(_write_file("secret.txt", "data", ctx=_ctx(secret_aiignore)))
        assert result == "error: path outside working directory"

    def test_grep_with_path_denied(self, secret_aiignore: Path):
        result = run(_grep("anything", path="secret.txt", ctx=_ctx(secret_aiignore)))
        assert result == "error: path outside working directory"

    def test_non_listed_file_still_readable(self, secret_aiignore: Path):
        _write(secret_aiignore / "normal.txt", "nothing special")
        result = run(_read_file("normal.txt", ctx=_ctx(secret_aiignore)))
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
        result = run(_grep("x", ctx=_ctx(tmp_path)))
        assert "ignored.txt" not in result

    def test_read_file_denied_without_grant_on_gitignored_file(self, tmp_path):
        _write(tmp_path / ".gitignore", "ignored.txt\n")
        _write(tmp_path / "ignored.txt", "xyz needle\n")
        result = run(_read_file("ignored.txt", ctx=_ctx(tmp_path)))
        assert result == "error: access to hidden path denied: ignored.txt"


# ---------------------------------------------------------------------------
# Hidden-path grant flow (grant_cb / allow_hidden)
# ---------------------------------------------------------------------------

class TestHiddenGrant:
    def test_grant_yes_allows_read_and_persists(self, secret_gitignore: Path):
        calls = []

        async def cb(rel, mode):
            calls.append((rel, mode))
            return True

        result = run(_read_file("secret.txt", ctx=_ctx(secret_gitignore, allow_hidden=set(), grant_cb=cb)))

        assert result == "top secret"
        assert calls == [("secret.txt", "read")]
        assert load_allow_hidden(secret_gitignore) == {"secret.txt"}

    def test_grant_no_denies(self, secret_gitignore: Path):
        async def cb(rel, mode):
            return False

        result = run(_read_file("secret.txt", ctx=_ctx(secret_gitignore, allow_hidden=set(), grant_cb=cb)))

        assert result == "error: access to hidden path denied: secret.txt"
        assert load_allow_hidden(secret_gitignore) == set()

    def test_granted_path_not_reprompted(self, secret_gitignore: Path):
        allow_hidden: set[str] = set()
        count = 0

        async def cb(rel, mode):
            nonlocal count
            count += 1
            return True

        result1 = run(_read_file("secret.txt", ctx=_ctx(secret_gitignore, allow_hidden=allow_hidden, grant_cb=cb)))
        result2 = run(_read_file("secret.txt", ctx=_ctx(secret_gitignore, allow_hidden=allow_hidden, grant_cb=cb)))

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

        result = run(_read_file("secret.txt", ctx=_ctx(tmp_path, allow_hidden=set(), grant_cb=cb)))

        assert result == "error: path outside working directory"
        assert count == 0

    def test_granted_path_still_hidden_from_discovery(self, tmp_path):
        _write(tmp_path / ".gitignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "secret content")
        _write(tmp_path / "visible.txt", "nothing here")

        allow_hidden: set[str] = set()

        async def cb(rel, mode):
            return True

        result = run(_read_file("secret.txt", ctx=_ctx(tmp_path, allow_hidden=allow_hidden, grant_cb=cb)))
        assert result == "secret content"
        assert allow_hidden == {"secret.txt"}

        list_result = run(_list_files("*", working_dir=tmp_path))
        assert "secret.txt" not in list_result
        assert "visible.txt" in list_result

        grep_result = run(_grep("secret", ctx=_ctx(tmp_path)))
        assert "secret.txt" not in grep_result

    def test_write_mode_passed_for_edit(self, tmp_path):
        _write(tmp_path / ".gitignore", "secret.txt\n")
        _write(tmp_path / "secret.txt", "x")

        modes = []

        async def cb(rel, mode):
            modes.append(mode)
            return True

        result = run(_edit_file("secret.txt", "x", "y", ctx=_ctx(tmp_path, allow_hidden=set(), grant_cb=cb)))

        assert result == "ok"
        assert modes == ["write"]

    def test_workspace_root_path_never_prompts(self, tmp_path):
        _write(tmp_path / "visible.txt", "hello")

        count = 0

        async def cb(rel, mode):
            nonlocal count
            count += 1
            return True

        result = run(_grep("hello", path=".", ctx=_ctx(tmp_path, allow_hidden=set(), grant_cb=cb)))

        assert count == 0
        assert "visible.txt" in result
        assert load_allow_hidden(tmp_path) == set()


class TestHiddenGrantConcurrency:
    def test_concurrent_double_call_not_double_prompted(self, secret_gitignore: Path):
        allow_hidden: set[str] = set()
        pending: set[str] = set()
        count = 0

        async def cb(rel, mode):
            nonlocal count
            count += 1
            await asyncio.sleep(0)
            return True

        async def both():
            ctx = _ctx(secret_gitignore, allow_hidden=allow_hidden, grant_cb=cb, pending=pending)
            return await asyncio.gather(
                _read_file("secret.txt", ctx=ctx),
                _read_file("secret.txt", ctx=ctx),
            )

        result1, result2 = run(both())

        assert count == 1
        results = {result1, result2}
        assert "top secret" in results
        assert "error: access to hidden path denied: secret.txt" in results
        assert pending == set()
        assert allow_hidden == {"secret.txt"}

    def test_concurrent_double_call_read_and_grep_not_double_prompted(self, secret_gitignore: Path):
        allow_hidden: set[str] = set()
        pending: set[str] = set()
        count = 0

        async def cb(rel, mode):
            nonlocal count
            count += 1
            await asyncio.sleep(0)
            return True

        async def both():
            ctx = _ctx(secret_gitignore, allow_hidden=allow_hidden, grant_cb=cb, pending=pending)
            return await asyncio.gather(
                _read_file("secret.txt", ctx=ctx),
                _grep("secret", path="secret.txt", ctx=ctx),
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

    def test_followup_call_after_race_not_reprompted(self, secret_gitignore: Path):
        allow_hidden: set[str] = set()
        pending: set[str] = set()
        count = 0

        async def cb(rel, mode):
            nonlocal count
            count += 1
            await asyncio.sleep(0)
            return True

        async def both():
            ctx = _ctx(secret_gitignore, allow_hidden=allow_hidden, grant_cb=cb, pending=pending)
            return await asyncio.gather(
                _read_file("secret.txt", ctx=ctx),
                _read_file("secret.txt", ctx=ctx),
            )

        run(both())
        assert count == 1

        result = run(_read_file(
            "secret.txt", ctx=_ctx(secret_gitignore, allow_hidden=allow_hidden, grant_cb=cb, pending=pending),
        ))

        assert result == "top secret"
        assert count == 1


# ---------------------------------------------------------------------------
# Directory-as-path guard (file-only tools must reject directory targets)
# ---------------------------------------------------------------------------

class TestDirectoryPathGuard:
    """Passing '.' or any directory path to file-only tools must return a clear
    'is a directory' error rather than an OS error or a grant prompt."""

    def test_read_file_dot_returns_directory_error(self, tmp_path):
        _write(tmp_path / "visible.txt", "hello")
        result = run(_read_file(".", ctx=_ctx(tmp_path)))
        assert "is a directory" in result
        assert result.startswith("error:")

    def test_read_file_subdir_returns_directory_error(self, tmp_path):
        (tmp_path / "sub").mkdir()
        result = run(_read_file("sub", ctx=_ctx(tmp_path)))
        assert "is a directory" in result
        assert result.startswith("error:")

    def test_file_info_dot_returns_directory_error(self, tmp_path):
        _write(tmp_path / "visible.txt", "hello")
        result = run(_file_info(".", ctx=_ctx(tmp_path)))
        assert "is a directory" in result
        assert result.startswith("error:")

    def test_edit_file_dot_returns_directory_error(self, tmp_path):
        result = run(_edit_file(".", "old", "new", ctx=_ctx(tmp_path)))
        assert "is a directory" in result
        assert result.startswith("error:")

    def test_write_file_dot_returns_directory_error(self, tmp_path):
        result = run(_write_file(".", "content", ctx=_ctx(tmp_path)))
        assert "is a directory" in result
        assert result.startswith("error:")

    def test_move_file_src_dir_returns_directory_error(self, tmp_path):
        (tmp_path / "srcdir").mkdir()
        result = run(_move_file("srcdir", "dst.txt", ctx=_ctx(tmp_path)))
        assert "is a directory" in result
        assert result.startswith("error:")

    def test_read_file_dot_does_not_prompt_grant(self, tmp_path):
        _write(tmp_path / "visible.txt", "hello")
        count = 0

        async def cb(rel, mode):
            nonlocal count
            count += 1
            return True

        result = run(_read_file(".", ctx=_ctx(tmp_path, allow_hidden=set(), grant_cb=cb)))
        assert "is a directory" in result
        assert count == 0
