import json

from agent.settings import load_allow_hidden, load_external, save_allow_hidden, save_external, save_permissions
from agent.permissions import Permissions


# ---------------------------------------------------------------------------
# load_allow_hidden / save_allow_hidden
# ---------------------------------------------------------------------------

class TestAllowHidden:
    def test_load_allow_hidden_empty_when_no_settings_file(self, tmp_path):
        assert load_allow_hidden(tmp_path) == set()

    def test_save_and_load_roundtrip(self, tmp_path):
        save_allow_hidden(tmp_path, "secret.txt")
        assert load_allow_hidden(tmp_path) == {"secret.txt"}

    def test_save_appends_without_duplicating(self, tmp_path):
        save_allow_hidden(tmp_path, "a.txt")
        save_allow_hidden(tmp_path, "a.txt")
        save_allow_hidden(tmp_path, "b.txt")

        assert load_allow_hidden(tmp_path) == {"a.txt", "b.txt"}

        path = tmp_path / ".gekai" / "settings.local.json"
        data = json.loads(path.read_text())
        allow_hidden = data["permissions"]["allow_hidden"]
        assert allow_hidden.count("a.txt") == 1

    def test_save_preserves_other_settings_keys(self, tmp_path):
        save_permissions(tmp_path, Permissions(read=True, write=False, exec=False))
        save_allow_hidden(tmp_path, "secret.txt")

        path = tmp_path / ".gekai" / "settings.local.json"
        data = json.loads(path.read_text())

        assert data["permissions"]["workspace"] == {
            "read": "allow",
            "write": "deny",
            "exec": "deny",
        }
        assert data["permissions"]["allow_hidden"] == ["secret.txt"]


# ---------------------------------------------------------------------------
# load_external / save_external
# ---------------------------------------------------------------------------

class TestExternal:
    def test_load_external_empty_when_no_settings_file(self, tmp_path):
        assert load_external(tmp_path) == []

    def test_save_and_load_roundtrip(self, tmp_path):
        other = tmp_path.parent / "other-project"
        save_external(tmp_path, other)
        assert load_external(tmp_path) == [other.resolve()]

    def test_save_noop_when_already_covered_by_existing_entry(self, tmp_path):
        parent = tmp_path.parent / "repo"
        child = parent / "src" / "sub"
        save_external(tmp_path, parent)
        save_external(tmp_path, child)

        assert load_external(tmp_path) == [parent.resolve()]

    def test_save_replaces_narrower_entry_subsumed_by_new_root(self, tmp_path):
        parent = tmp_path.parent / "repo"
        child = parent / "src" / "sub"
        save_external(tmp_path, child)
        save_external(tmp_path, parent)

        assert load_external(tmp_path) == [parent.resolve()]

    def test_save_does_not_falsely_match_sibling_with_shared_prefix(self, tmp_path):
        foo = tmp_path.parent / "foo"
        foobar = tmp_path.parent / "foobar"
        save_external(tmp_path, foo)
        save_external(tmp_path, foobar)

        assert set(load_external(tmp_path)) == {foo.resolve(), foobar.resolve()}
