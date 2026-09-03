import json

from agent.settings import load_allow_hidden, save_allow_hidden, save_permissions
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
