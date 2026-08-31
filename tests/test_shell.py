from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

import pytest

from agent.shell import ShellSpec, resolve_shell
from agent.tools import _run_command
from agent.tools.shell import ShellCommandError, _forbidden_token, _venv_env
from agent.workspace.ignore import load as _load_ignore_rules


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# ShellSpec.build_args
# ---------------------------------------------------------------------------

class TestShellSpec:
    def test_powershell_args(self) -> None:
        spec = ShellSpec("powershell", "pwsh")
        assert spec.build_args("echo hi") == [
            "-NoProfile", "-NonInteractive", "-Command", "echo hi"
        ]

    def test_posix_args(self) -> None:
        spec = ShellSpec("bash", "/bin/bash")
        assert spec.build_args("echo hi") == ["-lc", "echo hi"]


# ---------------------------------------------------------------------------
# resolve_shell
# ---------------------------------------------------------------------------

class TestResolveShell:
    @pytest.fixture(autouse=True)
    def _clear_shell_cache(self):
        resolve_shell.cache_clear()
        yield
        resolve_shell.cache_clear()

    def test_windows_prefers_pwsh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/pwsh" if name == "pwsh" else None)
        spec = resolve_shell()
        assert spec.kind == "powershell"
        assert spec.path == "/usr/bin/pwsh"

    def test_windows_falls_back_to_powershell_exe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "win32")

        def _which(name: str) -> str | None:
            if name == "pwsh":
                return None
            if name == "powershell":
                return r"C:\Windows\System32\powershell.exe"
            return None

        monkeypatch.setattr(shutil, "which", _which)
        spec = resolve_shell()
        assert spec.kind == "powershell"
        assert spec.path == r"C:\Windows\System32\powershell.exe"

    def test_windows_raises_if_no_shell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(ValueError):
            resolve_shell()

    def test_posix_uses_shell_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("SHELL", "/bin/zsh")
        spec = resolve_shell()
        assert spec.kind == "sh"
        assert spec.path == "/bin/zsh"

    def test_posix_falls_back_to_bash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("SHELL", raising=False)

        def _which(name: str) -> str | None:
            if name == "bash":
                return "/usr/bin/bash"
            return None

        monkeypatch.setattr(shutil, "which", _which)
        spec = resolve_shell()
        assert spec.kind == "bash"
        assert spec.path == "/usr/bin/bash"

    def test_posix_falls_back_to_sh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("SHELL", raising=False)

        def _which(name: str) -> str | None:
            if name == "sh":
                return "/bin/sh"
            return None

        monkeypatch.setattr(shutil, "which", _which)
        spec = resolve_shell()
        assert spec.kind == "sh"
        assert spec.path == "/bin/sh"

    def test_posix_raises_if_no_shell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("SHELL", raising=False)
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(ValueError):
            resolve_shell()


# ---------------------------------------------------------------------------
# _run_command
# ---------------------------------------------------------------------------

def _echo_cmd() -> str:
    return "echo hello" if sys.platform != "win32" else "Write-Host hello"


def _sleep_cmd() -> str:
    return "Start-Sleep 10" if sys.platform == "win32" else "sleep 10"


def _pwd_cmd() -> str:
    return "(Get-Location).Path" if sys.platform == "win32" else "pwd"


class TestRunCommand:
    def test_success(self, tmp_path: Path) -> None:
        result = run(_run_command(_echo_cmd(), working_dir=tmp_path))
        assert "hello" in result

    def test_nonzero_exit(self, tmp_path: Path) -> None:
        with pytest.raises(ShellCommandError) as exc_info:
            run(_run_command("exit 1", working_dir=tmp_path))
        assert "exit: 1" in str(exc_info.value)

    def test_timeout(self, tmp_path: Path) -> None:
        result = run(_run_command(_sleep_cmd(), working_dir=tmp_path, timeout=1))
        assert result == "error: timeout after 1s"

    def test_working_dir_is_cwd(self, tmp_path: Path) -> None:
        result = run(_run_command(_pwd_cmd(), working_dir=tmp_path))
        assert str(tmp_path).lower() in result.lower()


# ---------------------------------------------------------------------------
# _forbidden_token / .aiignore red zone
# ---------------------------------------------------------------------------

class TestForbiddenToken:
    @pytest.mark.parametrize(
        "command, expected",
        [("echo secret.txt", "secret.txt"), ("echo hello world", None)],
    )
    def test_forbidden_token(self, tmp_path: Path, command: str, expected: str | None) -> None:
        (tmp_path / ".aiignore").write_text("secret.txt\n")
        rules = _load_ignore_rules(tmp_path)
        assert _forbidden_token(command, tmp_path, rules) == expected


class TestVenvEnv:
    def test_no_venv_returns_none(self, tmp_path: Path) -> None:
        assert _venv_env(tmp_path) is None

    def test_venv_prepends_bin_dir_to_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bin_name = "Scripts" if os.name == "nt" else "bin"
        bin_dir = tmp_path / ".venv" / bin_name
        bin_dir.mkdir(parents=True)
        monkeypatch.setenv("PATH", r"C:\some\other\path" if os.name == "nt" else "/some/other/path")

        env = _venv_env(tmp_path)

        assert env is not None
        assert env["PATH"].startswith(str(bin_dir) + os.pathsep)
        assert env["PATH"].endswith(os.environ["PATH"])
        assert env["VIRTUAL_ENV"] == str(tmp_path / ".venv")


class TestRunCommandRedZone:
    def test_run_command_denies_forbidden_path(self, tmp_path: Path) -> None:
        (tmp_path / ".aiignore").write_text("secret.txt\n")
        result = run(_run_command("echo secret.txt", working_dir=tmp_path))
        assert result.startswith("error: command references a restricted path:")
        assert "secret.txt" in result

    def test_run_command_normal_still_works(self, tmp_path: Path) -> None:
        result = run(_run_command("echo hello", working_dir=tmp_path))
        assert "hello" in result
