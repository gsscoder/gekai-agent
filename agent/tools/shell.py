from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path

from agent.llm import tool
from agent.shell import resolve_shell
from agent.workspace.ignore import IgnoreRules, load as _load_ignore_rules

_MAX_OUTPUT_CHARS = 20_000


class ShellCommandError(Exception):
    """Raised when a shell command exits nonzero, so the tool-execution
    wrapper marks the result is_error=True instead of the failure reading
    as ordinary text."""


def _forbidden_token(command: str, working_dir: Path, rules: IgnoreRules) -> str | None:
    """Best-effort scan: does any bare-path-looking token in `command` resolve
    to a .aiignore-forbidden path under working_dir?

    Not airtight (can't catch every obfuscation - encoded paths, env var
    expansion, etc.) but stops the common case of an agent catting/grepping a
    red-zone file via run_command instead of the file tools.
    """
    root = working_dir.resolve()
    try:
        raw_tokens = shlex.split(command, posix=False)
    except ValueError:
        raw_tokens = command.split()

    candidates: list[str] = []
    for tok in raw_tokens:
        tok = tok.strip("'\"")
        candidates.append(tok)
        if "=" in tok:
            candidates.append(tok.split("=", 1)[1])

    for tok in candidates:
        if not tok or tok.startswith("-"):
            continue
        try:
            p = (root / tok).resolve()
        except OSError:
            continue
        if not p.is_relative_to(root):
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        if rules.is_forbidden(rel) or rules.is_forbidden(rel + "/"):
            return rel
    return None


def _venv_env(working_dir: Path) -> dict[str, str] | None:
    """If `working_dir` has a `.venv`, return an env dict with its executables
    directory prepended to PATH (and VIRTUAL_ENV set) — exactly what a venv
    `activate` script does. Returns None (no-op) if there's no usable `.venv`,
    so the subprocess inherits `os.environ` as-is.
    """
    venv_dir = working_dir / ".venv"
    if not venv_dir.is_dir():
        return None

    bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    if not bin_dir.is_dir():
        return None

    env = dict(os.environ)
    existing_path = env.get("PATH", "")
    env["PATH"] = str(bin_dir) + (os.pathsep + existing_path if existing_path else "")
    env["VIRTUAL_ENV"] = str(venv_dir)
    return env


async def _run_command(
    command: str,
    *,
    working_dir: Path,
    timeout: int = 30,
) -> str:
    rules = _load_ignore_rules(working_dir)
    hit = _forbidden_token(command, working_dir, rules)
    if hit is not None:
        return f"error: command references a restricted path: {hit}"

    spec = resolve_shell()
    env = _venv_env(working_dir)
    try:
        proc = await asyncio.create_subprocess_exec(
            spec.path,
            *spec.build_args(command),
            cwd=working_dir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                # wait_for cancels communicate() mid-flight — drain pipes so Windows
                # proactor transports close before the event loop shuts down
                try:
                    await proc.communicate()
                except ProcessLookupError:
                    pass
                return f"error: timeout after {timeout}s"
        finally:
            # ponytail: Windows ProactorEventLoop closes subprocess pipe
            # transports via a deferred loop callback; without this tick a
            # transport can still be pending when the app's own loop shuts
            # down (e.g. on /exit), and Python's GC finalizing it later
            # prints a harmless "Exception ignored in __del__" to stderr.
            # Force-close + one loop tick runs cleanup now instead of
            # racing the app's shutdown.
            transport = getattr(proc, "_transport", None)
            if transport is not None:
                transport.close()
            await asyncio.sleep(0)
    except Exception as exc:
        return f"error: {exc or type(exc).__name__}"

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    parts = []
    if stdout.strip():
        parts.append(stdout)
    if stderr.strip():
        parts.append(stderr)
    combined = "\n".join(parts)
    if proc.returncode != 0:
        combined = (combined + f"\nexit: {proc.returncode}").strip()
        if len(combined) > _MAX_OUTPUT_CHARS:
            combined = combined[:_MAX_OUTPUT_CHARS] + f"\n... (truncated at {_MAX_OUTPUT_CHARS} chars)"
        raise ShellCommandError(combined if combined else "(no output)")
    if len(combined) > _MAX_OUTPUT_CHARS:
        combined = combined[:_MAX_OUTPUT_CHARS] + f"\n... (truncated at {_MAX_OUTPUT_CHARS} chars)"
    return combined if combined else "(no output)"


def make_shell_tools(working_dir: Path) -> list:
    @tool(is_read_only=False, required_permission="exec", is_concurrency_safe=False)
    async def run_command(command: str, timeout: int = 30) -> str:
        """Run a shell command in the workspace root.

        Native shell: PowerShell on Windows, $SHELL/bash/sh on Unix. Resolved once at startup.
        Stateless: each call starts in workspace root — cd does NOT persist across calls.

        IMPORTANT: do NOT use this tool to read files, search content, or list directories.
        Use the dedicated tools instead: read_file (not cat), grep (not grep/rg/findstr),
        list_files (not ls/find/dir), symbols (not ctags).
        Reserve run_command for: build systems, test runners, git, package managers,
        and anything with no dedicated tool.

        Paths listed in .aiignore are off-limits even via shell commands (best-effort
        check, not airtight against obfuscation).
        """
        return await _run_command(command, working_dir=working_dir, timeout=timeout)

    return [run_command]
