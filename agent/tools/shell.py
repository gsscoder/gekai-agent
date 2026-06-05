from __future__ import annotations

import asyncio
from pathlib import Path

from agent.llm import tool
from agent.shell import resolve_shell

_MAX_OUTPUT_CHARS = 20_000


async def _run_command(
    command: str,
    *,
    working_dir: Path,
    timeout: int = 30,
) -> str:
    spec = resolve_shell()
    try:
        proc = await asyncio.create_subprocess_exec(
            spec.path,
            *spec.build_args(command),
            cwd=working_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return f"error: timeout after {timeout}s"
    except Exception as exc:
        return f"error: {exc}"

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
    return combined if combined else "(no output)"


def make_shell_tools(working_dir: Path) -> list:
    @tool(is_read_only=False, required_permission="exec", is_concurrency_safe=False)
    async def run_command(command: str, timeout: int = 30) -> str:
        """Run a shell command in the repository root.

        Native shell: PowerShell on Windows, $SHELL/bash/sh on Unix. Resolved once at startup.
        Stateless: each call starts in repo root — cd does NOT persist across calls.

        IMPORTANT: do NOT use this tool to read files, search content, or list directories.
        Use the dedicated tools instead: read_file (not cat), grep (not grep/rg/findstr),
        list_files (not ls/find/dir), symbols (not ctags).
        Reserve run_command for: build systems, test runners, git, package managers,
        and anything with no dedicated tool.
        """
        return await _run_command(command, working_dir=working_dir, timeout=timeout)

    return [run_command]
