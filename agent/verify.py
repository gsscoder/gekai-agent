from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

_VERIFY_SYSTEM = (
    "you generate minimal pytest tests\n"
    "output only valid Python code — no markdown fences, no explanation\n"
    "one test function per changed callable; names start with test_\n"
    "imports use the module path relative to the workspace root (sys.path is set there)\n"
    "no fixtures, no mocks, no external I/O; purely functional assertions on observable return values\n"
    "if a function has side effects only (writes files, etc.), write a trivial smoke call wrapped in try/except"
)


@dataclass
class VerificationResult:
    passed: bool
    output: str


def _strip_fences(code: str) -> str:
    lines = code.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


async def _generate_test_code(
    file_snippets: list[tuple[str, str]],
    model: str,
    api_key: str | None,
    api_base: str | None,
) -> str:
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    file_blocks = "\n\n".join(f"# {path}\n{content}" for path, content in file_snippets)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _VERIFY_SYSTEM},
            {"role": "user", "content": f"Write a minimal pytest test for the functions in these files:\n\n{file_blocks}"},
        ],
    )
    raw = response.choices[0].message.content or ""
    return _strip_fences(raw.strip())


_PYTEST_TIMEOUT_S = 30


def _run_pytest(test_file: Path, working_dir: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    ws = str(working_dir)
    env["PYTHONPATH"] = ws + os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ws
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-x", "--tb=short", "-q", "--no-header"],
        capture_output=True,
        text=True,
        cwd=ws,
        env=env,
        timeout=_PYTEST_TIMEOUT_S,
    )


async def run_verification(
    working_dir: Path,
    modified_paths: list[str],
    model: str,
    api_key: str | None,
    api_base: str | None,
) -> VerificationResult:
    file_snippets: list[tuple[str, str]] = []
    for path in modified_paths:
        target = (working_dir / path).resolve()
        try:
            file_snippets.append((path, target.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            pass

    if not file_snippets:
        return VerificationResult(passed=True, output="(no readable modified files)")

    test_code = await _generate_test_code(file_snippets, model, api_key, api_base)

    with tempfile.TemporaryDirectory() as tmp:
        test_file = Path(tmp) / "test_verify.py"
        test_file.write_text(test_code, encoding="utf-8")
        try:
            proc = await asyncio.to_thread(_run_pytest, test_file, working_dir)
        except subprocess.TimeoutExpired:
            return VerificationResult(passed=False, output=f"Verification timed out after {_PYTEST_TIMEOUT_S}s")
        output = (proc.stdout + proc.stderr).strip()
        return VerificationResult(passed=proc.returncode == 0, output=output)
