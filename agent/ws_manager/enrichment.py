from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

from ..manifest_parsers import extract_tech_stack, _parse_csproj
from ..persistence import now_utc_str
from ..workspace import _SKIP_DIRS


@dataclass
class EnrichmentResult:
    proj_brief: str
    tech_stack: list[str]
    prompt_tokens: int | None
    completion_tokens: int | None
    domain_map: dict[str, list[str]] = field(default_factory=dict)


def _get_git_state(working_dir: Path) -> tuple[str | None, bool]:
    try:
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working_dir, capture_output=True, text=True, timeout=5,
        )
        if hash_result.returncode != 0:
            return None, False
        commit_hash = hash_result.stdout.strip()
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=working_dir, capture_output=True, text=True, timeout=5,
        )
        dirty = bool(status_result.stdout.strip())
        return commit_hash, dirty
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None, False



def _extract_manifest_snippet(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    name = path.name.lower()

    if name == "package.json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        parts: list[str] = []
        desc = data.get("description")
        if desc:
            parts.append(f"description: {desc}")
        deps = list(data.get("dependencies", {}).keys())
        if deps:
            parts.append(f"dependencies: {', '.join(deps)}")
        return "\n".join(parts) if parts else None

    if path.suffix.lower() == ".csproj":
        pkgs = _parse_csproj(path)
        return f"dependencies: {', '.join(pkgs)}" if pkgs else None

    if name in ("pyproject.toml", "cargo.toml"):
        lines = text.splitlines()
        description: str | None = None
        deps: list[str] = []
        in_project = False
        in_deps = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("["):
                header = stripped.lower()
                in_project = header in (
                    "[project]",
                    "[tool.poetry]",
                    "[package]",
                )
                in_deps = header in (
                    "[project.dependencies]",
                    "[tool.poetry.dependencies]",
                    "[dependencies]",
                )
                continue

            if in_project and description is None:
                m = re.match(r'^description\s*=\s*["\'](.+?)["\']', stripped)
                if m:
                    description = m.group(1)

            if in_deps and stripped and not stripped.startswith("#"):
                dep_name = stripped.split("=")[0].strip().strip('"').strip("'")
                if dep_name:
                    deps.append(dep_name)

        parts = []
        if description:
            parts.append(f"description: {description}")
        if deps:
            parts.append(f"dependencies: {', '.join(deps)}")
        return "\n".join(parts) if parts else None

    return None


def _extract_doc_snippets(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    lines = text.splitlines()
    known_pattern = re.compile(
        r"^#{1,2}\s*(project brief|core technologies|tech stack|technology stack|overview|about|description|what is)",
        re.IGNORECASE,
    )

    first_para_lines: list[str] = []
    first_title_found = False
    first_para_done = False
    for line in lines:
        if not first_title_found:
            if line.startswith("#"):
                first_title_found = True
            continue
        if first_para_done:
            break
        if line.startswith("##"):
            first_para_done = True
            break
        if not line.strip() and not first_para_lines:
            continue
        first_para_lines.append(line)

    first_paragraph = "\n".join(first_para_lines).strip()

    sections: list[str] = []
    i = 0
    while i < len(lines):
        if known_pattern.match(lines[i]):
            section_lines = [lines[i]]
            i += 1
            while i < len(lines):
                if lines[i].startswith("##") and i > 0:
                    break
                section_lines.append(lines[i])
                i += 1
            sections.append("\n".join(section_lines).strip())
        else:
            i += 1

    if not sections:
        fallback = first_paragraph
        extra = "\n".join(lines[:60]).strip()
        combined = f"{fallback}\n{extra}".strip() if fallback != extra else fallback
        return combined if combined else None

    result_parts: list[str] = []
    if first_paragraph:
        result_parts.append(first_paragraph)
    for sec in sections:
        if sec not in result_parts:
            result_parts.append(sec)

    combined = "\n\n".join(result_parts).strip()
    return combined if combined else None


_CODE_SUFFIXES: frozenset[str] = frozenset({
    ".py", ".pyi",
    ".js", ".mjs", ".cjs", ".jsx",
    ".ts", ".tsx", ".mts", ".cts",
    ".css", ".scss", ".sass", ".less",
    ".html", ".htm",
    ".go",
    ".rs",
    ".java",
    ".kt", ".kts",
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx",
    ".rb",
    ".php",
    ".swift",
    ".sh", ".bash",
    ".sql",
    ".vue",
    ".svelte",
    ".dart",
    ".ex", ".exs",
    ".lua",
    ".scala",
    ".r",
})
_COLLAPSE_THRESHOLD: int = 5
_DEPTH_CAP: int = 3


def _collect_file_list(working_dir: Path) -> list[str]:
    paths: list[str] = []
    for root, dirnames, filenames in os.walk(working_dir):
        root_path = Path(root)
        rel_root = root_path.relative_to(working_dir)
        parts = rel_root.parts

        if any(p in _SKIP_DIRS or p == ".gekai" for p in parts):
            dirnames.clear()
            continue

        if len(parts) > _DEPTH_CAP:
            dirnames.clear()
            continue

        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and d != ".gekai" and not d.startswith(".")
        ]

        # keep only code files, skip hidden files
        candidates: list[str] = []
        for fname in filenames:
            if fname.startswith("."):
                continue
            if Path(fname).suffix in _CODE_SUFFIXES:
                candidates.append(fname)

        # auto-collapse: group by extension
        ext_groups: dict[str, list[str]] = {}
        for fname in candidates:
            ext_groups.setdefault(Path(fname).suffix, []).append(fname)

        dir_posix = rel_root.as_posix() if rel_root.parts else ""
        dir_prefix = (dir_posix + "/") if dir_posix else ""

        for ext, group in ext_groups.items():
            if len(group) > _COLLAPSE_THRESHOLD:
                paths.append(f"{dir_prefix}({len(group)} {ext} files)")
            else:
                for fname in group:
                    rel = (rel_root / fname).as_posix() if rel_root.parts else fname
                    paths.append(rel)

    return paths


def _parse_domain_map(response_text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for line in response_text.splitlines():
        if ":" not in line:
            continue
        domain, _, rest = line.partition(":")
        domain = domain.strip()
        if not domain:
            continue
        paths = [p.strip() for p in rest.split(",") if p.strip()]
        if paths:
            result[domain] = paths
    return result


def _count_lines(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return 0


async def enrich_workspace(
    working_dir: Path,
    client: AsyncOpenAI,
    model: str,
    commit_hash: str | None,
    dirty: bool,
    on_file: Callable[[str, int], Awaitable[None]] | None = None,
    on_infer_end: Callable[[int, int], Awaitable[None]] | None = None,
) -> EnrichmentResult:
    async def _notify(path: Path) -> None:
        if on_file:
            await on_file(str(path.relative_to(working_dir)), _count_lines(path))

    workspace_json_path = working_dir / ".gekai" / "workspace.json"
    try:
        workspace = json.loads(workspace_json_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        workspace = {}

    snippets: list[str] = []
    notified: set[Path] = set()

    for proj in workspace.get("projects", []):
        manifest = proj.get("manifest")
        proj_path = proj.get("path", ".")
        if not manifest:
            continue
        manifest_path = working_dir / manifest if proj_path == "." else working_dir / proj_path / manifest
        snippet = _extract_manifest_snippet(manifest_path)
        if snippet:
            await _notify(manifest_path)
            notified.add(manifest_path)
            snippets.append(f"[{manifest}]\n{snippet}")

    for rel in workspace.get("ai_instructions", []):
        doc_path = working_dir / rel
        snippet = _extract_doc_snippets(doc_path)
        if snippet:
            await _notify(doc_path)
            snippets.append(f"[{rel}]\n{snippet}")

    readme_path = working_dir / "README.md"
    if readme_path.exists():
        snippet = _extract_doc_snippets(readme_path)
        if snippet:
            await _notify(readme_path)
            snippets.append(f"[README.md]\n{snippet}")

    tech_stack: list[str] = []
    seen_langs: set[str] = set()
    for proj in workspace.get("projects", []):
        manifest = proj.get("manifest")
        lang = proj.get("lang", "")
        proj_path = proj.get("path", ".")
        if not manifest or not lang:
            continue
        manifest_path = (
            working_dir / manifest
            if proj_path == "."
            else working_dir / proj_path / manifest
        )
        if manifest_path not in notified:
            await _notify(manifest_path)
            notified.add(manifest_path)
        for entry in extract_tech_stack(manifest_path, lang):
            key = entry.lower()
            if key not in seen_langs:
                seen_langs.add(key)
                tech_stack.append(entry)

    combined_snippets = "\n\n".join(snippets) if snippets else "no project files found"

    async def _infer_brief() -> tuple[str, int, int]:
        prompt = (
            "given the following project excerpts, return exactly one line with no preamble:\n"
            "proj_brief: <one concise sentence describing the project>\n"
            "excerpts:\n"
            f"{combined_snippets}"
        )
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
        )
        text = ""
        p_tokens = 0
        c_tokens = 0
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
            if chunk.usage:
                p_tokens = chunk.usage.prompt_tokens or 0
                c_tokens = chunk.usage.completion_tokens or 0
        brief = ""
        for line in text.splitlines():
            if line.startswith("proj_brief:"):
                brief = line[len("proj_brief:"):].strip()
        if on_infer_end:
            await on_infer_end(p_tokens, c_tokens)
        return brief, p_tokens, c_tokens

    async def _infer_domain() -> tuple[dict[str, list[str]], int, int]:
        file_list = _collect_file_list(working_dir)
        file_list_text = "\n".join(file_list)
        domain_prompt = (
            "given this list of files from a software repository, group them into functional domain categories\n"
            "output format: one line per domain, exactly `domain_name: path1, path2, ...`\n"
            "use snake_case for domain names (e.g. entry_points, data_access, authentication, tests)\n"
            "no preamble, no explanation, no markdown\n"
            "files:\n"
            f"{file_list_text}"
        )
        domain_stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": domain_prompt}],
            stream=True,
            stream_options={"include_usage": True},
        )
        text = ""
        p_tokens = 0
        c_tokens = 0
        async for chunk in domain_stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
            if chunk.usage:
                p_tokens = chunk.usage.prompt_tokens or 0
                c_tokens = chunk.usage.completion_tokens or 0
        if on_infer_end:
            await on_infer_end(p_tokens, c_tokens)
        return _parse_domain_map(text), p_tokens, c_tokens

    (proj_brief, call1_prompt, call1_completion), (domain_map, call2_prompt, call2_completion) = (
        await asyncio.gather(_infer_brief(), _infer_domain())
    )

    try:
        existing = json.loads(workspace_json_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}

    existing["proj_brief"] = proj_brief
    existing["tech_stack"] = tech_stack
    existing["domain_map"] = domain_map
    existing["scan_state"] = {
        "timestamp": now_utc_str(),
        "commit_hash": commit_hash,
        "uncommitted": dirty,
    }
    existing.pop("enriched_at", None)

    try:
        workspace_json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except OSError:
        pass

    return EnrichmentResult(
        proj_brief=proj_brief,
        tech_stack=tech_stack,
        prompt_tokens=call1_prompt + call2_prompt,
        completion_tokens=call1_completion + call2_completion,
        domain_map=domain_map,
    )
