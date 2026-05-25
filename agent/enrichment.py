from __future__ import annotations

import json
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

from agent.manifest_parsers import extract_tech_stack, _parse_csproj
from agent.workspace import _SKIP_DIRS


@dataclass
class EnrichmentResult:
    proj_brief: str
    tech_stack: list[str]
    enriched_at: str
    prompt_tokens: int | None
    completion_tokens: int | None
    domain_map: dict[str, list[str]] = field(default_factory=dict)


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


def _collect_file_list(working_dir: Path, cap: int = 500) -> list[str]:
    paths: list[str] = []
    for root, dirnames, filenames in os.walk(working_dir):
        root_path = Path(root)
        rel_root = root_path.relative_to(working_dir)
        parts = rel_root.parts
        # skip .gekai and _SKIP_DIRS at any level
        if any(p in _SKIP_DIRS or p == ".gekai" for p in parts):
            dirnames.clear()
            continue
        dirnames[:] = [
            d for d in dirnames if d not in _SKIP_DIRS and d != ".gekai"
        ]
        for fname in filenames:
            rel = (rel_root / fname).as_posix()
            paths.append(rel)
            if len(paths) >= cap:
                dirnames.clear()
                return paths
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
    on_file: Callable[[str, int], Awaitable[None]] | None = None,
    on_infer_start: Callable[[], Awaitable[None]] | None = None,
    on_infer_end: Callable[[], Awaitable[None]] | None = None,
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

    prompt = (
        "given the following project excerpts, return exactly one line with no preamble:\n"
        "proj_brief: <one concise sentence describing the project>\n"
        "excerpts:\n"
        f"{combined_snippets}"
    )

    if on_infer_start:
        await on_infer_start()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    if on_infer_end:
        await on_infer_end()

    response_text = response.choices[0].message.content or ""
    proj_brief = ""
    for line in response_text.splitlines():
        if line.startswith("proj_brief:"):
            proj_brief = line[len("proj_brief:"):].strip()

    usage = response.usage
    prompt_tokens: int | None = usage.prompt_tokens if usage else None
    completion_tokens: int | None = usage.completion_tokens if usage else None

    # --- domain_map call ---
    file_list = _collect_file_list(working_dir)
    file_list_text = "\n".join(file_list)
    domain_prompt = (
        "given this list of files from a software repository, group them into functional domain categories\n"
        "output format: one line per domain, exactly `domain_name: path1, path2, ...`\n"
        "use snake_case for domain names (e.g. entry_points, data_access, authentication, tests)\n"
        "if a directory clearly contains many related files, use the directory path with trailing slash instead of listing every file\n"
        "no preamble, no explanation, no markdown\n"
        "files:\n"
        f"{file_list_text}"
    )
    if on_infer_start:
        await on_infer_start()
    domain_response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": domain_prompt}],
        stream=False,
    )
    if on_infer_end:
        await on_infer_end()

    domain_map = _parse_domain_map(domain_response.choices[0].message.content or "")

    enriched_at = datetime.now(timezone.utc).isoformat()

    try:
        existing = json.loads(workspace_json_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}

    existing["proj_brief"] = proj_brief
    existing["tech_stack"] = tech_stack
    existing["domain_map"] = domain_map
    existing["enriched_at"] = enriched_at

    try:
        workspace_json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except OSError:
        pass

    return EnrichmentResult(
        proj_brief=proj_brief,
        tech_stack=tech_stack,
        enriched_at=enriched_at,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        domain_map=domain_map,
    )
