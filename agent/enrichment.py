from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI


@dataclass
class EnrichmentResult:
    proj_brief: str
    tech_stack: list[str]
    enriched_at: str
    prompt_tokens: int | None
    completion_tokens: int | None


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


async def enrich_workspace(
    working_dir: Path,
    client: AsyncOpenAI,
    model: str,
    on_step: Callable[[str], None] | None = None,
) -> EnrichmentResult:
    def _step(msg: str) -> None:
        if on_step:
            on_step(msg)

    workspace_json_path = working_dir / ".gekai" / "workspace.json"
    try:
        workspace = json.loads(workspace_json_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        workspace = {}

    snippets: list[str] = []

    _step("Reading manifests")
    for proj in workspace.get("projects", []):
        manifest = proj.get("manifest")
        proj_path = proj.get("path", ".")
        if not manifest:
            continue
        if proj_path == ".":
            manifest_path = working_dir / manifest
        else:
            manifest_path = working_dir / proj_path / manifest
        snippet = _extract_manifest_snippet(manifest_path)
        if snippet:
            snippets.append(f"[{manifest}]\n{snippet}")

    _step("Reading instruction files")
    for rel in workspace.get("ai_instructions", []):
        doc_path = working_dir / rel
        snippet = _extract_doc_snippets(doc_path)
        if snippet:
            snippets.append(f"[{rel}]\n{snippet}")

    _step("Reading README")
    readme_path = working_dir / "README.md"
    if readme_path.exists():
        snippet = _extract_doc_snippets(readme_path)
        if snippet:
            snippets.append(f"[README.md]\n{snippet}")

    combined_snippets = "\n\n".join(snippets) if snippets else "no project files found"

    _step("Synthesizing")
    prompt = (
        "given the following project excerpts, return exactly two lines with no preamble:\n"
        "proj_brief: <one concise sentence describing the project>\n"
        "tech_stack: <comma-separated list of core technologies>\n"
        "excerpts:\n"
        f"{combined_snippets}"
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    response_text = response.choices[0].message.content or ""
    proj_brief = ""
    tech_stack_raw = ""
    for line in response_text.splitlines():
        if line.startswith("proj_brief:"):
            proj_brief = line[len("proj_brief:"):].strip()
        elif line.startswith("tech_stack:"):
            tech_stack_raw = line[len("tech_stack:"):].strip()

    tech_stack = [item.strip() for item in tech_stack_raw.split(",") if item.strip()]

    usage = response.usage
    prompt_tokens: int | None = usage.prompt_tokens if usage else None
    completion_tokens: int | None = usage.completion_tokens if usage else None

    enriched_at = datetime.now(timezone.utc).isoformat()

    _step("Saving")
    try:
        existing = json.loads(workspace_json_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}

    existing["proj_brief"] = proj_brief
    existing["tech_stack"] = tech_stack
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
    )
