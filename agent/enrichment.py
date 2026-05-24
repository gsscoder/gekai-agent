from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Algorithmic tech_stack extraction
# ---------------------------------------------------------------------------

_VERSION_SPECIFIERS = re.compile(r"(>=|<=|!=|==|~=|>|<|@\s*https?://).+")
_EXTRAS = re.compile(r"\[.*?\]")
_INLINE_COMMENT = re.compile(r"\s+#.*$")


def _parse_requirements_txt(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []

    packages: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if re.match(r"^-[rce]\s", line):
            continue
        line = _INLINE_COMMENT.sub("", line)
        line = _VERSION_SPECIFIERS.sub("", line)
        line = _EXTRAS.sub("", line)
        line = line.strip()
        if line:
            packages.append(line)
    return packages


def _parse_package_json(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    deps: list[str] = []
    deps.extend(data.get("dependencies", {}).keys())
    deps.extend(data.get("devDependencies", {}).keys())
    return deps


def _strip_toml_version(value: str) -> str:
    value = _INLINE_COMMENT.sub("", value)
    value = _VERSION_SPECIFIERS.sub("", value)
    value = _EXTRAS.sub("", value)
    return value.strip()


def _parse_pyproject_toml(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []

    deps: list[str] = []
    in_deps = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_deps = stripped.lower() in (
                "[project.dependencies]",
                "[tool.poetry.dependencies]",
            )
            continue
        if not in_deps or not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            name = stripped.split("=")[0].strip().strip('"').strip("'")
        else:
            name = _strip_toml_version(stripped.strip('"').strip("'"))
        name = _EXTRAS.sub("", name).strip()
        if name:
            deps.append(name)
    return deps


def _parse_cargo_toml(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []

    deps: list[str] = []
    in_deps = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_deps = stripped.lower() == "[dependencies]"
            continue
        if not in_deps or not stripped or stripped.startswith("#"):
            continue
        name = stripped.split("=")[0].strip().strip('"').strip("'")
        if name:
            deps.append(name)
    return deps


def _parse_csproj(path: Path) -> list[str]:
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError):
        return []

    packages: list[str] = []
    for elem in tree.iter():
        if elem.tag.split("}")[-1] == "PackageReference":
            name = elem.get("Include")
            if name:
                packages.append(name)
    return packages


_MANIFEST_PARSERS: dict[str, Callable[[Path], list[str]]] = {
    "requirements.txt": _parse_requirements_txt,
    "package.json": _parse_package_json,
    "pyproject.toml": _parse_pyproject_toml,
    "cargo.toml": _parse_cargo_toml,
    ".csproj": _parse_csproj,
}


def extract_tech_stack(manifest_path: Path, lang: str) -> list[str]:
    parser = _MANIFEST_PARSERS.get(manifest_path.name.lower()) or _MANIFEST_PARSERS.get(manifest_path.suffix.lower())
    deps = parser(manifest_path) if parser is not None else []
    seen: set[str] = set()
    result: list[str] = [lang]
    seen.add(lang.lower())
    for dep in deps:
        if dep.lower() not in seen:
            seen.add(dep.lower())
            result.append(dep)
    return result


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

    enriched_at = datetime.now(timezone.utc).isoformat()

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
