from __future__ import annotations

# dormant — not wired into the workspace scan/db pipeline yet; kept for future re-activation

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

# ---------------------------------------------------------------------------
# Algorithmic tech_stack extraction
# ---------------------------------------------------------------------------

_VERSION_SPECIFIERS = re.compile(r"(>=|<=|!=|==|~=|>|<|@\s*https?://).+")
_EXTRAS = re.compile(r"\[.*?\]")
_INLINE_COMMENT = re.compile(r"\s+#.*$")


def _read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _parse_requirements_txt(path: Path) -> list[str]:
    text = _read_text_or_none(path)
    if text is None:
        return []

    packages: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if re.match(r"^-[rce]\s", line):
            continue
        line = _strip_toml_version(line)
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
    text = _read_text_or_none(path)
    if text is None:
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
    text = _read_text_or_none(path)
    if text is None:
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


def _parse_setup_py(path: Path) -> list[str]:
    text = _read_text_or_none(path)
    if text is None:
        return []

    match = re.search(r"install_requires\s*=\s*\[([^\]]*)\]", text, re.DOTALL)
    if not match:
        return []

    packages: list[str] = []
    for raw in match.group(1).splitlines():
        line = raw.strip().strip(",").strip('"').strip("'").strip()
        if not line or line.startswith("#"):
            continue
        line = _VERSION_SPECIFIERS.sub("", line)
        line = _EXTRAS.sub("", line)
        line = line.strip()
        if line:
            packages.append(line)
    return packages


def _parse_go_mod(path: Path) -> list[str]:
    text = _read_text_or_none(path)
    if text is None:
        return []

    packages: list[str] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("require ("):
            in_block = True
            continue
        if in_block:
            if stripped == ")":
                in_block = False
                continue
            if not stripped or stripped.startswith("//"):
                continue
            module = stripped.split()[0]
            if module:
                packages.append(module)
        elif stripped.startswith("require ") and not stripped.endswith("("):
            parts = stripped.split()
            if len(parts) >= 2:
                packages.append(parts[1])
    return packages


def _parse_pom_xml(path: Path) -> list[str]:
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError):
        return []

    packages: list[str] = []
    for elem in tree.iter():
        if elem.tag.split("}")[-1] == "dependency":
            group_id = None
            artifact_id = None
            for child in elem:
                tag = child.tag.split("}")[-1]
                if tag == "groupId":
                    group_id = (child.text or "").strip()
                elif tag == "artifactId":
                    artifact_id = (child.text or "").strip()
            if group_id and artifact_id:
                packages.append(f"{group_id}:{artifact_id}")
    return packages


_GRADLE_DEP = re.compile(
    r"""^\s*(?:implementation|api|compile|testImplementation|testApi|runtimeOnly|compileOnly)\s+['"]([^'"]+)['"]""",
)


def _parse_build_gradle(path: Path) -> list[str]:
    text = _read_text_or_none(path)
    if text is None:
        return []

    packages: list[str] = []
    for line in text.splitlines():
        m = _GRADLE_DEP.match(line)
        if m:
            packages.append(m.group(1))
    return packages


_MANIFEST_PARSERS: dict[str, Callable[[Path], list[str]]] = {
    "requirements.txt": _parse_requirements_txt,
    "package.json": _parse_package_json,
    "pyproject.toml": _parse_pyproject_toml,
    "cargo.toml": _parse_cargo_toml,
    ".csproj": _parse_csproj,
    "setup.py": _parse_setup_py,
    "go.mod": _parse_go_mod,
    "pom.xml": _parse_pom_xml,
    "build.gradle": _parse_build_gradle,
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
