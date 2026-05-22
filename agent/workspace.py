from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from collections.abc import Callable
from pathlib import Path

# Manifest files → language, in priority order (highest first)
_MANIFEST_PRIORITY: list[tuple[str, str]] = [
    ("*.csproj", "C#"),
    ("*.sln", "C#"),
    ("pyproject.toml", "Python"),
    ("setup.py", "Python"),
    ("requirements.txt", "Python"),
    ("package.json", "JavaScript/TypeScript"),
    ("go.mod", "Go"),
    ("Cargo.toml", "Rust"),
    ("pom.xml", "Java"),
    ("build.gradle", "Java"),
]

_EXT_TO_LANG: dict[str, str] = {
    ".py": "Python",
    ".cs": "C#",
    ".ts": "JavaScript/TypeScript",
    ".js": "JavaScript/TypeScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".ps1": "PowerShell",
    ".sh": "Shell",
}

_SKIP_DIRS: frozenset[str] = frozenset(
    {".git", "node_modules", ".venv", "__pycache__", "bin", "obj"}
)

_AI_INSTRUCTION_FILES: list[str] = [
    "CLAUDE.md",
    "AGENTS.md",
    ".cursorrules",
    ".windsurfrules",
    ".clinerules",
    "SYSTEM_PROMPT.md",
    ".aider.conf.yml",
    "copilot-instructions.md",
]

_AI_INSTRUCTION_SPECIAL: list[str] = [
    ".github/copilot-instructions.md",
]


def _scan_ai_instructions(working_dir: Path) -> list[str]:
    found: list[str] = []
    for fname in _AI_INSTRUCTION_FILES:
        if (working_dir / fname).exists():
            found.append(fname)
    for special in _AI_INSTRUCTION_SPECIAL:
        if (working_dir / special).exists():
            found.append(special)
    for subdir in working_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith(".") and subdir.name not in _SKIP_DIRS:
            for fname in _AI_INSTRUCTION_FILES:
                p = subdir / fname
                if p.exists():
                    found.append(str(p.relative_to(working_dir)).replace("\\", "/"))
    return found


def _get_repo_name(working_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=working_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            segment = url.rstrip("/").split("/")[-1]
            return segment.removesuffix(".git") if segment else working_dir.name
    except FileNotFoundError:
        pass
    return working_dir.name


def _walk(working_dir: Path):
    """Yield all (dir, files) pairs, skipping _SKIP_DIRS."""
    for dirpath, dirnames, filenames in os.walk(working_dir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        yield Path(dirpath), filenames


def _scan_projects(working_dir: Path) -> list[dict]:
    projects: list[dict] = []
    # Track which dirs already have a language assigned and at what priority index
    dir_priority: dict[Path, int] = {}

    for dirpath, filenames in _walk(working_dir):
        fileset = set(filenames)
        for priority_idx, (pattern, lang) in enumerate(_MANIFEST_PRIORITY):
            # Glob-style match: patterns with * need fnmatch, others are exact
            if "*" in pattern:
                import fnmatch
                matched = next((f for f in filenames if fnmatch.fnmatch(f, pattern)), None)
                manifest_name = matched
            else:
                manifest_name = pattern if pattern in fileset else None

            if manifest_name is None:
                continue

            existing_priority = dir_priority.get(dirpath)
            if existing_priority is not None and existing_priority <= priority_idx:
                # A higher-priority (lower index) manifest already claimed this dir
                continue

            rel = dirpath.relative_to(working_dir)
            rel_str = str(rel).replace("\\", "/")
            if rel_str == ".":
                rel_str = "."

            # Remove any existing entry for this dir if we're replacing with higher priority
            if existing_priority is not None:
                projects[:] = [p for p in projects if p["path"] != rel_str]

            projects.append({"lang": lang, "path": rel_str, "manifest": manifest_name})
            dir_priority[dirpath] = priority_idx
            break  # only one entry per dir per pass; re-evaluate on next manifest match

    return projects


def _scan_extensions(working_dir: Path) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for dirpath, filenames in _walk(working_dir):
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext:
                counter[ext] += 1
    return dict(counter.most_common(15))


def scan_workspace(
    working_dir: Path,
    on_step: Callable[[str], None] | None = None,
) -> dict:
    """Scan the workspace and return a summary dict. Also writes .gekai/workspace.json."""
    if on_step:
        on_step("reading repo info")
    repo_name = _get_repo_name(working_dir)
    branch = get_git_branch(working_dir)
    if on_step:
        on_step("scanning manifests")
    projects = _scan_projects(working_dir)
    if on_step:
        on_step("counting file extensions")
    extensions = _scan_extensions(working_dir)
    if on_step:
        on_step("scanning AI instruction files")
    ai_instructions = _scan_ai_instructions(working_dir)

    if projects:
        seen: set[str] = set()
        primary_languages: list[str] = []
        for p in projects:
            if p["lang"] not in seen:
                seen.add(p["lang"])
                primary_languages.append(p["lang"])
    else:
        seen_langs: set[str] = set()
        primary_languages = []
        for ext in extensions:
            lang = _EXT_TO_LANG.get(ext)
            if lang and lang not in seen_langs:
                seen_langs.add(lang)
                primary_languages.append(lang)

    result: dict = {
        "repo_name": repo_name,
        "branch": branch,
        "projects": projects,
        "extensions": extensions,
        "primary_languages": primary_languages,
        "ai_instructions": ai_instructions,
    }

    try:
        gekai_dir = working_dir / ".gekai"
        gekai_dir.mkdir(exist_ok=True)
        (gekai_dir / "workspace.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    except OSError:
        pass

    return result


def get_git_branch(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            return branch if branch != "HEAD" else None
    except FileNotFoundError:
        pass
    return None
