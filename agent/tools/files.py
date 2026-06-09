from __future__ import annotations

import re
import shutil
from pathlib import Path

from agent.llm import tool

_MAX_RESULTS = 200


def _resolve_in_ws(path: str, working_dir: Path) -> Path | None:
    target = (working_dir / path).resolve()
    return target if target.is_relative_to(working_dir.resolve()) else None


_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
}

# (module_name, function_name) to obtain the language capsule
_LANG_TO_MODULE: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
}

_LANG_QUERIES: dict[str, dict[str, str]] = {
    "python": {
        "function": "(function_definition name: (identifier) @name)",
        "class": "(class_definition name: (identifier) @name)",
    },
    "javascript": {
        "function": "(function_declaration name: (identifier) @name)",
        "class": "(class_declaration name: (identifier) @name)",
        "method": "(method_definition name: (property_identifier) @name)",
    },
    "typescript": {
        "function": "(function_declaration name: (identifier) @name)",
        "class": "(class_declaration name: (type_identifier) @name)",
        "method": "(method_definition name: (property_identifier) @name)",
        "interface": "(interface_declaration name: (type_identifier) @name)",
    },
    "go": {
        "function": "(function_declaration name: (identifier) @name)",
        "method": "(method_declaration name: (field_identifier) @name)",
        "type": "(type_spec name: (type_identifier) @name)",
    },
}
_LANG_QUERIES["tsx"] = _LANG_QUERIES["typescript"]


async def _read_file(
    path: str,
    *,
    working_dir: Path,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    target = _resolve_in_ws(path, working_dir)
    if target is None:
        return "error: path outside working directory"
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
        if start_line is None and end_line is None:
            return text
        lines = text.splitlines(keepends=True)
        total = len(lines)
        s = max(0, (start_line or 1) - 1)
        e = min(total, end_line) if end_line is not None else total
        return "".join(lines[s:e])
    except FileNotFoundError:
        return f"error: file not found: {path}"
    except Exception as exc:
        return f"error: {exc or type(exc).__name__}"


async def _list_files(pattern: str, *, working_dir: Path) -> str:
    def _entry(p: Path) -> str:
        rel = str(p.relative_to(working_dir))
        return rel + "/" if p.is_dir() else rel

    matches = sorted(
        _entry(p)
        for p in working_dir.glob(pattern)
        if p.is_file() or p.is_dir()
    )[:_MAX_RESULTS]
    return "\n".join(matches) if matches else "(no matches)"


async def _file_info(path: str, *, working_dir: Path) -> str:
    target = _resolve_in_ws(path, working_dir)
    if target is None:
        return "error: path outside working directory"
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
        line_count = len(text.splitlines())
        byte_size = target.stat().st_size
        return f"lines: {line_count}, size: {byte_size} bytes"
    except FileNotFoundError:
        return f"error: file not found: {path}"
    except Exception as exc:
        return f"error: {exc or type(exc).__name__}"


async def _grep(pattern: str, path: str | None = None, *, working_dir: Path) -> str:
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"error: invalid pattern: {exc}"

    root = working_dir.resolve()
    if path:
        target = _resolve_in_ws(path, working_dir)
        if target is None:
            return "error: path outside working directory"
        candidates = [target] if target.is_file() else [p for p in target.rglob("*") if p.is_file()]
    else:
        candidates = [p for p in root.rglob("*") if p.is_file()]

    results: list[str] = []
    for file in candidates:
        if len(results) >= _MAX_RESULTS:
            break
        try:
            for i, line in enumerate(
                file.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if regex.search(line):
                    results.append(f"{file.relative_to(root)}:{i}: {line}")
                    if len(results) >= _MAX_RESULTS:
                        break
        except Exception:
            continue

    return "\n".join(results) if results else "(no matches)"


async def _edit_file(path: str, old_str: str, new_str: str, *, working_dir: Path) -> str:
    target = _resolve_in_ws(path, working_dir)
    if target is None:
        return "error: path outside working directory"
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"error: file not found: {path}"
    except Exception as exc:
        return f"error: {exc or type(exc).__name__}"
    if old_str not in text:
        return f"error: old_str not found in {path}"
    target.write_text(text.replace(old_str, new_str, 1), encoding="utf-8")
    return "ok"


async def _write_file(path: str, content: str, *, working_dir: Path) -> str:
    target = _resolve_in_ws(path, working_dir)
    if target is None:
        return "error: path outside working directory"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return "ok"
    except Exception as exc:
        return f"error: {exc or type(exc).__name__}"


async def _symbols(
    path: str,
    *,
    working_dir: Path,
    kind: str | None = None,
) -> str:
    try:
        import importlib
        from tree_sitter import Language, Parser, Query, QueryCursor
    except ImportError:
        return "error: tree-sitter not installed (pip install tree-sitter tree-sitter-python tree-sitter-typescript tree-sitter-javascript tree-sitter-go)"

    target = _resolve_in_ws(path, working_dir)
    if target is None:
        return "error: path outside working directory"

    ext = target.suffix.lower()
    lang_name = _EXT_TO_LANG.get(ext)
    if lang_name is None:
        return f"error: unsupported file type '{ext}'; supported: {', '.join(_EXT_TO_LANG)}"

    try:
        source = target.read_bytes()
    except FileNotFoundError:
        return f"error: file not found: {path}"
    except Exception as exc:
        return f"error: {exc or type(exc).__name__}"

    mod_name, fn_name = _LANG_TO_MODULE[lang_name]
    try:
        mod = importlib.import_module(mod_name)
        language = Language(getattr(mod, fn_name)())
    except Exception as exc:
        return f"error: could not load language '{lang_name}': {exc}"

    requested: set[str] | None = {k.strip() for k in kind.split(",")} if kind else None
    queries = _LANG_QUERIES[lang_name]
    parser = Parser(language)
    tree = parser.parse(source)

    results: list[tuple[int, str, str]] = []
    for symbol_kind, query_str in queries.items():
        if requested and symbol_kind not in requested:
            continue
        try:
            cursor = QueryCursor(Query(language, query_str))
            caps: dict[str, list] = cursor.captures(tree.root_node)
            for node in caps.get("name", []):
                name = node.text.decode("utf-8", errors="replace")
                line = node.start_point[0] + 1
                results.append((line, name, symbol_kind))
        except Exception:
            continue

    if not results:
        return "(no symbols found)"
    results.sort(key=lambda r: r[0])
    return "\n".join(f"{name}:{line}:{kind_}" for line, name, kind_ in results)


async def _move_file(src: str, dst: str, *, working_dir: Path) -> str:
    src_path = _resolve_in_ws(src, working_dir)
    if src_path is None:
        return "error: path outside working directory"
    dst_path = _resolve_in_ws(dst, working_dir)
    if dst_path is None:
        return "error: path outside working directory"
    if not src_path.exists():
        return f"error: source not found: {src}"
    if dst_path.exists():
        return f"error: destination already exists: {dst}"
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        return "ok"
    except Exception as exc:
        return f"error: {exc or type(exc).__name__}"


async def _copy_file(src: str, dst: str, *, working_dir: Path) -> str:
    src_path = _resolve_in_ws(src, working_dir)
    if src_path is None:
        return "error: path outside working directory"
    dst_path = _resolve_in_ws(dst, working_dir)
    if dst_path is None:
        return "error: path outside working directory"
    if not src_path.is_file():
        return f"error: source not a file: {src}"
    if dst_path.exists():
        return f"error: destination already exists: {dst}"
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_path), str(dst_path))
        return "ok"
    except Exception as exc:
        return f"error: {exc or type(exc).__name__}"


async def _delete_file(path: str, *, working_dir: Path) -> str:
    target = _resolve_in_ws(path, working_dir)
    if target is None:
        return "error: path outside working directory"
    if target.is_dir():
        return f"error: target is a directory, not a file: {path}"
    if not target.exists():
        return f"error: file not found: {path}"
    try:
        target.unlink()
        return "ok"
    except Exception as exc:
        return f"error: {exc or type(exc).__name__}"


async def _make_dir(path: str, *, working_dir: Path) -> str:
    target = _resolve_in_ws(path, working_dir)
    if target is None:
        return "error: path outside working directory"
    try:
        target.mkdir(parents=True, exist_ok=True)
        return "ok"
    except Exception as exc:
        return f"error: {exc or type(exc).__name__}"


def make_file_tools(working_dir: Path) -> list:
    @tool(is_read_only=True, required_permission="read")
    async def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        """Read a file in the workspace.

        For large files, read a targeted range rather than the full file.
        Use file_info to check line count first, or symbols/grep to locate
        relevant lines. Then pass start_line/end_line (1-based, inclusive) to
        read only what is needed. Omit both to read the full file.
        When showing multiple symbols from the same file, make ONE call spanning
        from the lowest to the highest line (add ~5 line buffer) — never one
        call per symbol.
        """
        return await _read_file(path, working_dir=working_dir, start_line=start_line, end_line=end_line)

    @tool(is_read_only=True, required_permission="read")
    async def list_files(pattern: str) -> str:
        """List files and directories matching a glob pattern (e.g. '**/*.py', '*').
        Directories appear with a trailing '/'."""
        return await _list_files(pattern, working_dir=working_dir)

    @tool(is_read_only=True, required_permission="read")
    async def grep(pattern: str, path: str | None = None) -> str:
        """Search file contents for a regex pattern. Returns matching lines as file:line: content."""
        return await _grep(pattern, path=path, working_dir=working_dir)

    @tool(is_read_only=True, required_permission="read")
    async def file_info(path: str) -> str:
        """Return line count and byte size for a file. Use before read_file to decide
        whether to read the full file or a targeted range."""
        return await _file_info(path, working_dir=working_dir)

    @tool(is_read_only=True, required_permission="read")
    async def symbols(path: str, kind: str | None = None) -> str:
        """Find symbol declarations in a source file using AST parsing.

        Returns one match per line as name:line_number:kind.
        Supported languages: Python (.py), TypeScript (.ts/.tsx), JavaScript (.js), Go (.go).
        kind: comma-separated filter — 'function', 'class', 'method', 'interface', 'type'.
        Omit kind to return all declaration types.
        For simple functions the name:line output is often sufficient to answer
        signature questions — only read_file if the full signature is required.
        """
        return await _symbols(path, working_dir=working_dir, kind=kind)

    @tool(is_read_only=False, required_permission="write")
    async def edit_file(path: str, old_str: str, new_str: str) -> str:
        """Edit a file by replacing the first occurrence of old_str with new_str.

        old_str must match the file content exactly (including whitespace and indentation).
        Returns 'ok' on success or an error string on failure.
        To replace a larger block, include enough surrounding context to make old_str unique.
        """
        return await _edit_file(path, old_str=old_str, new_str=new_str, working_dir=working_dir)

    @tool(is_read_only=False, required_permission="write")
    async def write_file(path: str, content: str) -> str:
        """Write content to a file, creating it if absent or overwriting if present.

        Use for new files or complete rewrites. Prefer edit_file for targeted changes.
        Returns 'ok' on success or an error string on failure.
        """
        return await _write_file(path, content=content, working_dir=working_dir)

    @tool(is_read_only=False, required_permission="write", is_concurrency_safe=False)
    async def move_file(src: str, dst: str) -> str:
        """Move or rename a file within the workspace.

        Both src and dst must be paths relative to the workspace root.
        Refuses if dst already exists — no silent overwrite.
        Returns 'ok' on success or an error string on failure.
        """
        return await _move_file(src, dst, working_dir=working_dir)

    @tool(is_read_only=False, required_permission="write", is_concurrency_safe=False)
    async def copy_file(src: str, dst: str) -> str:
        """Copy a file within the workspace.

        src must be a file (not a directory). Both paths must be within the workspace.
        Refuses if dst already exists — no silent overwrite.
        Returns 'ok' on success or an error string on failure.
        """
        return await _copy_file(src, dst, working_dir=working_dir)

    @tool(is_read_only=False, required_permission="write", is_concurrency_safe=False)
    async def delete_file(path: str) -> str:
        """Delete a single file from the workspace.

        Refuses directories — use this only for files.
        Returns 'ok' on success or an error string on failure.
        """
        return await _delete_file(path, working_dir=working_dir)

    @tool(is_read_only=False, required_permission="write", is_concurrency_safe=False)
    async def make_dir(path: str) -> str:
        """Create a directory (and any missing parents) in the workspace.

        Safe to call when the directory already exists.
        Returns 'ok' on success or an error string on failure.
        """
        return await _make_dir(path, working_dir=working_dir)

    return [read_file, list_files, grep, file_info, symbols, edit_file, write_file,
            move_file, copy_file, delete_file, make_dir]
