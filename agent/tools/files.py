from __future__ import annotations

import re
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from agent.llm import tool
from agent.settings import load_allow_hidden, save_allow_hidden
from agent.workspace import scanner
from agent.workspace.ignore import IgnoreRules, load as _load_ignore_rules
from agent.workspace.symbols import _EXT_TO_LANG, _LANG_TO_MODULE, _LANG_QUERIES

HiddenGrantCallback = Callable[[str, str], Awaitable[bool]]

_MAX_RESULTS = 200
_MAX_GREP_FILES = 5000


@dataclass(slots=True)
class FileToolContext:
    """The workspace-root/hidden-grant plumbing every file-tool helper needs —
    repeats verbatim across `_authorize` and every helper that calls it, and
    again from each helper's matching tool closure in `make_file_tools`;
    carried as one unit instead of 4 loose parameters."""
    working_dir: Path
    allow_hidden: set[str] | None = None
    grant_cb: HiddenGrantCallback | None = None
    pending: set[str] | None = None


def _resolve_in_ws(path: str, working_dir: Path) -> Path | None:
    target = (working_dir / path).resolve()
    base = working_dir.resolve()
    if not target.is_relative_to(base):
        return None
    rel = str(target.relative_to(base)).replace("\\", "/")
    rules = _load_ignore_rules(base)
    if rules.is_forbidden(rel) or rules.is_forbidden(rel + "/"):
        return None
    return target


async def _authorize(
    path: str,
    ctx: FileToolContext,
    *,
    mode: str,
) -> Path | str:
    """Resolve `path` and gate hidden-but-not-forbidden access behind a grant.

    Returns the resolved Path on success, or an "error: ..." string on
    failure. Forbidden (.aiignore) paths fail via _resolve_in_ws before any
    grant logic runs - the red zone is never prompted.

    `ctx.pending` tracks hidden paths with an in-flight grant request. If a
    concurrent same-batch call targets the same path while a grant is
    already pending, it is denied immediately rather than double-prompting
    (mirrors PermissionGate._pending).
    """
    target = _resolve_in_ws(path, ctx.working_dir)
    if target is None:
        return "error: path outside working directory"

    rel = str(target.relative_to(ctx.working_dir.resolve())).replace("\\", "/")
    rules = _load_ignore_rules(ctx.working_dir)
    if rules.is_hidden(rel) or rules.is_hidden(rel + "/"):
        granted = ctx.allow_hidden if ctx.allow_hidden is not None else set()
        if rel not in granted:
            in_flight = ctx.pending if ctx.pending is not None else set()
            if rel in in_flight:
                return f"error: access to hidden path denied: {rel}"
            if ctx.grant_cb is None:
                return f"error: access to hidden path denied: {rel}"
            in_flight.add(rel)
            try:
                if not await ctx.grant_cb(rel, mode):
                    return f"error: access to hidden path denied: {rel}"
            finally:
                in_flight.discard(rel)
            granted.add(rel)
            save_allow_hidden(ctx.working_dir, rel)

    return target


async def _authorize_file(
    path: str,
    ctx: FileToolContext,
    *,
    mode: str,
) -> Path | str:
    """Like `_authorize`, but also rejects paths that resolve to a directory."""
    result = await _authorize(path, ctx, mode=mode)
    if isinstance(result, str):
        return result
    if result.is_dir():
        return f"error: {path!r} is a directory — specify a file path"
    return result


def _walk_files(base: Path, root: Path) -> list[Path]:
    """Files under `base`, skipping ignored dirs (.git, .venv, .gekai, etc.).

    Reuses scanner._walk so grep never reads VCS internals, virtualenvs, or
    build output — walking those reads thousands of files and can stall a
    single grep call for minutes. Bounded by _MAX_GREP_FILES.
    """
    files: list[Path] = []
    for dirpath, _, filenames in scanner._walk(base, root=root):
        for fname in filenames:
            files.append(dirpath / fname)
            if len(files) >= _MAX_GREP_FILES:
                return files
    return files


async def _read_file(
    path: str,
    *,
    ctx: FileToolContext,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    result = await _authorize_file(path, ctx, mode="read")
    if isinstance(result, str):
        return result
    target = result
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
    rules = _load_ignore_rules(working_dir)
    matches: list[str] = []
    for p in working_dir.glob(pattern):
        if not (p.is_file() or p.is_dir()):
            continue
        rel = str(p.relative_to(working_dir)).replace("\\", "/")
        if p.is_dir():
            rel += "/"
        if rules.is_hidden(rel):
            continue
        if rel.endswith("/"):
            matches.append(rel)
        else:
            matches.append(f"{rel} ({p.stat().st_size} bytes)")
    matches = sorted(matches)[:_MAX_RESULTS]
    if matches:
        return "\n".join(matches)
    hint = _no_match_hint(pattern, working_dir=working_dir, rules=rules)
    return hint if hint else "(no matches)"


def _no_match_hint(pattern: str, *, working_dir: Path, rules: IgnoreRules) -> str:
    """Find the deepest existing ancestor of `pattern` and list its children as a hint.

    ponytail: single-ancestor hint, not a fuzzy/recursive search — if this
    ceiling is ever hit, upgrade to a bounded basename walk instead of full
    recursion.
    """
    parts = Path(pattern).parts
    ancestor = working_dir
    ancestor_rel = ""
    for part in parts:
        if any(ch in part for ch in "*?["):
            break
        candidate = ancestor / part
        if not candidate.is_dir():
            break
        ancestor = candidate
        ancestor_rel = str(ancestor.relative_to(working_dir)).replace("\\", "/")

    if ancestor_rel and rules.is_hidden(ancestor_rel + "/"):
        return ""

    try:
        children = sorted(ancestor.iterdir())
    except OSError:
        return ""

    names: list[str] = []
    for child in children:
        rel = str(child.relative_to(working_dir)).replace("\\", "/")
        name = child.name + "/" if child.is_dir() else child.name
        if rules.is_hidden(rel + "/" if child.is_dir() else rel):
            continue
        names.append(name)
        if len(names) >= _MAX_RESULTS:
            break

    if not names:
        return ""
    where = ancestor_rel if ancestor_rel else "."
    return f"no matches for {pattern!r}; did you mean one of these in '{where}': {', '.join(names)}?"


async def _file_info(
    path: str,
    *,
    ctx: FileToolContext,
) -> str:
    result = await _authorize_file(path, ctx, mode="read")
    if isinstance(result, str):
        return result
    target = result
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
        line_count = len(text.splitlines())
        byte_size = target.stat().st_size
        return f"lines: {line_count}, size: {byte_size} bytes"
    except FileNotFoundError:
        return f"error: file not found: {path}"
    except Exception as exc:
        return f"error: {exc or type(exc).__name__}"


async def _grep(
    pattern: str,
    path: str | None = None,
    *,
    ctx: FileToolContext,
) -> str:
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"error: invalid pattern: {exc}"

    root = ctx.working_dir.resolve()
    if path:
        result = await _authorize(path, ctx, mode="read")
        if isinstance(result, str):
            return result
        target = result
        candidates = [target] if target.is_file() else _walk_files(target, root)
    else:
        candidates = _walk_files(root, root)

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
        except (OSError, UnicodeDecodeError):
            continue

    return "\n".join(results) if results else "(no matches)"


async def _edit_file(
    path: str,
    old_str: str | None = None,
    new_str: str | None = None,
    edits: list[dict[str, str]] | None = None,
    *,
    ctx: FileToolContext,
) -> str:
    if edits is not None:
        if old_str is not None or new_str is not None:
            return "error: pass either old_str/new_str or edits, not both"
    elif old_str is None or new_str is None:
        return "error: old_str/new_str required when edits is not given"

    result = await _authorize_file(path, ctx, mode="write")
    if isinstance(result, str):
        return result
    target = result
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"error: file not found: {path}"
    except Exception as exc:
        return f"error: {exc or type(exc).__name__}"

    if edits is None:
        assert old_str is not None and new_str is not None  # guaranteed by the guard above
        if old_str not in text:
            return f"error: old_str not found in {path}"
        target.write_text(text.replace(old_str, new_str, 1), encoding="utf-8")
        return "ok"

    working_text = text
    for i, hunk in enumerate(edits):
        hunk_old = hunk.get("old_str")
        hunk_new = hunk.get("new_str")
        if hunk_old is None or hunk_new is None:
            return f"error: edits[{i}] missing old_str/new_str"
        if hunk_old not in working_text:
            return f"error: edits[{i}].old_str not found in {path}"
        working_text = working_text.replace(hunk_old, hunk_new, 1)

    target.write_text(working_text, encoding="utf-8")
    return "ok"


async def _write_file(
    path: str,
    content: str,
    *,
    ctx: FileToolContext,
) -> str:
    result = await _authorize_file(path, ctx, mode="write")
    if isinstance(result, str):
        return result
    target = result
    try:
        # ponytail: exists()-then-write is a TOCTOU gap if two write_file
        # calls race on the same new path in one turn (edit_file/write_file
        # are concurrency-safe by default) — both would see existed=False and
        # tag "ok" instead of the second being "ok: overwritten". Narrow:
        # misclassifies the verifier gate's via, doesn't corrupt file
        # content. Upgrade path if it matters: is_concurrency_safe=False.
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return "ok: overwritten" if existed else "ok"
    except Exception as exc:
        return f"error: {exc or type(exc).__name__}"


async def _symbols(
    path: str,
    *,
    ctx: FileToolContext,
    kind: str | None = None,
) -> str:
    try:
        import importlib
        from tree_sitter import Language, Parser, Query, QueryCursor, QueryError
    except ImportError:
        return "error: tree-sitter not installed (pip install tree-sitter tree-sitter-python tree-sitter-typescript tree-sitter-javascript tree-sitter-go)"

    result = await _authorize(path, ctx, mode="read")
    if isinstance(result, str):
        return result
    target = result

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
        except QueryError:
            # QueryError (tree_sitter): a hand-authored query in _LANG_QUERIES is
            # malformed for this grammar version - skip that symbol kind, keep the rest.
            continue

    if not results:
        return "(no symbols found)"
    results.sort(key=lambda r: r[0])
    return "\n".join(f"{name}:{line}:{kind_}" for line, name, kind_ in results)


async def _move_file(
    src: str,
    dst: str,
    *,
    ctx: FileToolContext,
) -> str:
    src_result = await _authorize_file(src, ctx, mode="write")
    if isinstance(src_result, str):
        return src_result
    src_path = src_result
    dst_result = await _authorize(dst, ctx, mode="write")
    if isinstance(dst_result, str):
        return dst_result
    dst_path = dst_result
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


async def _copy_file(
    src: str,
    dst: str,
    *,
    ctx: FileToolContext,
) -> str:
    src_result = await _authorize(src, ctx, mode="write")
    if isinstance(src_result, str):
        return src_result
    src_path = src_result
    dst_result = await _authorize(dst, ctx, mode="write")
    if isinstance(dst_result, str):
        return dst_result
    dst_path = dst_result
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


async def _delete_file(
    path: str,
    *,
    ctx: FileToolContext,
) -> str:
    result = await _authorize(path, ctx, mode="write")
    if isinstance(result, str):
        return result
    target = result
    if target.is_dir():
        return f"error: target is a directory, not a file: {path}"
    if not target.exists():
        return f"error: file not found: {path}"
    try:
        target.unlink()
        return "ok"
    except Exception as exc:
        return f"error: {exc or type(exc).__name__}"


async def _make_dir(
    path: str,
    *,
    ctx: FileToolContext,
) -> str:
    result = await _authorize(path, ctx, mode="write")
    if isinstance(result, str):
        return result
    target = result
    try:
        target.mkdir(parents=True, exist_ok=True)
        return "ok"
    except Exception as exc:
        return f"error: {exc or type(exc).__name__}"


def make_file_tools(working_dir: Path, grant_cb: HiddenGrantCallback | None = None) -> list:
    ctx = FileToolContext(
        working_dir=working_dir,
        allow_hidden=load_allow_hidden(working_dir),
        grant_cb=grant_cb,
        pending=set(),
    )

    @tool(is_read_only=True, required_permission="read")
    async def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        """Read a file in the workspace.

        For files over ~300 lines or ~15KB, read a targeted range rather than
        the full file. Check the size first — list_files shows byte size per
        entry, or use file_info for exact line count. Use symbols/grep to
        locate relevant lines. Then pass start_line/end_line (1-based,
        inclusive) to read only what is needed. Omit both to read the full file.
        When showing multiple symbols from the same file, make ONE call spanning
        from the lowest to the highest line (add ~5 line buffer) — never one
        call per symbol.
        """
        return await _read_file(path, ctx=ctx, start_line=start_line, end_line=end_line)

    @tool(is_read_only=True, required_permission="read")
    async def list_files(pattern: str) -> str:
        """List files and directories matching a glob pattern (e.g. '**/*.py', '*').
        Directories appear with a trailing '/'; files show their byte size,
        e.g. 'agent/foo.py (1234 bytes)' — use this to decide whether
        read_file needs a targeted range (see read_file for the size threshold)
        before reading. On no match, returns a hint listing the nearest
        existing ancestor directory's contents instead of a dead end."""
        return await _list_files(pattern, working_dir=working_dir)

    @tool(is_read_only=True, required_permission="read")
    async def grep(pattern: str, path: str | None = None) -> str:
        """Search file contents for a regex pattern. Returns matching lines as file:line: content."""
        return await _grep(pattern, path=path, ctx=ctx)

    @tool(is_read_only=True, required_permission="read")
    async def file_info(path: str) -> str:
        """Return line count and byte size for a file. Use before read_file to decide
        whether to read the full file or a targeted range — files over ~300 lines
        or ~15KB are candidates for a targeted range instead of a full read."""
        return await _file_info(path, ctx=ctx)

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
        return await _symbols(path, ctx=ctx, kind=kind)

    @tool(is_read_only=False, required_permission="write")
    async def edit_file(
        path: str,
        old_str: str | None = None,
        new_str: str | None = None,
        edits: list[dict[str, str]] | None = None,
    ) -> str:
        """Edit a file by replacing the first occurrence of old_str with new_str.

        old_str must match the file content exactly (including whitespace and indentation).
        Returns 'ok' on success or an error string on failure.
        To replace a larger block, include enough surrounding context to make old_str unique.

        For several edits to the same file in one call, pass `edits` instead — a list of
        {"old_str": ..., "new_str": ...} hunks applied in order, each checked against the
        file text as already modified by the prior hunks in the same call. The write is
        atomic: either every hunk applies and the file is written once, or none of them
        are written. Use this to cut round trips when a file needs several edits at once;
        do not pass old_str/new_str together with edits.
        """
        return await _edit_file(path, old_str=old_str, new_str=new_str, edits=edits, ctx=ctx)

    @tool(is_read_only=False, required_permission="write")
    async def write_file(path: str, content: str) -> str:
        """Write content to a file, creating it if absent or overwriting if present.

        Use for new files or complete rewrites. Prefer edit_file for targeted changes.
        Returns 'ok' on success or an error string on failure.
        """
        return await _write_file(path, content=content, ctx=ctx)

    @tool(is_read_only=False, required_permission="write", is_concurrency_safe=False)
    async def move_file(src: str, dst: str) -> str:
        """Move or rename a file within the workspace.

        Both src and dst must be paths relative to the workspace root.
        Refuses if dst already exists — no silent overwrite.
        Returns 'ok' on success or an error string on failure.
        """
        return await _move_file(src, dst, ctx=ctx)

    @tool(is_read_only=False, required_permission="write", is_concurrency_safe=False)
    async def copy_file(src: str, dst: str) -> str:
        """Copy a file within the workspace.

        src must be a file (not a directory). Both paths must be within the workspace.
        Refuses if dst already exists — no silent overwrite.
        Returns 'ok' on success or an error string on failure.
        """
        return await _copy_file(src, dst, ctx=ctx)

    @tool(is_read_only=False, required_permission="write", is_concurrency_safe=False)
    async def delete_file(path: str) -> str:
        """Delete a single file from the workspace.

        Refuses directories — use this only for files.
        Returns 'ok' on success or an error string on failure.
        """
        return await _delete_file(path, ctx=ctx)

    @tool(is_read_only=False, required_permission="write", is_concurrency_safe=False)
    async def make_dir(path: str) -> str:
        """Create a directory (and any missing parents) in the workspace.

        Safe to call when the directory already exists.
        Returns 'ok' on success or an error string on failure.
        """
        return await _make_dir(path, ctx=ctx)

    return [read_file, list_files, grep, file_info, symbols, edit_file, write_file,
            move_file, copy_file, delete_file, make_dir]
