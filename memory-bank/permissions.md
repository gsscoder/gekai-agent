# Permissions System
Session-scoped filesystem access control stored in `.gekai/settings.local.json`

## Settings File

Path: `{working_dir}/.gekai/settings.local.json`

Serialized schema (what `save_permissions` writes):

```json
{
  "permissions": {
    "workspace": { "read": "allow", "write": "deny", "exec": "deny" },
    "external": []
  }
}
```

`external` is initialized as `[]` on every save; external path permission handling is not yet implemented

## Dataclass

Defined in `agent/settings.py`:

```python
@dataclass
class Permissions:
    read: bool
    write: bool
    exec: bool = False
```

`exec` is serialized and deserialized via the `workspace.exec` field in the settings file

## Permission Gate (`agent/permissions.py`)

`PermissionCallback = Callable[[str, str], Awaitable[bool]]` — async callback: `(permission_name, tool_name) -> bool`. Used by the TUI to prompt the user at runtime

`PermissionGate` — dataclass with fields `permissions: Permissions` and `on_request: PermissionCallback | None`. Internal `_pending: set[str]` deduplicates concurrent requests for the same permission

`PermissionGate.check(tool) -> bool`:
- Returns `True` if `tool.required_permission == "none"` or the flag is already set on `permissions`
- If not granted and `on_request` is set, calls it; on grant, sets the flag on `permissions` and returns `True`
- Returns `False` if `on_request` is `None` or the callback returns `False`

Tools expose `required_permission: str` (`"none"`, `"read"`, or `"write"`). `ToolRegistry.run` calls `gate.check(tool)` before executing; on deny, returns a `ToolResultBlock` with `is_error=True` and message `"Permission denied: '{tool_name}' requires '{perm}' permission which was not granted"`

## Startup Dialog

Inline `_ask_choice` call in `GekaiApp` (not a separate screen). Shows three choices from `PERMISSION_CHOICES` defined in `agent/settings.py`. On cancel or unknown choice, defaults to `"deny"`

| Choice key | Label | Permissions result |
|---|---|---|
| `read_only` | Read Only — scan and read files, no modifications | `Permissions(read=True, write=False)` |
| `full` | Full Access — read, write, and delete files | `Permissions(read=True, write=True)` |
| `deny` | No Access — chat only, no file operations | `Permissions(read=False, write=False)` |

`resolve_permissions(choice: str) -> Permissions | None` maps a choice key to a `Permissions` instance. Returns `None` for unknown keys

`exec` is never set at startup — always defaults to `False`

Session continues after "No Access"; chat still works, file tools are blocked

## Load / Save

`load_permissions(working_dir: Path) -> Permissions | None` — returns `None` if the settings file does not exist (triggers startup dialog). Reads `permissions.workspace.read`, `.write`, `.exec`; `"allow"` → `True`, anything else → `False`

`save_permissions(working_dir: Path, permissions: Permissions) -> None` — creates `.gekai/` if absent; reads existing file to preserve other keys; writes nested `permissions.workspace` block with `read`/`write`/`exec`; initializes `permissions.external` to `[]` if not present

## exec Permission (Forward-Looking)

`exec` on `Permissions` is serialized to/from `permissions.workspace.exec` in the settings file (defaults to `"deny"`). Intended lazy-grant behavior (not yet implemented):

- When a tool attempts a shell command, the TUI asks inline via `_ask_choice`
- If granted, the settings file is updated to persist the grant
- No shell execution tool exists yet; this path is not reachable in the current codebase

## External Path Permissions

`ExternalGrantCallback = Callable[[Path, str], Awaitable[bool]]` (`agent/tools/files.py`); threaded as `external_grant_cb` through `make_tools` → `FileToolContext`

`FileToolContext.external: list[Path]` is populated at construction via `load_external(working_dir)` (`agent/settings.py`) — previously-granted roots skip re-prompting

On any file operation whose resolved target falls outside `working_dir`:
- `candidate_root` = first non-workspace ancestor directory
- in-flight dedup: same `candidate_root + mode` key asked at most once concurrently
- `external_grant_cb is None` → deny; callback returns `False` → deny; either way returns `"error: access to external path denied: {target}"`
- on grant: `ctx.external.append(candidate_root)` + `save_external(ctx.working_dir, candidate_root)` — persisted to `permissions.external` in `.gekai/settings.local.json` as resolved absolute path strings

`save_external` merge rule (`agent/settings.py`): if `root.is_relative_to(e)` for any existing entry `e` → no-op (already covered); entries narrower than `root` are removed and replaced by `root` (widest-wins merge)

TUI: `GekaiApp._external_grant_callback(root, mode)` — prompts via `ChoiceBar` "Grant full access to external path '{root}' (outside the workspace)?" (y/n); pauses spinner clock; returns `False` immediately if worker already cancelled

`external` is initialized as `[]` in the settings file when first created by `save_permissions`
