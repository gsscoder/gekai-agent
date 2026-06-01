# Permissions System
Session-scoped filesystem access control stored in `.gekai/settings.local.json`

## Settings File

Path: `{working_dir}/.gekai/settings.local.json`

Current serialized schema (what `save_permissions` actually writes):

```json
{
  "permissions": {
    "read": "allow",
    "write": "deny"
  }
}
```

Planned schema (not yet implemented — forward-looking):

```json
{
  "permissions": {
    "workspace": { "read": "allow", "write": "deny", "exec": "deny" },
    "external": [
      {
        "path": "C:\\MyRepos\\Other Project",
        "read": "allow",
        "write": "deny",
        "exec": "deny"
      }
    ]
  }
}
```

The `workspace` nesting and `external` array are not written or read by the current code. `load_permissions` reads `data["permissions"]["read"]` and `data["permissions"]["write"]` directly from the flat block.

## Dataclass

Defined in `agent/settings.py`:

```python
@dataclass
class Permissions:
    read: bool
    write: bool
    exec: bool = False
```

`exec` is present on the dataclass but never serialized to or deserialized from the settings file. `load_permissions` does not read an `exec` field; it is always `False` on loaded instances (Python default).

## Startup Dialog

`PermissionScreen` in `agent/tui/permissions.py` — a `ModalScreen[str | None]` that shows three choices from `PERMISSION_CHOICES` in `agent/settings.py`. Navigation: up/down arrows, Enter to confirm, Escape to cancel (returns `None`; app exits on cancel).

| Choice key | Label | Permissions result |
|---|---|---|
| `read_only` | Read Only — scan and read files, no modifications | `Permissions(read=True, write=False)` |
| `full` | Full Access — read, write, and delete files | `Permissions(read=True, write=True)` |
| `deny` | No Access — chat only, no file operations | `Permissions(read=False, write=False)` |

`resolve_permissions(choice: str) -> Permissions | None` maps a choice key to a `Permissions` instance. Returns `None` for unknown keys.

`exec` is never set at startup — always defaults to `False`.

Session continues after "No Access"; chat still works, file tools are blocked.

## Load / Save

`load_permissions(working_dir: Path) -> Permissions | None` — returns `None` if the settings file does not exist (triggers startup dialog). Reads `permissions.read` and `permissions.write` string values; `"allow"` → `True`, anything else → `False`.

`save_permissions(working_dir: Path, permissions: Permissions) -> None` — creates `.gekai/` if absent, writes the flat `read`/`write` block. Does not write `exec`, `workspace` nesting, or `external`.

## exec Permission (Forward-Looking)

`exec` on the `Permissions` dataclass is reserved for lazy grant of shell execution. Intended behavior (not yet implemented):

- Both workspace and external `exec` start as `"deny"`.
- When a tool attempts a shell command, the TUI asks inline via `_ask_choice`.
- If granted, the settings file is updated to persist the grant.
- No shell execution tool exists yet; this path is not reachable in the current codebase.

## External Path Permissions (Forward-Looking)

Not yet implemented. Intended behavior:

- Any filesystem access outside the workspace checks the `external` list in the settings file.
- If no entry matches the target path, the TUI asks inline; on grant, a new entry is appended.
- Lookup uses longest-prefix match — most-specific entry wins.
- Path overlap / merge rule: granting access at a path that is a parent of (or equal to) an existing entry replaces that entry with the wider parent path.
- `external` is initialized as an empty array `[]` when the settings file is first created.
