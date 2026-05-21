from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich.console import Console


@dataclass
class Permissions:
    read: bool
    write: bool


_CHOICES = [
    ("read_only", "Read Only — scan and read files, no modifications"),
    ("full", "Full Access — read, write, and delete files"),
    ("deny", "Deny — exit to terminal"),
]


async def _run_selector() -> str | None:
    selected = [0]

    bindings = KeyBindings()

    @bindings.add("up")
    def _up(event):
        selected[0] = (selected[0] - 1) % len(_CHOICES)

    @bindings.add("down")
    def _down(event):
        selected[0] = (selected[0] + 1) % len(_CHOICES)

    @bindings.add("enter")
    def _enter(event):
        event.app.exit(result=_CHOICES[selected[0]][0])

    @bindings.add("c-c")
    @bindings.add("c-d")
    def _cancel(event):
        event.app.exit(result=None)

    def _get_text():
        lines = []
        for i, (_, label) in enumerate(_CHOICES):
            if i == selected[0]:
                lines.append(("class:selected", f"❯ {label}\n"))
            else:
                lines.append(("class:unselected", f"  {label}\n"))
        return lines

    style = Style.from_dict({
        "selected": "ansicyan bold",
        "unselected": "ansibrightblack",
    })

    layout = Layout(Window(content=FormattedTextControl(_get_text), height=len(_CHOICES)))
    app: Application[str | None] = Application(layout=layout, key_bindings=bindings, style=style)
    return await app.run_async()


def _settings_path(working_dir: Path) -> Path:
    return working_dir / ".gekai" / "settings.local.json"


def load_permissions(working_dir: Path) -> Permissions | None:
    path = _settings_path(working_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    perms = data.get("permissions", {})
    return Permissions(
        read=perms.get("read") == "allow",
        write=perms.get("write") == "allow",
    )


def save_permissions(working_dir: Path, permissions: Permissions) -> None:
    path = _settings_path(working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "permissions": {
            "read": "allow" if permissions.read else "deny",
            "write": "allow" if permissions.write else "deny",
        }
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


async def prompt_permissions(working_dir: Path) -> Permissions | None:
    console = Console()
    console.print("[bold]Gekai needs access to this workspace[/bold]")
    console.print()

    result = await _run_selector()

    if result is None or result == "deny":
        return None

    if result == "read_only":
        perms = Permissions(read=True, write=False)
    else:
        perms = Permissions(read=True, write=True)

    save_permissions(working_dir, perms)
    return perms
