from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class ShellSpec:
    kind: str           # "powershell" | "bash" | "sh"
    path: str

    def build_args(self, cmd: str) -> list[str]:
        if self.kind == "powershell":
            return ["-NoProfile", "-NonInteractive", "-Command", cmd]
        else:
            return ["-lc", cmd]


@lru_cache(maxsize=1)
def resolve_shell() -> ShellSpec:
    """Resolve the native shell once at startup. Cached — call freely."""
    if sys.platform == "win32":
        pwsh = shutil.which("pwsh")
        if pwsh:
            return ShellSpec(kind="powershell", path=pwsh)
        ps = shutil.which("powershell")
        if ps:
            return ShellSpec(kind="powershell", path=ps)
        raise ValueError(
            "No PowerShell found on Windows. Install pwsh or ensure powershell.exe is in PATH."
        )
    else:
        import os
        shell = os.environ.get("SHELL") or shutil.which("bash") or shutil.which("sh")
        if not shell:
            raise ValueError("No shell found. Ensure bash or sh is in PATH.")
        kind = "bash" if "bash" in shell else "sh"
        return ShellSpec(kind=kind, path=shell)
