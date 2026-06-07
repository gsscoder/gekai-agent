from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .persona import SYSTEM_PROMPT
from .settings import Permissions


@dataclass
class Session:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict] = field(
        default_factory=lambda: [{"role": "system", "content": SYSTEM_PROMPT}]
    )
    working_dir: Path = field(default_factory=Path.cwd)
    permissions: Permissions = field(default_factory=lambda: Permissions(read=True, write=False, exec=False))
    scope_gate: bool = True
    blast_radius_limit: int = 5
