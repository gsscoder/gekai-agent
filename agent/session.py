from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .directive_audit import file_sha
from .persona import ROOT_SYSTEM_PROMPT
from .permissions import Permissions


@dataclass(frozen=True)
class IngestedFile:
    """A file read whole and verbatim into a model's system base (plan 35
    Concept 1) rather than message history. `sha` uses the same
    `file_sha()` the directive auditor's cache keys off (Phase 2), so a
    future audit can reuse it instead of hashing the text twice."""

    rel_path: str
    text: str
    sha: str


@dataclass
class Session:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict] = field(
        default_factory=lambda: [{"role": "system", "content": ROOT_SYSTEM_PROMPT}]
    )
    working_dir: Path = field(default_factory=Path.cwd)
    permissions: Permissions = field(default_factory=lambda: Permissions(read=True, write=False, exec=False))
    # Root's own project-context file (plan 35 decision 4): auto-read once
    # at session start, singular (not a list — decision 6 stops filename
    # auto-discovery at GEKAI.md), ingested into root's system base for the
    # whole session rather than a message-history read.
    gekai_md: IngestedFile | None = None
