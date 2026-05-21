from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Session:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict] = field(default_factory=list)


@dataclass
class PipelineStep:
    name: str
    handler: Callable
    condition: Callable[[Session], bool] | None = None
    requires_confirmation: bool = False
