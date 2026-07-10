from __future__ import annotations

import random

from rich.console import Console

console = Console()

_ACCENT_COLORS = [
    "yellow", "ansi_bright_yellow", "gold", "orange", "darkorange",
    "goldenrod", "ansi_bright_red", "coral", "salmon",
]

_OPERATIVE_VERB = ("Operating", "Operated")
OPERATIVE_COLOR = "#5da9e9"

def random_accent_color() -> str:
    return random.choice(_ACCENT_COLORS)


def random_operative_verb() -> tuple[str, str]:
    return _OPERATIVE_VERB
