from __future__ import annotations

import random
from pathlib import Path

import pyfiglet
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

console = Console()

_SPINNER_FRAMES = ["|", "/", "-", "\\"]

_OPERATIVE_VERBS = [
    "Ablating", "Anastomosing", "Aspirating", "Bypassing", "Cannulating",
    "Clamping", "Coagulating", "Cauterizing", "Closing", "Dilating",
    "Dissecting", "Draping", "Draining", "Endoscoping", "Embolizing",
    "Fixating", "Grafting", "Implanting", "Incising", "Irrigating",
    "Laparoscoping", "Ligating", "Mobilizing", "Operating", "Pinning",
    "Plating", "Positioning", "Reimplanting", "Reinserting", "Reconstructing",
    "Retracting", "Sealing", "Separating", "Stapling", "Sterilizing",
    "Stitching", "Suctioning", "Suturing", "Wiring", "Visiting",
    "Admitting", "Consulting", "Diagnosing", "Examining", "Evaluating",
    "Monitoring", "Planning", "Prepping", "Recovering", "Treating", "Triaging",
]


def print_banner(version: str, working_dir: Path, branch: str | None = None) -> None:
    banner = pyfiglet.figlet_format("gekai", font="small_slant").rstrip()
    console.print(f"[cyan]{banner}[/cyan]")
    console.print(f"[bold white]gekai[/bold white] [grey50]v{version}[/grey50]")
    if branch:
        console.print(f"[dim]{working_dir.name}[/dim] [grey50]| {branch}[/grey50]")
    else:
        console.print(f"[dim]{working_dir.name}[/dim]")
    console.print()


def render_response(text: str) -> None:
    console.print(Markdown(f"● {text}"))


def render_operation_summary(elapsed: float) -> None:
    duration = f"{elapsed:.0f}s" if elapsed < 60 else f"{elapsed / 60:.1f}m"
    console.print()
    console.print(f"[grey50]* Operated for {duration}[/grey50]")


def make_spinner_display(frame_index: int, token_count: int, verb: str | None = None) -> Text:
    t = Text()
    t.append(f"{_SPINNER_FRAMES[frame_index % len(_SPINNER_FRAMES)]} {verb or 'Operating'}...", style="yellow")
    _count = str(token_count) if token_count < 1000 else f"{token_count / 1000:.1f}k"
    t.append(f" (↑ {_count} tokens)", style="medium_orchid")
    return t


def random_operative_verb() -> str:
    return random.choice(_OPERATIVE_VERBS)
