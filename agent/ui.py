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
    ("Ablating", "Ablated"), ("Anastomosing", "Anastomosed"), ("Aspirating", "Aspirated"),
    ("Bypassing", "Bypassed"), ("Cannulating", "Cannulated"), ("Clamping", "Clamped"),
    ("Coagulating", "Coagulated"), ("Cauterizing", "Cauterized"), ("Closing", "Closed"),
    ("Dilating", "Dilated"), ("Dissecting", "Dissected"), ("Draping", "Draped"),
    ("Draining", "Drained"), ("Endoscoping", "Endoscoped"), ("Embolizing", "Embolized"),
    ("Fixating", "Fixated"), ("Grafting", "Grafted"), ("Implanting", "Implanted"),
    ("Incising", "Incised"), ("Irrigating", "Irrigated"), ("Laparoscoping", "Laparoscoped"),
    ("Ligating", "Ligated"), ("Mobilizing", "Mobilized"), ("Operating", "Operated"),
    ("Pinning", "Pinned"), ("Plating", "Plated"), ("Positioning", "Positioned"),
    ("Reimplanting", "Reimplanted"), ("Reinserting", "Reinserted"), ("Reconstructing", "Reconstructed"),
    ("Retracting", "Retracted"), ("Sealing", "Sealed"), ("Separating", "Separated"),
    ("Stapling", "Stapled"), ("Sterilizing", "Sterilized"), ("Stitching", "Stitched"),
    ("Suctioning", "Suctioned"), ("Suturing", "Sutured"), ("Wiring", "Wired"),
    ("Visiting", "Visited"), ("Admitting", "Admitted"), ("Consulting", "Consulted"),
    ("Diagnosing", "Diagnosed"), ("Examining", "Examined"), ("Evaluating", "Evaluated"),
    ("Monitoring", "Monitored"), ("Planning", "Planned"), ("Prepping", "Prepped"),
    ("Recovering", "Recovered"), ("Treating", "Treated"), ("Triaging", "Triaged"),
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


def render_operation_summary(elapsed: float, verb: tuple[str, str] | None = None) -> None:
    duration = f"{elapsed:.0f}s" if elapsed < 60 else f"{elapsed / 60:.1f}m"
    past = verb[1] if verb else "Operated"
    console.print()
    console.print(f"[grey50]* {past} for {duration}[/grey50]")


def make_spinner_display(frame_index: int, token_count: int, verb: tuple[str, str] | None = None) -> Text:
    t = Text()
    t.append(f"{_SPINNER_FRAMES[frame_index % len(_SPINNER_FRAMES)]} {verb[0] if verb else 'Operating'}...", style="yellow")
    _count = str(token_count) if token_count < 1000 else f"{token_count / 1000:.1f}k"
    t.append(f" (↑ {_count} tokens)", style="medium_orchid")
    return t


def random_operative_verb() -> tuple[str, str]:
    return random.choice(_OPERATIVE_VERBS)
