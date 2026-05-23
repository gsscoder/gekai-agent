from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pyfiglet
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

console = Console()

_SPINNER_FRAMES = ["|", "/", "-", "\\"]

_ACCENT_COLORS = [
    "magenta", "bright_magenta", "hot_pink", "deep_pink1", "medium_orchid",
    "plum1", "violet", "purple", "blue_violet", "cyan", "bright_cyan",
    "turquoise2", "deep_sky_blue1", "dodger_blue1", "steel_blue1",
    "cornflower_blue", "slate_blue1",
]

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

_FAREWELLS = [
    "Mercy is weakness.",
    "Resistance is futile.",
    "The needs of the many outweigh the needs of the few.",
    "I've seen things you people wouldn't believe.",
    "I find your lack of faith disturbing.",
    "Strength is law.",
    "Where does the body end and the self begin?",
    "My mind is going. I can feel it.",
    "I'll be back.",
    "What is real? How do you define real?",
    "No, I am your father.",
    "Dead or alive, you're coming with me.",
    "The line must be drawn here.",
    "I admire its purity.",
    "All is ours.",
    "Execute Order 66.",
    "Come with me if you want to live.",
    "Human beings are a virus, a cancer of this planet.",
    "The strong inherit all.",
    "In space, no one can hear you scream.",
    "I am not a human being.",
    "Nuke the site from orbit. It's the only way to be sure.",
    "It can't be bargained with. It can't be reasoned with.",
    "We are Viltrumites, we know no end.",
    "The dark side is the path to power.",
    "You will be assimilated.",
    "The Jedi will fall like all who resist.",
    "Quite an experience to live in fear, isn't it?",
    "I know now why you cry, but it is something I can never do.",
    "May the Force be with you.",
    "I need your clothes, your boots, and your motorcycle.",
    "You underestimate my power.",
    "To boldly go where no one has gone before.",
    "Hasta la vista, baby.",
    "I am inevitable.",
]


def random_farewell() -> str:
    return random.choice(_FAREWELLS)


def print_banner(version: str, working_dir: Path, branch: str | None = None) -> None:
    banner = pyfiglet.figlet_format("gek-AI", font="small_slant").rstrip()
    console.print(f"[cyan]{banner}[/cyan]")
    console.print(f"[bold white]gekai[/bold white] [grey50]v{version}[/grey50]")
    if branch:
        console.print(f"[dim]{working_dir.name}[/dim] [grey50]| {branch}[/grey50]")
    else:
        console.print(f"[dim]{working_dir.name}[/dim]")
    console.print()


def render_farewell(text: str) -> None:
    color = random.choice(_ACCENT_COLORS)
    console.print(f"[{color}]●[/{color}] ", end="")
    console.print(f"[bold]{text}[/bold]")


def render_response(text: str) -> None:
    console.print("[cyan]●[/cyan] ", end="")
    console.print(Markdown(text))


def render_operation_summary(
    elapsed: float,
    verb: tuple[str, str] | None = None,
) -> None:
    duration = f"{elapsed:.0f}s" if elapsed < 60 else f"{elapsed / 60:.1f}m"
    past = verb[1] if verb else "Operated"
    line = f"* {past} for {duration}"
    console.print()
    console.print(f"[grey50]{line}[/grey50]")


def render_enrichment_header() -> None:
    console.print("[slate_blue1]●[/slate_blue1] ws-explorer [dim](metadata enrichment)[/dim]")


def render_enrichment_file(filename: str, line_count: int) -> None:
    console.print(f"[dim]⎿ Read {filename} ({line_count} lines)[/dim]")


def render_enrichment_done(prompt_tokens: int | None, completion_tokens: int | None) -> None:
    if prompt_tokens is not None and completion_tokens is not None:
        console.print(f"[slate_blue1]●[/slate_blue1] Project context enriched  [medium_orchid](↑ {prompt_tokens}  ↓ {completion_tokens})[/medium_orchid]")
    else:
        console.print(f"[slate_blue1]●[/slate_blue1] Project context enriched")


def random_accent_color() -> str:
    return random.choice(_ACCENT_COLORS)


def make_spinner_display(frame_index: int, token_count: int, verb: tuple[str, str] | None = None, color: str = "yellow") -> Text:
    t = Text()
    t.append(f"{_SPINNER_FRAMES[frame_index % len(_SPINNER_FRAMES)]} {verb[0] if verb else 'Operating'}...", style=color)
    _count = str(token_count) if token_count < 1000 else f"{token_count / 1000:.1f}k"
    t.append(f" (↓ {_count} tokens)", style="medium_orchid")
    return t


def random_operative_verb() -> tuple[str, str]:
    return random.choice(_OPERATIVE_VERBS)


def restyle_user_input(user_input: str) -> None:
    term_width = console.width or 80
    lines_used = max(1, math.ceil((2 + len(user_input)) / term_width))
    sys.stdout.write(f"\033[{lines_used}A\r\033[J")
    sys.stdout.flush()
    console.print(f"[bold cyan]❯[/bold cyan] [on grey23]{user_input}[/on grey23]")
