from __future__ import annotations

import pyfiglet
from rich.console import Console
from rich.text import Text

console = Console()

_SPINNER_FRAMES = ["|", "/", "-", "\\"]


def print_banner(version: str) -> None:
    banner = pyfiglet.figlet_format("gekai", font="small_slant").rstrip()
    console.print(f"[cyan]{banner}[/cyan]")
    console.print(f"[bold white]gekai[/bold white] [grey50]v{version}[/grey50]")
    console.print()


def render_response(text: str) -> None:
    console.print(f"[bold cyan]●[/bold cyan] {text}")


def render_operation_summary(elapsed: float) -> None:
    duration = f"{elapsed:.0f}s" if elapsed < 60 else f"{elapsed / 60:.1f}m"
    console.print()
    console.print(f"[grey50]* Operated for {duration}[/grey50]")


def make_spinner_display(frame_index: int, token_count: int) -> Text:
    t = Text()
    t.append(f"{_SPINNER_FRAMES[frame_index % len(_SPINNER_FRAMES)]} Operating...", style="yellow")
    _count = str(token_count) if token_count < 1000 else f"{token_count / 1000:.1f}k"
    t.append(f" (↑ {_count} tokens)", style="medium_orchid")
    return t
