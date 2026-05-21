import argparse
import asyncio
import logging

logging.getLogger("LiteLLM").setLevel(logging.ERROR)

import pyfiglet
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.live import Live
from rich.text import Text

from . import __version__
from .agent import GekaiAgent
from .commands.exit import ExitCommand
from .commands.registry import CommandRegistry

console = Console()


def _render_response(text: str) -> None:
    console.print(f"[bold cyan]●[/bold cyan] {text}")


def _print_banner() -> None:
    banner = pyfiglet.figlet_format("gekai", font="small_slant").rstrip()
    console.print(f"[cyan]{banner}[/cyan]")
    console.print(f"[bold white]gekai[/bold white] [grey50]v{__version__}[/grey50]")
    console.print()


async def _run(debug: bool = False) -> None:
    agent = GekaiAgent(debug=debug)
    session = agent.start_session()

    registry = CommandRegistry()
    registry.register(ExitCommand())

    pt_session: PromptSession[str] = PromptSession(history=InMemoryHistory())

    _print_banner()

    prompt_message = FormattedText([("ansicyan bold", "❯ ")])

    while True:
        try:
            user_input = await pt_session.prompt_async(prompt_message)
        except (EOFError, KeyboardInterrupt):
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        if stripped.startswith("/"):
            result = await registry.dispatch(stripped)
            if result.output:
                console.print(result.output)
            if result.exit_app:
                break
            continue

        try:
            import time
            chunks: list[str] = []
            token_count = 0
            frame_index = 0
            _spinner = ["|", "/", "-", "\\"]
            _start = time.monotonic()

            def _make_display() -> Text:
                _t = Text()
                _t.append(f"{_spinner[frame_index]} Operating...", style="yellow")
                _count = str(token_count) if token_count < 1000 else f"{token_count / 1000:.1f}k"
                _t.append(f" (↑ {_count} tokens)", style="medium_orchid")
                return _t

            with Live(console=console, refresh_per_second=12, transient=True) as live:
                live.update(_make_display())

                async def _animate() -> None:
                    nonlocal frame_index
                    while True:
                        await asyncio.sleep(0.1)
                        frame_index = (frame_index + 1) % len(_spinner)
                        live.update(_make_display())

                spinner_task = asyncio.create_task(_animate())
                try:
                    async for chunk in agent.process_stream(session, stripped):
                        chunks.append(chunk)
                        token_count += 1
                        live.update(_make_display())
                finally:
                    spinner_task.cancel()
                    try:
                        await spinner_task
                    except asyncio.CancelledError:
                        pass
            _elapsed = time.monotonic() - _start
            _duration = f"{_elapsed:.0f}s" if _elapsed < 60 else f"{_elapsed / 60:.1f}m"
            console.print()
            console.print(f"[grey50]* Operated for {_duration}[/grey50]")
            reply = "".join(chunks)
            console.print()
            _render_response(reply)
            console.print()
        except Exception as exc:
            console.print(f"[red]error:[/red] {exc}")

    console.print("\n[dim]bye[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(prog="gekai")
    parser.add_argument("--debug", action="store_true", help="show intent classification")
    args = parser.parse_args()
    asyncio.run(_run(debug=args.debug))


if __name__ == "__main__":
    main()
