import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .agent import GekaiAgent
from .commands.exit import ExitCommand
from .commands.registry import CommandRegistry
from .settings import load_permissions, prompt_permissions
from .ui import console, make_spinner_display, print_banner, render_operation_summary, render_response
from .workspace import get_git_branch, scan_workspace


def _block_until_esc(stop: threading.Event) -> None:
    if sys.platform == "win32":
        import msvcrt
        while not stop.is_set():
            if msvcrt.kbhit() and msvcrt.getch() == b"\x1b":
                return
            time.sleep(0.05)
    else:
        import select
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not stop.is_set():
                if select.select([sys.stdin], [], [], 0.05)[0] and sys.stdin.read(1) == "\x1b":
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


async def _run(working_dir: Path, debug: bool = False) -> None:
    branch = get_git_branch(working_dir)
    print_banner(__version__, working_dir, branch)

    permissions = load_permissions(working_dir)
    if permissions is None:
        permissions = await prompt_permissions(working_dir)
        if permissions is None:
            console.print("\n[dim]access denied, exiting[/dim]")
            return

    agent = GekaiAgent(working_dir=working_dir, permissions=permissions, debug=debug)

    with Progress(
        SpinnerColumn(),
        TextColumn("[dim]{task.description}[/dim]"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("initializing workspace", total=None)
        workspace = scan_workspace(working_dir, on_step=lambda msg: progress.update(task, description=msg))

    session = agent.start_session(workspace)

    registry = CommandRegistry()
    registry.register(ExitCommand())

    pt_session: PromptSession[str] = PromptSession(history=InMemoryHistory())

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
            start = time.monotonic()
            stop_esc = threading.Event()
            loop = asyncio.get_running_loop()
            state = {"frame": 0, "tokens": 0}
            chunks: list[str] = []

            async def _interact(live: Live) -> None:
                segments = await agent.classify(stripped)
                if debug:
                    for intent, sub in segments:
                        console.print(f"[grey50]debug: {intent.value}: {sub}[/grey50]")
                async for chunk in agent.process_stream(session, segments):
                    chunks.append(chunk)
                    state["tokens"] += 1
                    live.update(make_spinner_display(state["frame"], state["tokens"]))

            with Live(console=console, refresh_per_second=12, transient=True) as live:
                live.update(make_spinner_display(state["frame"], state["tokens"]))

                async def _animate() -> None:
                    while True:
                        await asyncio.sleep(0.1)
                        state["frame"] += 1
                        live.update(make_spinner_display(state["frame"], state["tokens"]))

                spinner_task = asyncio.create_task(_animate())
                interact_task = asyncio.create_task(_interact(live))
                esc_future = loop.run_in_executor(None, _block_until_esc, stop_esc)

                done, _ = await asyncio.wait(
                    {interact_task, esc_future},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                stop_esc.set()
                spinner_task.cancel()
                try:
                    await spinner_task
                except asyncio.CancelledError:
                    pass

            if interact_task in done:
                interact_task.result()
                render_operation_summary(time.monotonic() - start)
                console.print()
                render_response("".join(chunks))
                console.print()
            else:
                interact_task.cancel()
                try:
                    await interact_task
                except asyncio.CancelledError:
                    pass
                console.print("\n[dim]cancelled[/dim]")

        except Exception as exc:
            console.print(f"[red]error:[/red] {exc}")

    console.print("\n[dim]bye[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(prog="gekai")
    parser.add_argument(
        "-d", "--working-dir",
        type=Path,
        default=Path.cwd(),
        metavar="DIR",
        help="working directory (default: current directory)",
    )
    parser.add_argument("--debug", action="store_true", help="show intent classification")
    args = parser.parse_args()
    asyncio.run(_run(working_dir=args.working_dir.resolve(), debug=args.debug))


if __name__ == "__main__":
    main()
