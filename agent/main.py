import argparse
import asyncio
import logging
import time
from pathlib import Path

logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM.utils").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from rich.live import Live

from . import __version__
from .agent import GekaiAgent
from .commands.exit import ExitCommand
from .commands.registry import CommandRegistry
from .settings import load_permissions, prompt_permissions
from .ui import console, make_spinner_display, print_banner, render_operation_summary, render_response
from .workspace import get_git_branch


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
    session = agent.start_session()

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
            segments = await agent.classify(stripped)

            if debug:
                for intent, sub in segments:
                    console.print(f"[grey50]debug: {intent.value}: {sub}[/grey50]")

            chunks: list[str] = []
            token_count = 0
            frame_index = 0

            with Live(console=console, refresh_per_second=12, transient=True) as live:
                live.update(make_spinner_display(frame_index, token_count))

                async def _animate() -> None:
                    nonlocal frame_index
                    while True:
                        await asyncio.sleep(0.1)
                        frame_index += 1
                        live.update(make_spinner_display(frame_index, token_count))

                spinner_task = asyncio.create_task(_animate())
                try:
                    async for chunk in agent.process_stream(session, segments):
                        chunks.append(chunk)
                        token_count += 1
                        live.update(make_spinner_display(frame_index, token_count))
                finally:
                    spinner_task.cancel()
                    try:
                        await spinner_task
                    except asyncio.CancelledError:
                        pass

            render_operation_summary(time.monotonic() - start)
            console.print()
            render_response("".join(chunks))
            console.print()

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
