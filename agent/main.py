import argparse
import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console

from .agent import GekaiAgent
from .commands.exit import ExitCommand
from .commands.registry import CommandRegistry

console = Console()


def _render_response(text: str) -> None:
    console.print(f"[bold cyan]◆[/bold cyan] {text}")


async def _run(debug: bool = False) -> None:
    agent = GekaiAgent(debug=debug)
    session = agent.start_session()

    registry = CommandRegistry()
    registry.register(ExitCommand())

    pt_session: PromptSession[str] = PromptSession(history=InMemoryHistory())

    console.print(
        "[bold cyan]gekai[/bold cyan] [dim]— type your prompt, Ctrl+D to exit[/dim]"
    )
    console.print()

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
            reply = await agent.process(session, stripped)
            _render_response(reply)
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
