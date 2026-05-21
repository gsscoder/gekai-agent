import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console

from .agent import GekaiAgent

console = Console()


def _render_response(text: str) -> None:
    console.print(f"[bold cyan]◆[/bold cyan] {text}")


async def _run() -> None:
    agent = GekaiAgent()
    session = agent.start_session()

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

        try:
            reply = await agent.chat(session, stripped)
            _render_response(reply)
        except Exception as exc:
            console.print(f"[red]error:[/red] {exc}")

    console.print("\n[dim]bye[/dim]")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
