from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console

console = Console()


def _render_echo(user_input: str) -> None:
    console.print(f"[bold cyan]◆[/bold cyan] {user_input}")


def main() -> None:
    session: PromptSession[str] = PromptSession(history=InMemoryHistory())

    console.print(
        "[bold cyan]gekai[/bold cyan] [dim]— type your prompt, Ctrl+D to exit[/dim]"
    )
    console.print()

    prompt_message = FormattedText([("ansicyan bold", "❯ ")])

    while True:
        try:
            user_input = session.prompt(prompt_message)
        except (EOFError, KeyboardInterrupt):
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        _render_echo(stripped)

    console.print("\n[dim]bye[/dim]")


if __name__ == "__main__":
    main()
