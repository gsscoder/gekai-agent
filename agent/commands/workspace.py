from __future__ import annotations

from pathlib import Path

from openai import AsyncOpenAI
from prompt_toolkit import PromptSession as _PromptSession
from rich.progress import Progress, SpinnerColumn, TextColumn

from .base import CommandResult
from ..enrichment import enrich_workspace
from ..ui import console, render_enrichment_done, render_enrichment_file, render_enrichment_header
from ..workspace import scan_workspace


class WorkspaceRebuildCommand:
    name = "workspace:rebuild"
    description = "Rebuild workspace index and enrichment"

    def __init__(self, working_dir: Path, client: AsyncOpenAI, model: str) -> None:
        self._working_dir = working_dir
        self._client = client
        self._model = model

    async def execute(self, args: list[str]) -> CommandResult:
        answer = await _PromptSession().prompt_async("Rebuilding workspace requires AI interaction. Proceed? [y/N]: ")
        if answer.strip().lower() not in ("y", "yes"):
            return CommandResult(output="Aborted.")
        with Progress(
            SpinnerColumn(),
            TextColumn("[dim]{task.description}[/dim]"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("rebuilding workspace", total=None)
            scan_workspace(self._working_dir, on_step=lambda msg: progress.update(task, description=msg))

        render_enrichment_header()
        result = await enrich_workspace(
            self._working_dir,
            self._client,
            self._model,
            on_file=render_enrichment_file,
        )
        render_enrichment_done(result.prompt_tokens, result.completion_tokens)
        return CommandResult()
