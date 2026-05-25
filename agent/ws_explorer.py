from __future__ import annotations

import asyncio
import enum
from collections.abc import AsyncIterator
from pathlib import Path

from openai import AsyncOpenAI

from .enrichment import EnrichmentResult, enrich_workspace
from .subagent import SubAgent, SubAgentEvent, SubAgentStartEvent, LogEvent, InferStartEvent, InferEndEvent, DoneEvent
from .workspace import scan_workspace


class Mode(enum.Enum):
    SCAN = "scan"           # startup: directory walk only, no LLM
    UNDERSTAND = "understand"  # mid-session: LLM enrichment only
    FULL = "full"           # /workspace:rebuild: scan + enrich


class WsExplorer(SubAgent):
    name = "ws-explorer"

    def __init__(
        self,
        working_dir: Path,
        mode: Mode,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self._working_dir = working_dir
        self._mode = mode
        self._client = client
        self._model = model
        self.workspace: dict | None = None
        self.enrichment: EnrichmentResult | None = None

    async def run(self) -> AsyncIterator[SubAgentEvent]:
        yield SubAgentStartEvent(name=self.name)

        # --- SCAN phase ---
        if self._mode in (Mode.SCAN, Mode.FULL):
            yield LogEvent(message="scanning workspace...")
            self.workspace = await asyncio.to_thread(scan_workspace, self._working_dir)

        # --- UNDERSTAND phase ---
        if self._mode in (Mode.UNDERSTAND, Mode.FULL):
            queue: asyncio.Queue[SubAgentEvent | None] = asyncio.Queue()

            async def on_file(filename: str, line_count: int) -> None:
                await queue.put(LogEvent(message=f"Read {filename} ({line_count} lines)"))

            async def on_infer_start() -> None:
                await queue.put(InferStartEvent())

            async def on_infer_end() -> None:
                pass  # tokens only available after enrich completes

            async def _enrich() -> EnrichmentResult:
                result = await enrich_workspace(
                    self._working_dir,
                    self._client,
                    self._model,
                    on_file=on_file,
                    on_infer_start=on_infer_start,
                    on_infer_end=on_infer_end,
                )
                await queue.put(None)  # sentinel
                return result

            task = asyncio.create_task(_enrich())
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event

            result = await task
            self.enrichment = result
            yield InferEndEvent(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )

        yield DoneEvent()
