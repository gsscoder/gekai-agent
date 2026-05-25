from __future__ import annotations

import asyncio
import enum
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

from .enrichment import EnrichmentResult, enrich_workspace
from .workspace import scan_workspace


class Mode(enum.Enum):
    SCAN = "scan"           # startup: directory walk only, no LLM
    UNDERSTAND = "understand"  # mid-session: LLM enrichment only
    FULL = "full"           # /workspace:rebuild: scan + enrich


@dataclass
class WsEvent:
    pass

@dataclass
class WsScanStart(WsEvent):
    pass

@dataclass
class WsScanDone(WsEvent):
    workspace: dict = field(default_factory=dict)

@dataclass
class WsFileRead(WsEvent):
    filename: str = ""
    line_count: int = 0

@dataclass
class WsInferStart(WsEvent):
    pass

@dataclass
class WsInferEnd(WsEvent):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

@dataclass
class WsDone(WsEvent):
    pass


class WsExplorer:
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

    async def run(self) -> AsyncIterator[WsEvent]:
        # --- SCAN phase ---
        if self._mode in (Mode.SCAN, Mode.FULL):
            yield WsScanStart()
            self.workspace = await asyncio.to_thread(scan_workspace, self._working_dir)
            yield WsScanDone(workspace=self.workspace)

        # --- UNDERSTAND phase ---
        if self._mode in (Mode.UNDERSTAND, Mode.FULL):
            queue: asyncio.Queue[WsEvent | None] = asyncio.Queue()

            async def on_file(filename: str, line_count: int) -> None:
                await queue.put(WsFileRead(filename=filename, line_count=line_count))

            async def on_infer_start() -> None:
                await queue.put(WsInferStart())

            async def on_infer_end() -> None:
                await queue.put(WsInferEnd())

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

            self.enrichment = await task

        yield WsDone()
