from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import openai
from openai import AsyncOpenAI

from .enrichment import EnrichmentResult, enrich_workspace, _get_git_state, _should_run, _write_scan_state
from ..subagent import SubAgent, SubAgentEvent, SubAgentStartEvent, LogEvent, InferStartEvent, InferDeltaEvent, InferEndEvent, DoneEvent
from ..workspace import scan_workspace


class WsExplorer(SubAgent):
    name = "ws-explorer"
    color = "#008000"

    @property
    def description(self) -> str:
        return "Scan and understand workspace" if self._enrich else "Scan workspace"

    def __init__(
        self,
        working_dir: Path,
        force: bool = False,
        enrich: bool = True,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self._working_dir = working_dir
        self._force = force
        self._enrich = enrich
        self._client = client
        self._model = model
        self.workspace: dict | None = None
        self.enrichment: EnrichmentResult | None = None

    async def run(self) -> AsyncIterator[SubAgentEvent]:
        commit_hash, dirty = await asyncio.to_thread(_get_git_state, self._working_dir)

        if not _should_run(self._working_dir, self._force, commit_hash, dirty):
            cache_path = self._working_dir / ".gekai" / "workspace.json"
            try:
                self.workspace = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            return

        yield SubAgentStartEvent(name=self.name, description=self.description, color=self.color)

        self.workspace = await asyncio.to_thread(scan_workspace, self._working_dir)

        if not self._enrich:
            await asyncio.to_thread(_write_scan_state, self._working_dir, commit_hash, dirty)
            yield DoneEvent()
            return

        queue: asyncio.Queue[SubAgentEvent | None] = asyncio.Queue()

        async def on_file(filename: str, line_count: int) -> None:
            await queue.put(LogEvent(message=f"Read {filename} ({line_count} lines)"))

        async def on_infer_start() -> None:
            await queue.put(InferStartEvent())

        async def on_infer_delta(completion_tokens: int) -> None:
            await queue.put(InferDeltaEvent(completion_tokens=completion_tokens))

        async def on_infer_end(prompt_tokens: int, completion_tokens: int) -> None:
            await queue.put(InferEndEvent(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))

        async def _enrich() -> EnrichmentResult:
            try:
                return await enrich_workspace(
                    self._working_dir,
                    self._client,
                    self._model,
                    commit_hash=commit_hash,
                    dirty=dirty,
                    on_file=on_file,
                    on_infer_start=on_infer_start,
                    on_infer_delta=on_infer_delta,
                    on_infer_end=on_infer_end,
                )
            except openai.APIError as exc:
                await queue.put(LogEvent(message=f"enrichment failed: {exc.message}"))
                return EnrichmentResult(
                    proj_brief="",
                    tech_stack=[],
                    prompt_tokens=None,
                    completion_tokens=None,
                    domain_map={},
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(_enrich())
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

        result = await task
        self.enrichment = result

        yield DoneEvent()
