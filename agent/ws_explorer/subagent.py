from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import openai
from openai import AsyncOpenAI

from .enrichment import EnrichmentResult, enrich_workspace, _get_git_state
from ..subagent import SubAgent, SubAgentEvent, SubAgentStartEvent, LogEvent, InferStartEvent, InferDeltaEvent, InferEndEvent, DoneEvent, StatusUpdateEvent
from ..workspace import scan_workspace


class WsExplorer(SubAgent):
    name = "ws-explorer"
    color = "#008000"

    @property
    def description(self) -> str:
        cache_path = self._working_dir / ".gekai" / "workspace.json"
        return "Onboarding workspace" if not cache_path.exists() else "Scan workspace"

    def __init__(
        self,
        working_dir: Path,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self._working_dir = working_dir
        self._client = client
        self._model = model
        self.workspace: dict | None = None
        self.enrichment: EnrichmentResult | None = None

    async def run(self) -> AsyncIterator[SubAgentEvent]:
        commit_hash, dirty = await asyncio.to_thread(_get_git_state, self._working_dir)

        yield SubAgentStartEvent(name=self.name, description=self.description, color=self.color)

        self.workspace = await asyncio.to_thread(scan_workspace, self._working_dir)

        # Estimate total steps: one per file that will trigger on_file, plus 2 LLM calls
        ws = self.workspace or {}
        manifest_count = len([p for p in ws.get("projects", []) if p.get("manifest")])
        ai_doc_count = len(ws.get("ai_instructions", []))
        has_readme = (self._working_dir / "README.md").exists()
        estimated_files = manifest_count + ai_doc_count + (1 if has_readme else 0)
        total_steps = max(estimated_files, 1) + 2  # +2 for two parallel LLM calls
        current_step = 0
        yield StatusUpdateEvent(progress=0, total=total_steps)
        await asyncio.sleep(0)  # let TUI render the 0% bar before enrichment starts

        queue: asyncio.Queue[SubAgentEvent | None] = asyncio.Queue()

        async def on_file(filename: str, line_count: int) -> None:
            await queue.put(LogEvent(message=f"Read {filename} ({line_count} lines)"))

        async def on_infer_start() -> None:
            await queue.put(InferStartEvent())

        async def on_infer_delta(completion_tokens: int) -> None:
            await queue.put(InferDeltaEvent(completion_tokens=completion_tokens))

        async def on_infer_end(prompt_tokens: int, completion_tokens: int) -> None:
            await queue.put(InferEndEvent(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))

        async def _run_enrich() -> EnrichmentResult:
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

        task = asyncio.create_task(_run_enrich())
        while True:
            event = await queue.get()
            if event is None:
                break
            if isinstance(event, LogEvent):
                current_step = min(current_step + 1, total_steps - 2)
                yield StatusUpdateEvent(progress=current_step, total=total_steps)
            elif isinstance(event, InferEndEvent):
                current_step = min(current_step + 1, total_steps)
                yield StatusUpdateEvent(progress=current_step, total=total_steps)
            yield event

        result = await task
        self.enrichment = result

        yield DoneEvent()
