from __future__ import annotations

from llmstitch import Agent
from llmstitch.providers.openai import OpenAIAdapter
from llmstitch.types import TextBlock

from ..router import Session, SYSTEM_PROMPT
from ..tools import make_tools

_TOOL_INSTRUCTION = (
    "if the answer is not already present in this conversation, "
    "you MUST call the available tools to inspect the repository before responding — "
    "do not rely on training knowledge about the codebase"
)


class QueryHandler:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base

    async def handle(self, session: Session, user_input: str) -> str:
        adapter = OpenAIAdapter(api_key=self._api_key, base_url=self._api_base)
        agent = Agent(
            provider=adapter,
            model=self._model,
            system=f"{SYSTEM_PROMPT}\n\n{_TOOL_INSTRUCTION}",
        )
        for t in make_tools(session.working_dir):
            agent.tools.register(t)
        history = await agent.run(user_input)
        last = history[-1]
        if isinstance(last.content, list):
            return "\n".join(b.text for b in last.content if isinstance(b, TextBlock))
        return last.content or ""
