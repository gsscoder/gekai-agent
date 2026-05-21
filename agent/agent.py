from __future__ import annotations

import os

import litellm
from dotenv import load_dotenv

from .pipeline import PipelineStep, Session

load_dotenv()


class GekaiAgent:
    def __init__(self, pipeline: list[PipelineStep] | None = None) -> None:
        self.pipeline = pipeline or []
        self.model: str = os.environ["GEKAI_DEFAULT_MODEL"]
        self._api_key: str | None = os.environ.get("GEKAI_API_KEY")
        self._api_base: str | None = os.environ.get("GEKAI_BASE_URL")

    def start_session(self) -> Session:
        return Session()

    async def chat(self, session: Session, user_input: str) -> str:
        session.messages.append({"role": "user", "content": user_input})
        response = await litellm.acompletion(
            model=self.model,
            messages=session.messages,
            api_key=self._api_key,
            api_base=self._api_base,
        )
        reply: str = response.choices[0].message.content
        session.messages.append({"role": "assistant", "content": reply})
        return reply
