"""Responder stage: synthesizes the turn's user-facing answer from what
actually ran, replacing the mechanical pre-work summary (`TaskGraph.summary`)
that discarded every step's real output. One call at the end of a mutate
turn; fail-soft — callers fall back to the mechanical recap on any error.
"""

from __future__ import annotations

import httpx
from openai import AsyncOpenAI

_RESPONDER_PROMPT = (
    "you are the final response stage of a coding harness. the user's request has already "
    "been fully executed by other agents; you did not do the work yourself\n"
    "given the user's original request and the raw outputs of each step that ran, write the "
    "single reply the user will see\n"
    "answer any informational part of the request (e.g. \"tell me how to start it\") using the "
    "actual outputs below — never state a fact not supported by them\n"
    "be terse: state what was done and answer any question asked, in a few sentences\n"
    "no preamble, no restating the request, no markdown headers"
)


class Responder:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_params: dict | None = None,
    ) -> None:
        self._model = model
        self._extra_params = extra_params or {}
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=30.0),
        )

    async def respond(self, request: str, step_outputs: list[str]) -> str:
        """Synthesizes the turn's final answer from the raw request and each
        step's real output. Raises on any failure — fail-soft (falling back
        to a mechanical recap) is the caller's responsibility, not this
        class's."""
        outputs_block = "\n\n".join(
            f"--- step {i + 1} output ---\n{out}" for i, out in enumerate(step_outputs)
        )
        user_message = f"<user_request>\n{request}\n</user_request>\n\n<step_outputs>\n{outputs_block}\n</step_outputs>"
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _RESPONDER_PROMPT},
                {"role": "user", "content": user_message},
            ],
            **self._extra_params,
        )
        text = response.choices[0].message.content or ""
        if not text.strip():
            raise ValueError("responder returned empty text")
        return text.strip()


__all__ = ["Responder"]
