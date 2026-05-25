from __future__ import annotations

from openai import AsyncOpenAI

_NORMALIZER_PROMPT = (
    "you receive a user prompt intended for a coding agent\n"
    "your job is to normalize it:\n"
    "- if the prompt is already correct English, respond exactly: OK\n"
    "- if the prompt is in English but has grammar/spelling issues, respond with the corrected version (preserve meaning)\n"
    "- if the prompt is not in English, translate it to English and add the original language ISO 639-1 code on the last line\n"
    "no preamble, no explanation — only the normalized text (and optional language code)"
)


class PromptNormalizer:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def normalize(self, user_input: str) -> tuple[str, str | None]:
        """Returns (normalized_prompt, source_language_or_None)."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _NORMALIZER_PROMPT},
                {"role": "user", "content": user_input},
            ],
        )
        raw: str = response.choices[0].message.content.strip()
        if raw == "OK":
            return user_input, None
        lines = raw.rsplit("\n", 1)
        if len(lines) == 2 and len(lines[1].strip()) == 2 and lines[1].strip().isalpha():
            return lines[0].strip(), lines[1].strip().upper()
        return raw, None
