from __future__ import annotations

from openai import AsyncOpenAI

_SYSTEM = (
    "you rewrite a coding-change request so a downstream agent knows exactly which files to touch\n"
    "you are given the original request and a list of files already located for it, each as: path | keywords\n"
    "the files are verified to exist — your job is attribution, not discovery\n"
    "<rules>\n"
    "preserve the user's intent and constraints exactly — never add, drop, or reinterpret what is asked\n"
    "weave the full relative path inline wherever a file clearly corresponds to something the request names\n"
    "  e.g. 'update the passcode dialog to allow 8 chars' + 'src/app/auth/passcoder.tsx | passcode, dialog'\n"
    "       -> \"update 'src/app/auth/passcoder.tsx' to allow 8 chars\"\n"
    "quote paths verbatim from the provided list — never invent, guess, or alter a path\n"
    "for located files you cannot confidently tie to a phrase, list them under a trailing <reference_files> block, "
    "one path per line\n"
    "if every located file is woven inline, omit the <reference_files> block entirely\n"
    "<output>\n"
    "output ONLY the rewritten request, optionally followed by the <reference_files> block\n"
    "no preamble, no explanation, no markdown fences, no closing tags"
)


def _format_entries(entries: list[tuple[str, list[str]]]) -> str:
    lines = []
    for path, keywords in entries:
        kw = ", ".join(keywords)
        lines.append(f"{path} | {kw}" if kw else path)
    return "\n".join(lines)


class PromptRewriter:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def rewrite(
        self,
        request: str,
        entries: list[tuple[str, list[str]]],
    ) -> str:
        user = f"request:\n{request}\n\nlocated files:\n{_format_entries(entries)}"
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("prompt rewriter returned empty output")
        return text
