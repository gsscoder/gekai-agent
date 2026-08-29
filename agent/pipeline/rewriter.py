from __future__ import annotations

import re

from ..openai_client import build_openai_client
from ._directives import PIPELINE_DIRECTIVES

_SYSTEM_TEMPLATE = (
    "you rewrite a user request so a downstream agent knows exactly which files to touch, "
    "and you produce a short UI label summarizing the request\n"
    "you are given the original user request and a list of files already located for it, "
    "each as: `path` | keyword1, keyword2, ...\n"
    "the files are verified to exist — your job is attribution, not discovery\n"
    "<user_request>\n"
    "{request}\n"
    "<rules>\n"
    "if no files are provided, output the original request unchanged — do not modify it\n"
    "preserve the user's intent and constraints exactly — never add, drop, or reinterpret what is asked\n"
    "weave the full relative path inline wherever a file clearly corresponds to something the request names\n"
    "  e.g. 'update the passcode dialog to allow 8 chars' + 'src/app/auth/passcoder.tsx | passcode, dialog'\n"
    "       -> \"update `src/app/auth/passcoder.tsx` to allow 8 chars\"\n"
    "wrap paths in backticks — quote verbatim from the provided list — never invent, guess, or alter a path\n"
    "for located files you cannot confidently tie to a phrase, list them under a trailing <reference_files> block, "
    "one backtick-quoted path per line\n"
    "if every located file is woven inline, omit the <reference_files> block entirely\n"
    "<ui_label>\n"
    "summarize the request's intent in 5-7 words, for display in a UI badge\n"
    "never include file names, paths, or backticks in the label\n"
    "<output>\n"
    "first line: the UI label wrapped as <ui_label>...</ui_label>\n"
    "then: the rewritten request, optionally followed by the <reference_files> block\n"
    "no preamble, no explanation, no markdown fences, no other closing tags"
)

_UI_LABEL_RE = re.compile(r"<ui_label>\s*(.*?)\s*</ui_label>\s*\n?(.*)", re.DOTALL)


def _format_entries(entries: list[tuple[str, list[str]]]) -> str:
    lines = []
    for path, keywords in entries:
        kw = ", ".join(keywords)
        lines.append(f"{path} | {kw}" if kw else path)
    return "\n".join(lines)


def _split_ui_label(text: str) -> tuple[str, str]:
    """Split off the leading <ui_label> block; fail-soft — a missing/malformed
    label must never block the turn, only degrade the UI badge to a fallback."""
    match = _UI_LABEL_RE.match(text)
    if not match:
        return "", text
    label, rest = match.group(1).strip(), match.group(2).strip()
    return label, (rest or text)


class PromptRewriter:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._client = build_openai_client(api_key, api_base)

    async def rewrite(
        self,
        request: str,
        entries: list[tuple[str, list[str]]],
    ) -> tuple[str, str]:
        """Returns (rewritten_request, ui_label).

        rewritten_request keeps the original fail-hard contract — any exception,
        including empty output, propagates and blocks the turn (the locator
        already verified the files; this stage only attributes them).

        ui_label is fail-soft — a missing/malformed label degrades to "" rather
        than blocking the turn; it is cosmetic (UI badge text), not load-bearing.
        """
        system = PIPELINE_DIRECTIVES + _SYSTEM_TEMPLATE.format(request=request)
        user = f"located files:\n{_format_entries(entries)}\n\nrewrite the request"
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("prompt rewriter returned empty output")
        ui_label, rewritten = _split_ui_label(text)
        if not rewritten:
            raise ValueError("prompt rewriter returned empty output")
        return rewritten, ui_label
