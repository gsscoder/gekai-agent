"""Model-agnostic cleanup of assistant output text.

Prose directives asking the model to self-format ("never use consecutive
blank lines", "never output horizontal separators") only hold reliably for
whichever model they were tuned against (DeepSeek, during development) —
this replaces that prompt-engineering with deterministic post-processing so
output layout no longer depends on which model is plugged in.

Fenced code blocks are left untouched: blank lines inside real source
snippets can be legitimate style, so mechanical stripping only runs on the
surrounding prose.
"""

from __future__ import annotations

import re

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_SEPARATOR_LINE = re.compile(r"^[ \t]*[-─=*]{3,}[ \t]*$\n?", re.MULTILINE)
_INTRO_BLANK = re.compile(r"^([^\n]*:)\n[ \t]*\n+", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")

_PLACEHOLDER = "\x00FENCE{}\x00"


def clean_output(text: str) -> str:
    if not text:
        return text

    fences: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        fences.append(match.group(0))
        return _PLACEHOLDER.format(len(fences) - 1)

    working = _FENCE.sub(_stash, text)
    working = _SEPARATOR_LINE.sub("", working)
    working = _INTRO_BLANK.sub(r"\1\n", working)
    working = _BLANK_RUN.sub("\n\n", working)
    working = working.strip("\n")

    for i, fence in enumerate(fences):
        working = working.replace(_PLACEHOLDER.format(i), fence)
    return working


def _demo() -> None:
    assert clean_output("") == ""
    assert clean_output("\n\nhello") == "hello"
    assert clean_output("a\n\n\n\nb") == "a\n\nb"
    assert clean_output("intro:\n\n\nbody") == "intro:\nbody"
    assert clean_output("a\n---\nb") == "a\nb"
    assert clean_output("a\n***\n\nb") == "a\n\nb"
    code = "```\ndef f():\n\n    pass\n```"
    assert clean_output(code) == code
    assert clean_output(f"before\n\n\n{code}\n\n\nafter") == f"before\n\n{code}\n\nafter"
    print("ok")


if __name__ == "__main__":
    _demo()
