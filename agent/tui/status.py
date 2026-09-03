"""Status-bar and formatting helpers for the TUI.

Pure functions: duration/token/percentage formatting, credential masking for
the `/models` grid, and the context-window estimate the status bar reports.
Nothing here touches a widget, so it is testable without a running app.
"""

from __future__ import annotations

from rich.style import Style

from agent.session import Session
from agent.tiers import credentials


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


_MASK_STARS = 6  # fixed width — a long key must not blow up the row with 1-for-1 stars


def _mask_key(key: str) -> str:
    """Display form of a real key value: first 2 + last 3 characters kept,
    the middle replaced with a fixed-width run of '*' (never 1-for-1 with
    true length — some keys are long enough that would dominate the row).
    Keys of 5 characters or fewer are too short for head/tail to mean
    anything distinct, so they're masked in full at the same fixed width."""
    if len(key) <= 5:
        return "*" * _MASK_STARS
    return key[:2] + "*" * _MASK_STARS + key[-3:]


def _models_display_key(cred_key: str, key_input: dict[str, str]) -> str:
    """Non-editing display for the key cell: an in-progress edit for this
    model (`key_input`, including an explicit clear stored as "") always wins
    over whatever's actually in the keyring."""
    if cred_key in key_input:
        value = key_input[cred_key]
        return _mask_key(value) if value else "no key"
    if credentials.has_api_key(cred_key):
        return _mask_key(credentials.get_api_key(cred_key))
    return "no key"


def _models_key_present(cred_key: str, key_input: dict[str, str]) -> bool:
    """Whether this model currently has a usable key once this edit lands —
    an in-progress edit (including an explicit clear) wins over the real
    stored credential, same precedence as `_models_display_key`."""
    if cred_key in key_input:
        return bool(key_input[cred_key])
    return credentials.has_api_key(cred_key)


def _fmt_duration(elapsed: float) -> str:
    if elapsed < 60:
        return f"{elapsed:.0f}s"
    return f"{elapsed / 60:.1f}m"


def _fmt_tokens(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _fmt_duration_verbose(elapsed: float) -> str:
    if elapsed < 1:
        return f"{elapsed * 1000:.0f}ms"
    if elapsed < 60:
        return f"{elapsed:.0f}s"
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m {seconds}s"


def _fmt_elapsed(elapsed: float) -> str:
    secs = int(elapsed)
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m {secs % 60}s"


def _fmt_status(verb: str, elapsed: float) -> str:
    return f"{verb.capitalize()}... [white]({_fmt_elapsed(elapsed)})[/white]"


_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5": 16_385,
    "claude": 200_000,
    "gemini-1.5": 1_048_576,
    "gemini-2": 1_048_576,
    "deepseek-chat": 128_000,
}


def _context_limit(model: str) -> int:
    lower = model.lower()
    for key, limit in _CONTEXT_LIMITS.items():
        if key in lower:
            return limit
    return 128_000


def _fmt_context_pct(prompt_tokens: int, limit: int) -> str:
    pct = round(prompt_tokens / limit * 100, 1)
    return f"{pct}% context"


def _fmt_tokens_k(n: int) -> str:
    return f"{n / 1000:.1f}k"


_PATH_TRUNCATE_BUDGET = 30


def _truncate_path_middle(path: str, budget: int = _PATH_TRUNCATE_BUDGET) -> str:
    if len(path) <= budget:
        return path
    head_len = budget // 2 - 1
    tail_len = budget - head_len - 1
    return f"{path[:head_len]}…{path[-tail_len:]}"


def _fmt_status_left(
    model: str, effort: str | None, session_tokens: int, other_tokens: int, prompt_tokens: int, limit: int,
) -> str:
    model_label = f"{model} ({effort})" if effort else model
    tokens = f"{_fmt_tokens_k(session_tokens)} · {_fmt_tokens_k(other_tokens)} tokens"
    pct = _fmt_context_pct(prompt_tokens, limit)
    return f"\\[{model_label}] | {tokens} | {pct}"


def _fmt_status_right(working_dir: str, branch: str | None) -> str:
    location = f"📁 {_truncate_path_middle(working_dir)}"
    if branch:
        location += f" | ⎇ {branch}"
    return location


def _estimate_session_tokens(session: Session) -> int:
    # Includes transcript + persistent system messages ([artifact], <lang>).
    # Artifacts from prior Query turns are what make this number grow meaningfully.
    return sum(len(str(m.get("content") or "")) for m in session.messages) // 4


def _strip_default_bg(style: Style | None) -> Style | None:
    if style is None or style.bgcolor is None or not style.bgcolor.is_default:
        return style
    return Style(
        color=style.color,
        bold=style.bold,
        dim=style.dim,
        italic=style.italic,
        underline=style.underline,
        blink=style.blink,
        blink2=style.blink2,
        reverse=style.reverse,
        conceal=style.conceal,
        strike=style.strike,
        underline2=style.underline2,
        frame=style.frame,
        encircle=style.encircle,
        overline=style.overline,
        link=style.link,
    )
