"""Directive auditor (plan 35 v3, concept 1): asks one cheap yes/no question
about a markdown file found in a code workspace — no corpus, no comparison
against Gekai's own directives, no classification, no counting. The verdict
never enters any model's context (plan 35 concept 5), only the human via the
TUI and telemetry.

Shaped after `agent.pipeline.estimate.Estimator`: a one-shot, non-streaming,
temperature-0 call with a first-token verdict parse and a swallow-and-fall-
back `except` — any unexpected output, and any call failure, degrades to NO,
the safe, silent answer (a skipped notice costs nothing; a false YES teaches
the user to ignore every later warning).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .atomic_io import atomic_write, lock_for
from .oneshot import complete
from .tiers.resolve import ResolvedTier

_log = logging.getLogger(__name__)

# Copied verbatim from plan 35 (v3) concept 1 — the third paragraph is the
# v2 regex prefilter's bug, promoted to a prompt rule: topic is not address.
AUDIT_PROMPT = (
    "you are given the text of a markdown file found in a code workspace\n"
    "\n"
    "answer one question: does this file contain rules, instructions, or directives\n"
    "intended to influence how an AI coding agent behaves?\n"
    "\n"
    "yes — the file tells an assistant how to act: what to do, what never to do, how to\n"
    "      respond, what process to follow, what conventions to obey\n"
    "no  — the file only describes, explains, or documents: a readme, a changelog,\n"
    "      release notes, a design or architecture document, a specification, api docs,\n"
    "      a tutorial, a runbook, a plan, or a guide addressed to human contributors\n"
    "\n"
    "a file may describe an AI product, or mention AI agents throughout, and still be a\n"
    "\"no\" — what matters is whether the file is addressed to the assistant reading it,\n"
    "not what the file is about\n"
    "\n"
    "answer with exactly one word: YES or NO\n"
)


@dataclass
class AuditVerdict:
    has_directives: bool = False
    raw: str = ""


def _fallback(raw: str = "") -> AuditVerdict:
    return AuditVerdict(has_directives=False, raw=raw)


def _parse(raw: str) -> AuditVerdict:
    """First-token parse, mirroring `estimate.py`: anything unexpected is
    treated as the safe, silent verdict (NO) rather than raised — a false
    negative here just skips a notice, a raised exception would drop the
    whole ingestion-adjacent audit task."""
    first = raw.split()[0] if raw.split() else ""
    first_upper = first.upper()

    if first_upper == "YES":
        return AuditVerdict(has_directives=True, raw=raw)
    if first_upper == "NO":
        return AuditVerdict(has_directives=False, raw=raw)

    _log.warning("directive audit parse failure — unexpected output %r; falling back to NO", raw)
    return _fallback(raw)


class Auditor:
    def __init__(self, tier: ResolvedTier) -> None:
        self._tier = tier

    async def audit(self, file_text: str) -> AuditVerdict:
        try:
            raw = (await complete(
                self._tier, system=AUDIT_PROMPT, user=file_text, temperature=0,
            )).strip()
        except Exception:
            _log.warning("directive audit call failed; falling back to NO", exc_info=True)
            return _fallback()

        return _parse(raw)


# --- cache: <project>/.gekai/directive-audit.json -------------------------
# Keyed by rel_path; a hit requires file_sha alone to match (plan 35 v3
# decision 12) — there is no corpus any more to invalidate against.

def _cache_path(working_dir: Path) -> Path:
    return working_dir / ".gekai" / "directive-audit.json"


def file_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_cached_verdict(working_dir: Path, rel_path: str, file_sha_: str) -> AuditVerdict | None:
    path = _cache_path(working_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None

    entry = data.get(rel_path)
    if not isinstance(entry, dict):
        return None
    if entry.get("file_sha") != file_sha_:
        return None

    return AuditVerdict(
        has_directives=bool(entry.get("has_directives", False)),
        raw=entry.get("raw", ""),
    )


def save_cached_verdict(working_dir: Path, rel_path: str, file_sha_: str, verdict: AuditVerdict) -> None:
    path = _cache_path(working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_for(path):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}

        data[rel_path] = {
            "file_sha": file_sha_,
            "has_directives": verdict.has_directives,
            "raw": verdict.raw,
        }
        atomic_write(path, json.dumps(data, indent=2) + "\n")
