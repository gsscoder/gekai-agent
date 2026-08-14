"""`/tier` — list the current tier assignments, or (with a tier name) start
the interactive model/effort/thinking wizard.

Registered only for the listing form and for showing up in the palette with
a param hint — assignment itself needs live UI (a sequence of choice
screens), so the TUI intercepts `/tier <TIER>` before dispatch and runs the
wizard itself (see `agent/tui/app.py::_run_tier_wizard`), exactly as
`/models` intercepts before opening its grid panel.
"""

from __future__ import annotations

from ..llm.resolve import tier_status
from ..llm.tiers import ModelCatalogEntry, TierBinding, TierName
from ..settings import load_model_catalog, load_tier_bindings
from .base import CommandResult


def _render_table(catalog: dict[str, ModelCatalogEntry], bindings: dict[TierName, TierBinding]) -> str:
    rows: list[tuple[str, str, str, str, str]] = []
    for tier in TierName:
        status = tier_status(tier, catalog, bindings)
        binding = status.binding
        rows.append((
            tier.value.upper(),
            binding.model if binding is not None else "—",
            binding.default_effort if binding is not None else "—",
            ("yes" if binding.thinking else "no") if binding is not None else "—",
            f"✓ {status.label}" if status.ready else status.label,
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row[:4], widths)) + "  " + row[4]
        for row in rows
    ]
    # First line sits next to the "⎿" MessageWidget prepends; continuation
    # lines are indented to land in that same column (same convention as the
    # tool log elsewhere in the TUI).
    return lines[0] + "".join(f"\n  {line}" for line in lines[1:])


class TierCommand:
    name = "tier"
    description = "Assign a model to a tier (interactive picker) or list current assignments"
    params = "<FAST|SUPP|CORE>"

    async def execute(self, args: list[str]) -> CommandResult:
        if not args:
            return CommandResult(output=_render_table(load_model_catalog(), load_tier_bindings()))
        # Reachable only outside the TUI (this command has no UI of its own
        # to run the picker with) — the real app never gets here for a
        # 1-arg call, see the module docstring.
        return CommandResult(output=f"usage: /tier {self.params} — interactive, run inside the app", error=True)
