from . import Subagent

subagent = Subagent(
    name="code-expert",
    namespace="coding",
    description=(
        "general code changes when no specialized subagent fits: features, fixes, tests. "
        "Not for: pure refactors or complexity-reduction passes with no behavior change"
    ),
    mandate="your specialization is general code changes — features, fixes, tests — when no other subagent fits",
    is_fallback=True,
)
