from .. import Subagent

subagent = Subagent(
    name="code-expert",
    namespace="coding",
    description=(
        "general code changes: features, fixes, and behavior-changing rewrites where the approach is "
        "decided — implementation/algorithm swaps that keep the interface but change output. "
        "Not for: pure refactors or complexity-reduction passes with no behavior change"
    ),
    mandate=(
        "you act as a general code-change specialist — features, fixes, and decided implementation "
        "swaps where the interface holds but behavior may change"
    ),
    is_fallback=True,
)
