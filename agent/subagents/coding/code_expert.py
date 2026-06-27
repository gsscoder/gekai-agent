from .. import Subagent

subagent = Subagent(
    name="code-expert",
    short_description="features, fixes, and decided implementation swaps",
    namespace="coding",
    description=(
        "changes to existing code: features, fixes, and behavior-changing rewrites where the approach "
        "is decided — implementation/algorithm swaps that keep the interface but change output. "
        "Not for: creating a new app or project from scratch (that's main), or pure refactors / "
        "complexity-reduction passes with no behavior change"
    ),
    mandate=(
        "you act as a coding specialist for existing code — features, fixes, and decided "
        "implementation swaps where the interface holds but behavior may change"
    ),
)
