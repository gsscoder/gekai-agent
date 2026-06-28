from .. import Subagent

subagent = Subagent(
    name="code-expert",
    short_description="features, fixes, and decided implementation swaps",
    namespace="coding",
    description=(
        "substantial or specialized code work: features, fixes, and behavior-changing rewrites where "
        "the approach is decided (implementation/algorithm swaps that keep the interface but change "
        "output). Builds its deliverable whole, structure included. Not for: small or simple apps "
        "(that's main, end to end), or pure refactors / complexity-reduction passes with no behavior "
        "change (that's code-refactorer)"
    ),
    mandate=(
        "you act as a coding specialist for substantial code work — features, fixes, and decided "
        "implementation swaps where the interface holds but behavior may change"
    ),
)
