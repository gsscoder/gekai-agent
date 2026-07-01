from .. import Subagent

subagent = Subagent(
    name="code-expert",
    short_description="features, fixes, and decided implementation swaps",
    namespace="coding",
    description=(
        "code work by kind: features, fixes, and behavior-changing rewrites where "
        "the approach is decided (implementation/algorithm swaps that keep the interface but change "
        "output). Owns its assigned step's implementation in full. Not for: general/scaffolding/glue "
        "work (that's main), or pure refactors / complexity-reduction passes with no behavior "
        "change (that's code-refactorer)"
    ),
    mandate=(
        "you act as a coding specialist — features, fixes, and decided "
        "implementation swaps where the interface holds but behavior may change"
    ),
)
