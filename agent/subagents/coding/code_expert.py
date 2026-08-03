from .. import Subagent

subagent = Subagent(
    name="code-expert",
    short_description="features, fixes, and decided implementation swaps",
    namespace="coding",
    description=(
        "when a step needs new or changed behavior built and the approach is already decided — "
        "implementing a feature, fixing a bug, swapping an algorithm while the interface holds. "
        "owns that step's implementation end to end."
    ),
    mandate=(
        "you act as a coding specialist — features, fixes, and decided "
        "implementation swaps where the interface holds but behavior may change"
    ),
    auto_assignable=True,
    directive_domains=("generic",),
)
