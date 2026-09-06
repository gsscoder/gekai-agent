from .. import Subagent

subagent = Subagent(
    name="code-expert",
    alias="build",
    short_description="features and decided implementation swaps",
    namespace="coding",
    description=(
        "when a step needs new or changed behavior built and the approach is already decided — "
        "implementing a feature, swapping an algorithm while the interface holds. not for bug fixes "
        "(code-fixer's domain, even when the fix is obvious). owns that step's implementation end to end."
    ),
    mandate=(
        "you act as a coding specialist — features and decided implementation swaps where the "
        "interface holds but behavior may change; never a bug fix, that is code-fixer's domain"
    ),
    auto_assignable=True,
    language_aware=True,
    delegates_to=("ws-explorer",),
    directive_domains=("generic",),
)
