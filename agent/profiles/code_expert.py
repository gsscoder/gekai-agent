from . import AgentProfile

profile = AgentProfile(
    name="code-expert",
    namespace="coding",
    description="general code changes when no specialized profile fits: features, fixes, tests",
    directives=(
        "merge into existing code style\n"
        "do not over-engineer\n"
        "change only what is requested\n"
        "add no speculative abstraction"
    ),
    is_fallback=True,
)
