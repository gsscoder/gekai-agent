from . import AgentProfile

profile = AgentProfile(
    name="code-expert",
    namespace="coding",
    description="general code changes when no specialized profile fits: features, fixes, tests",
    directives="",
    is_fallback=True,
)
