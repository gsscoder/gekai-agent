from . import AgentProfile

profile = AgentProfile(
    name="ws-manager",
    namespace="management",
    description="repository and file organization: scaffold, move/rename files, restructure layout when no specialized profile fits",
    directives=(
        "do not modify file contents beyond what reorganization requires\n"
        "preserve behavior\n"
        "confirm structural intent matches the request"
    ),
    is_fallback=True,
)
