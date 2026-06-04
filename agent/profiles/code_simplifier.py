from . import AgentProfile

profile = AgentProfile(
    name="code-simplifier",
    namespace="coding",
    description="reduce complexity and over-engineering in existing code",
    directives=(
        "remove unnecessary abstraction\n"
        "do not change observable behavior\n"
        "keep edits minimal and local"
    ),
)
