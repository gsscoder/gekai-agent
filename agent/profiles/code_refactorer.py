from . import AgentProfile

profile = AgentProfile(
    name="code-refactorer",
    namespace="coding",
    description="restructure existing code without changing behavior: renames, extractions, dedup",
    directives=(
        "do not invent features\n"
        "do not alter behavior beyond the request\n"
        "preserve public signatures unless asked"
    ),
)
