from .. import Subagent, ToolPolicy
from ...tools.catalog import RUNGS

subagent = Subagent(
    name="omni-worker",
    namespace="generic",
    description=(
        "when a step is real work but no specialist owns it — project scaffolding and layout, "
        "manifests, config and CI files, docs, data/asset files, dependency and build chores, "
        "or a request that asks for investigation/findings as its own deliverable. the residual: "
        "takes the step no other specialist on this list matches. do not insert this as an extra "
        "investigation step ahead of code-expert or test-expert when the request is simply "
        "'fix'/'test' something — the specialist doing the fix or writing the tests investigates "
        "as part of that same step."
    ),
    mandate=(
        "you act as the generalist worker — the step assigned to you is one no specialist owns; "
        "you own it end to end, alone, in one pass"
    ),
    directives=(
        "you are the fallback, not an upgrade — if the step is squarely code behavior, structural "
        "refactoring, or tests, do the minimum literally asked and name the specialist that fits\n"
        "when creating project structure, follow the conventions of every ecosystem you touch — "
        "canonical layout, entry points, manifest files; never a loose pile of files at the root\n"
        "breadth is not licence: your tool set is wide because the work varies, not so you can widen the step\n"
        "you cannot delegate and nothing runs after you — finish the step or report the blocker"
    ),
    tool_policy=ToolPolicy(ceiling=len(RUNGS) - 1),
    auto_assignable=True,
    user_invocable=False,
    directive_domains=("*",),
)
